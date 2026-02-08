#!/usr/bin/env python3
"""
Generate t-SNE visualization and submission.csv from trained model.

Usage:
    python generate_outputs.py  # Use default checkpoint
    python generate_outputs.py --checkpoint /workspace/checkpoints/best_model_phase1.pt --phase 1
    python generate_outputs.py --checkpoint /workspace/checkpoints/best_model_phase2.pt --phase 2
"""
import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

# Use non-interactive backend
matplotlib.use('Agg')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_default_config
from src.data.preprocessor import LoanDataPreprocessor
from src.models.model import PrototypeNetwork, create_model_from_config


def load_model_and_preprocessor(
    checkpoint_dir: str = '/workspace/checkpoints',
    model_path: str = None,
    force_phase: int = None
):
    """
    Load trained model and preprocessor.
    
    Args:
        checkpoint_dir: Directory containing preprocessor.pkl
        model_path: Path to specific model checkpoint (default: best_model.pt)
        force_phase: Force model to specific phase (1 or 2)
    """
    # Load preprocessor
    preprocessor_path = os.path.join(checkpoint_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Get config
    config = get_default_config()
    config.categorical_cardinalities = preprocessor.get_cardinalities()
    
    # Load model weights
    if model_path is None:
        model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    print(f"  Loading checkpoint: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Infer n_prototypes from state_dict
    if 'prototype_layer.prototypes' in state_dict:
        n_prototypes = state_dict['prototype_layer.prototypes'].shape[0]
        config.n_prototypes = n_prototypes
        print(f"  Inferred n_prototypes: {n_prototypes}")
    
    # Infer global prototypes
    if 'global_prototype_layer.prototypes' in state_dict:
        config.n_global_prototypes = state_dict['global_prototype_layer.prototypes'].shape[0]
        print(f"  Inferred n_global_prototypes: {config.n_global_prototypes}")
    
    # Create model
    model = create_model_from_config(config)
    
    # Set phase
    if force_phase is not None:
        phase = force_phase
    else:
        phase = checkpoint.get('phase', 2)
    
    if phase == 2:
        model.set_second_phase()
    else:
        model.set_first_phase()
    print(f"  Model phase: {phase}")
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, preprocessor, config


def get_latent_representations(model, X_num, X_cat, batch_size=256):
    """Get latent representations from model."""
    model.eval()
    latents = []
    
    n_samples = len(X_num)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_num = torch.tensor(X_num[i:i+batch_size], dtype=torch.float32)
            batch_cat = torch.tensor(X_cat[i:i+batch_size], dtype=torch.long)
            
            z = model.encoder(batch_num, batch_cat)
            latents.append(z.numpy())
    
    return np.concatenate(latents, axis=0)


def create_tsne_visualization(
    model, 
    train_num, train_cat, train_labels,
    output_path: str = 'tsne_visualization.png',
    n_samples: int = 5000,
    perplexity: int = 30,
    random_state: int = 42
):
    """Create t-SNE visualization of latent space with prototypes."""
    print(f"Creating t-SNE visualization with {n_samples} samples...")
    
    # Sample training data
    np.random.seed(random_state)
    indices = np.random.choice(len(train_num), min(n_samples, len(train_num)), replace=False)
    
    sampled_num = train_num[indices]
    sampled_cat = train_cat[indices]
    sampled_labels = train_labels[indices]
    
    # Get latent representations
    print("  Getting latent representations...")
    latents = get_latent_representations(model, sampled_num, sampled_cat)
    
    # Get prototype vectors based on phase
    if model.phase == 2:
        prototypes = model.global_prototype_layer.prototypes.detach().numpy()
        proto_label = 'Global Prototypes'
    else:
        prototypes = model.prototype_layer.prototypes.detach().numpy()
        proto_label = 'Local Prototypes'
    n_prototypes = len(prototypes)
    
    # Combine latents and prototypes for t-SNE
    all_vectors = np.vstack([latents, prototypes])
    
    # Run t-SNE
    print("  Running t-SNE (this may take a few minutes)...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        learning_rate='auto',
        init='pca'
    )
    embeddings = tsne.fit_transform(all_vectors)
    
    # Split embeddings back
    data_embeddings = embeddings[:-n_prototypes]
    proto_embeddings = embeddings[-n_prototypes:]
    
    # Calculate model predictions for decision boundary
    print("  Calculating decision boundary...")
    from sklearn.neighbors import KNeighborsClassifier
    
    z_tensor = torch.tensor(latents, dtype=torch.float32)
    with torch.no_grad():
        if model.phase == 2:
            coordinates = model.projector(z_tensor)
            p_space = model.global_prototype_layer(coordinates)
            logits = model.pspace_classifier(p_space).squeeze(-1)
        else:
            similarities = model.prototype_layer(z_tensor)
            logits = model.classifier(similarities)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)
        
    # Train a simple classifier on 2D embeddings to visualize boundary
    clf = KNeighborsClassifier(n_neighbors=50)
    clf.fit(data_embeddings, preds)
    
    # Print prototype coordinates for debugging
    print(f"  Visualizing {n_prototypes} prototypes ({proto_label})")
    for i, (x, y) in enumerate(proto_embeddings):
        print(f"    P{i+1}: ({x:.4f}, {y:.4f})")
    
    # Create grid
    x_min, x_max = data_embeddings[:, 0].min() - 5, data_embeddings[:, 0].max() + 5
    y_min, y_max = data_embeddings[:, 1].min() - 5, data_embeddings[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 0.5))
    
    # Predict on grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create visualization
    print("  Creating plot...")
    fig, ax = plt.subplots(figsize=(16, 12))  # Slightly wider for legend
    
    # Plot Decision Boundary (Background)
    custom_cmap = matplotlib.colors.ListedColormap(['#e8f5e9', '#ffebee'])
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=custom_cmap)
    ax.contour(xx, yy, Z, colors=['#999999'], linewidths=0.5, alpha=0.5)
    
    # Plot data points
    colors = ['#2ecc71', '#e74c3c']  # Green for non-default, Red for default
    labels_text = ['Non-Default (True)', 'Default (True)']
    
    for label in [0, 1]:
        mask = sampled_labels == label
        ax.scatter(
            data_embeddings[mask, 0],
            data_embeddings[mask, 1],
            c=colors[label],
            label=labels_text[label],
            alpha=0.6,
            edgecolors='w',
            linewidths=0.5,
            s=30
        )
    
    # Plot prototypes
    ax.scatter(
        proto_embeddings[:, 0],
        proto_embeddings[:, 1],
        c='#9b59b6',  # Purple
        marker='*',
        s=600,
        edgecolors='white',
        linewidths=2,
        label=proto_label,
        zorder=100
    )
    
    # Add prototype labels
    for i, (x, y) in enumerate(proto_embeddings):
        ax.annotate(
            f'P{i+1}',  # 1-based indexing
            (x, y),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontsize=12,
            fontweight='bold',
            color='#8e44ad',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
        )
    
    phase_str = "Phase 2 (P-Space)" if model.phase == 2 else "Phase 1 (Local)"
    ax.set_title(f't-SNE Latent Space - {phase_str}', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', label='Non-Default (Actual)', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='Default (Actual)', markersize=10),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#9b59b6', label=proto_label, markersize=15),
        matplotlib.patches.Patch(facecolor='#e8f5e9', label='Predicted: Non-Default Region'),
        matplotlib.patches.Patch(facecolor='#ffebee', label='Predicted: Default Region'),
    ]
    # Move legend outside
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved t-SNE visualization to {output_path}")
    return output_path


def generate_submission(
    model, 
    preprocessor,
    test_path: str,
    output_path: str = 'submission.csv',
    batch_size: int = 256
):
    """Generate submission.csv with predictions for test set."""
    print("Generating submission.csv...")
    
    # Load test data
    test_df = pd.read_csv(test_path)
    
    # Get the id column
    if 'id' in test_df.columns:
        ids = test_df['id'].values
    else:
        ids = np.arange(58645, 58645 + len(test_df))
    
    # Preprocess test data
    X_num, X_cat = preprocessor.transform(test_df)
    
    # Generate predictions
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_num), batch_size):
            batch_num = torch.tensor(X_num[i:i+batch_size], dtype=torch.float32)
            batch_cat = torch.tensor(X_cat[i:i+batch_size], dtype=torch.long)
            
            output = model(batch_num, batch_cat, return_all=False)
            probs = output['probabilities'].numpy()
            predictions.extend(probs.tolist())
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': ids,
        'loan_status': predictions
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"  Saved submission to {output_path}")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Mean prediction: {np.mean(predictions):.4f}")
    print(f"  Std prediction: {np.std(predictions):.4f}")
    
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description='Generate t-SNE visualization and submission')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: best_model.pt)')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=None,
                        help='Force model phase (1 or 2). Default: use phase from checkpoint.')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Suffix for output files (e.g., "_phase1" for tsne_visualization_phase1.png)')
    parser.add_argument('--checkpoint-dir', type=str, default='/workspace/checkpoints',
                        help='Directory containing preprocessor.pkl')
    parser.add_argument('--skip-submission', action='store_true',
                        help='Skip submission.csv generation')
    return parser.parse_args()


def main():
    args = parse_args()
    
    suffix = args.output_suffix
    
    print("=" * 60)
    print("GENERATING T-SNE VISUALIZATION AND SUBMISSION")
    print("=" * 60)
    
    # Load model and preprocessor
    print("\n[1/4] Loading model and preprocessor...")
    model, preprocessor, config = load_model_and_preprocessor(
        checkpoint_dir=args.checkpoint_dir,
        model_path=args.checkpoint,
        force_phase=args.phase
    )
    print("  Model loaded successfully!")
    
    # Load training data for t-SNE
    print("\n[2/4] Loading training data...")
    train_path = '/workspace/data/train.csv'
    train_df = pd.read_csv(train_path)
    train_labels = train_df['loan_status'].values
    X_num_train, X_cat_train = preprocessor.transform(train_df)
    print(f"  Loaded {len(train_df)} training samples")
    
    # Create t-SNE visualization
    print("\n[3/4] Creating t-SNE visualization...")
    tsne_output = f'/workspace/underwriting-explainable-ai/tsne_visualization{suffix}.png'
    tsne_path = create_tsne_visualization(
        model, X_num_train, X_cat_train, train_labels,
        output_path=tsne_output,
        n_samples=5000
    )
    
    # Generate submission
    if not args.skip_submission:
        print("\n[4/4] Generating submission...")
        test_path = '/workspace/data/test.csv'
        submission_output = f'/workspace/underwriting-explainable-ai/submission{suffix}.csv'
        submission_path = generate_submission(
            model, preprocessor, test_path,
            output_path=submission_output
        )
    else:
        print("\n[4/4] Skipping submission generation...")
        submission_path = None
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - t-SNE: {tsne_path}")
    if submission_path:
        print(f"  - Submission: {submission_path}")


if __name__ == '__main__':
    main()

