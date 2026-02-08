#!/usr/bin/env python3
"""
Generate t-SNE visualization and submission.csv from trained model.
"""
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
from src.models.model import (
    PrototypeNetwork, 
    create_model_from_config,
    create_class_balanced_model_from_config,
    create_ptarl_model_from_config
)


def load_model_and_preprocessor(checkpoint_dir: str = '/workspace/checkpoints'):
    """Load trained model and preprocessor."""
    # Load preprocessor
    preprocessor_path = os.path.join(checkpoint_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Get config
    config = get_default_config()
    config.categorical_cardinalities = preprocessor.get_cardinalities()
    
    # Load model weights first to check architecture and dimensions
    model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Infer n_prototypes from state_dict
    if 'prototype_layer.prototypes' in state_dict:
        n_prototypes = state_dict['prototype_layer.prototypes'].shape[0]
        config.n_prototypes = n_prototypes
        print(f"  Inferred n_prototypes: {n_prototypes}")
    
    # Check model type
    is_ptarl = 'global_prototype_layer.prototypes' in state_dict
    is_class_balanced = any('prototype_classes' in k for k in state_dict.keys())
    
    if is_ptarl:
        print("  Detected PTaRL structure")
        # Infer global/local prototypes
        if 'global_prototype_layer.prototypes' in state_dict:
            config.n_global_prototypes = state_dict['global_prototype_layer.prototypes'].shape[0]
            
        model = create_ptarl_model_from_config(config)
        # Force phase to 2 for evaluation/generation if it's a PTaRL model
        if hasattr(model, 'set_second_phase'):
            model.set_second_phase()
            
    elif is_class_balanced:
        print("  Detected ClassBalancedPrototypeNetwork structure")
        # Infer n_prototypes_per_class (Class 0) from prototype_classes buffer
        proto_classes = state_dict['prototype_layer.prototype_classes']
        n_class0 = (proto_classes == 0).sum().item()
        config.n_prototypes_per_class = n_class0
        print(f"  Inferred n_prototypes_per_class (Class 0): {n_class0}")
        
        model = create_class_balanced_model_from_config(config)
    else:
        print("  Detected standard PrototypeNetwork structure")
        model = create_model_from_config(config)
    
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
    
    # Get prototype vectors
    prototypes = model.prototype_layer.prototypes.detach().numpy()
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
    
    # We need predictions for the sampled points
    # latents is already computed
    z_tensor = torch.tensor(latents, dtype=torch.float32)
    with torch.no_grad():
        # Forward pass from latent space
        similarities = model.prototype_layer(z_tensor)
        logits = model.classifier(similarities)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)
        
    # Train a simple classifier on 2D embeddings to visualize boundary
    # KNN preserves local structure well, matching t-SNE's nature
    clf = KNeighborsClassifier(n_neighbors=50)
    clf.fit(data_embeddings, preds)
    
    # Create grid
    x_min, x_max = data_embeddings[:, 0].min() - 1, data_embeddings[:, 0].max() + 1
    y_min, y_max = data_embeddings[:, 1].min() - 1, data_embeddings[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict on grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create visualization
    print("  Creating plot...")
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot Decision Boundary (Background)
    # Red for Default (1), Green for Non-Default (0)
    # We use semi-transparent contourf
    custom_cmap = matplotlib.colors.ListedColormap(['#e8f5e9', '#ffebee']) # Light Green, Light Red
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=custom_cmap)
    
    # Add boundary line
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
        label='Prototypes',
        zorder=100
    )
    
    # Add prototype labels
    for i, (x, y) in enumerate(proto_embeddings):
        ax.annotate(
            f'P{i}',
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold',
            color='#8e44ad',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
        )
    
    ax.set_title('t-SNE Latent Space with Model Decision Boundary', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', label='Non-Default (Actual)', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='Default (Actual)', markersize=10),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#9b59b6', label='Prototypes', markersize=15),
        matplotlib.patches.Patch(facecolor='#e8f5e9', label='Predicted: Non-Default Region'),
        matplotlib.patches.Patch(facecolor='#ffebee', label='Predicted: Default Region'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=11, framealpha=0.9)
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
        # Create ids starting from training size
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


def main():
    print("=" * 60)
    print("GENERATING T-SNE VISUALIZATION AND SUBMISSION")
    print("=" * 60)
    
    # Load model and preprocessor
    print("\n[1/4] Loading model and preprocessor...")
    model, preprocessor, config = load_model_and_preprocessor()
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
    tsne_path = create_tsne_visualization(
        model, X_num_train, X_cat_train, train_labels,
        output_path='/workspace/underwriting-explainable-ai/tsne_visualization.png',
        n_samples=5000
    )
    
    # Generate submission
    print("\n[4/4] Generating submission...")
    test_path = '/workspace/data/test.csv'
    submission_path = generate_submission(
        model, preprocessor, test_path,
        output_path='/workspace/underwriting-explainable-ai/submission.csv'
    )
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - t-SNE: {tsne_path}")
    print(f"  - Submission: {submission_path}")


if __name__ == '__main__':
    main()
