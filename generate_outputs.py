#!/usr/bin/env python3
"""
Generate t-SNE visualization from trained model (MNIST).

Usage:
    python generate_outputs.py --checkpoint /workspace/checkpoints/mnist/best_model_phase1.pt --phase 1
    python generate_outputs.py --checkpoint /workspace/checkpoints/mnist/best_model_phase2.pt --phase 2
"""
import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_default_config
from src.models.model import create_model_from_config
from src.utils.visualization import create_tsne_visualization
from src.data.preprocessor import load_mnist_data


def load_model_and_preprocessor(
    checkpoint_dir: str = None,
    model_path: str = None,
    force_phase: int = None
):
    """Load trained model and preprocessor."""
    config = get_default_config()
    
    # Resolve checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = config.model_save_path
    
    # Infer checkpoint_dir from model_path if preprocessor exists there
    if model_path:
        model_dir = os.path.dirname(model_path)
        if os.path.exists(os.path.join(model_dir, 'preprocessor.pkl')):
            checkpoint_dir = model_dir
            
    # Load preprocessor
    preprocessor_path = os.path.join(checkpoint_dir, 'preprocessor.pkl')
    # If not found, create new one (stateless for MNIST)
    
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
    else:
        from src.data.preprocessor import MNISTPreprocessor
        preprocessor = MNISTPreprocessor()
        preprocessor.fit()
    
    config.categorical_cardinalities = preprocessor.get_cardinalities()
    if hasattr(preprocessor, 'numerical_features'):
        config.numerical_features = preprocessor.numerical_features
    
    # Load checkpoint
    if model_path is None:
        for name in ['best_model_phase2.pt', 'best_model_phase1.pt', 'best_model.pt']:
            candidate = os.path.join(checkpoint_dir, name)
            if os.path.exists(candidate):
                model_path = candidate
                break
        if model_path is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    
    print(f"  Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Infer prototype counts from state_dict
    if 'prototype_layer.prototypes' in state_dict:
        config.n_prototypes = state_dict['prototype_layer.prototypes'].shape[0]
    if 'global_prototype_layer.prototypes' in state_dict:
        config.n_global_prototypes = state_dict['global_prototype_layer.prototypes'].shape[0]
        print(f"  Inferred n_global_prototypes: {config.n_global_prototypes}")
    
    # Create model
    model = create_model_from_config(config)
    
    # Set phase
    phase = force_phase if force_phase is not None else checkpoint.get('phase', 2)
    if phase == 2:
        model.set_second_phase()
    else:
        model.set_first_phase()
    print(f"  Model phase: {phase}")
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, preprocessor, config


def parse_args():
    parser = argparse.ArgumentParser(description='Generate t-SNE visualization')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific model checkpoint (optional)')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=1,
                        help='Force model phase (1 or 2)')
    parser.add_argument('--output-dir', type=str, default='/workspace/underwriting-explainable-ai',
                        help='Output directory for visualization files')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory containing preprocessor.pkl')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to visualize (1 to N)')
    return parser.parse_args()


def process_epoch(epoch, phase, config, args, preprocessor):
    """Process a single epoch checkpoint."""
    checkpoint_dir = args.checkpoint_dir or config.model_save_path
    model_path = os.path.join(checkpoint_dir, f'model_phase{phase}_epoch{epoch}.pt')
    
    if not os.path.exists(model_path):
        print(f"  Skipping Epoch {epoch}: Checkpoint not found at {model_path}")
        return

    print(f"\n>>> Processing Epoch {epoch} (Phase {phase})...")
    
    # Load model
    try:
        model, _, _ = load_model_and_preprocessor(
            checkpoint_dir=checkpoint_dir,
            model_path=model_path,
            force_phase=phase
        )
    except Exception as e:
        print(f"  Error loading model: {e}")
        return

    # Load Data (freshly stratified sample each time or consistent sample better?)
    # Users usually want consistent samples across epochs to see movement.
    # But for simplicity, we reload/resample. For rigorous analysis, we should fix the seed.
    # We fix the seed in main/global.
    
    # Load Data
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    root = config.train_path or './data'
    # Use TEST set for consistent evaluation/visualization, or TRAIN?
    # User likely wants TRAIN to see how it learned.
    dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, targets = next(iter(loader))
    
    X_num = data.view(data.size(0), -1).numpy()
    X_cat = np.zeros((len(dataset), 0), dtype=np.int64)
    y_raw = targets.numpy()
    
    # Stratified Sample
    limit = 2000 # Enough points
    indices = np.random.choice(len(X_num), size=min(len(X_num), limit), replace=False)
    
    X_num_sample = X_num[indices]
    X_cat_sample = X_cat[indices]
    y_raw_sample = y_raw[indices]
    
    # Define Labels
    labels_10 = y_raw_sample
    labels_2 = (y_raw_sample % 4 == 0).astype(int)
    
    # Extract Embeddings
    device = next(model.parameters()).device
    model.eval()
    
    z_list = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_num_sample), batch_size):
            b_num = torch.tensor(X_num_sample[i:i+batch_size], dtype=torch.float32).to(device)
            b_cat = torch.tensor(X_cat_sample[i:i+batch_size], dtype=torch.long).to(device)
            outputs = model(b_num, b_cat, return_all=True)
            z_list.append(outputs['z'].cpu().numpy())
            
    z_all = np.concatenate(z_list, axis=0)
    
    # Helper
    def plot_simple_tsne(data, labels, path, title, num_classes):
        print(f"  Plotting {title}...")
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        emb = tsne.fit_transform(data)
        
        plt.figure(figsize=(10, 8))
        if num_classes <= 10:
            cmap = plt.get_cmap('tab10')
        else:
            cmap = plt.get_cmap('tab20')
            
        unique_labels = sorted(np.unique(labels))
        for lbl in unique_labels:
            mask = labels == lbl
            label_text = f"Digit {lbl}" if num_classes == 10 else f"Class {lbl}"
            if num_classes == 2:
                label_text = "Multiple of 4" if lbl == 1 else "Other"
                
            plt.scatter(
                emb[mask, 0], emb[mask, 1],
                color=cmap(lbl) if num_classes == 10 else cmap(lbl),
                label=label_text,
                alpha=0.7, 
                s=20, 
                edgecolors='none'
            )
            
        plt.title(title, fontsize=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    # 1. 10-Class Plot
    plot_simple_tsne(
        z_all, labels_10,
        os.path.join(args.output_dir, f'tsne_phase{phase}_epoch{epoch}_10class.png'),
        f'Phase {phase} Epoch {epoch} (10 Classes)',
        10
    )
    
    # 2. 2-Class Plot
    plot_simple_tsne(
        z_all, labels_2,
        os.path.join(args.output_dir, f'tsne_phase{phase}_epoch{epoch}_2class.png'),
        f'Phase {phase} Epoch {epoch} (Binary)',
        2
    )


def main():
    args = parse_args()
    
    print("=" * 60)
    print("GENERATING PER-EPOCH VISUALIZATION (MNIST)")
    print("=" * 60)
    
    # Basic setup to get config
    config = get_default_config()
    
    # We need a preprocessor for consistency, though we load data manually
    # Just to satisfy load_model_and_preprocessor signature
    from src.data.preprocessor import MNISTPreprocessor
    preprocessor = MNISTPreprocessor() 
    
    # Loop 1 to 5 (or args.epochs)
    for epoch in range(1, args.epochs + 1):
        process_epoch(epoch, args.phase, config, args, preprocessor)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
