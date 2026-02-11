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
                        help='Path to model checkpoint')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=None,
                        help='Force model phase (1 or 2)')
    parser.add_argument('--output-dir', type=str, default='/workspace/underwriting-explainable-ai',
                        help='Output directory for visualization files')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory containing preprocessor.pkl (auto-detected if not provided)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("GENERATING T-SNE VISUALIZATION (MNIST)")
    print("=" * 60)
    
    # [1/3] Load model and preprocessor
    print("\n[1/3] Loading model and preprocessor...")
    model, preprocessor, config = load_model_and_preprocessor(
        checkpoint_dir=args.checkpoint_dir,
        model_path=args.checkpoint,
        force_phase=args.phase
    )
    phase = args.phase or 2
    print("  Model loaded successfully!")
    
    # [2/3] Load training data
    print("\n[2/3] Loading training data...")
    from torchvision import datasets, transforms
    
    # Use standard transform matching preprocessor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    root = config.train_path or './data'
    train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    
    # Use preprocessor only for X transformation logic (if needed) or just manual flatten
    # The preprocessor.transform method does this:
    # loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    # data, targets = next(iter(loader))
    # X_num = data.view(data.size(0), -1).numpy()
    
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    data, targets = next(iter(loader))
    
    train_num = data.view(data.size(0), -1).numpy()
    train_cat = np.zeros((len(train_dataset), 0), dtype=np.int64)
    train_target = targets.numpy() # Original 0-9 labels!
    
    print(f"  Loaded {len(train_num)} training samples")
    print(f"  Target classes found: {np.unique(train_target)}")
    
    # Limit for visualization
    limit = 5000
    indices = np.random.choice(len(train_num), size=min(len(train_num), limit), replace=False)
    
    X_num_train = train_num[indices]
    X_cat_train = train_cat[indices]
    train_labels = train_target[indices]
    
    # [3/3] Create t-SNE visualization
    print(f"\n[3/3] Creating t-SNE visualizations (Phase {phase})...")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Extract embeddings
    z_list = []
    p_list = []
    
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_num_train), batch_size):
            b_num = torch.tensor(X_num_train[i:i+batch_size], dtype=torch.float32).to(device)
            b_cat = torch.tensor(X_cat_train[i:i+batch_size], dtype=torch.long).to(device)
            
            outputs = model(b_num, b_cat, return_all=True)
            z_list.append(outputs['z'].cpu().numpy())
            
            if 'p_space' in outputs:
                p_list.append(outputs['p_space'].cpu().numpy())
            elif phase == 2:
                # If model is in phase 2 but p_space not returned (should be there if return_all=True)
                pass

    z_all = np.concatenate(z_list, axis=0)
    
    # Helper for plotting
    def plot_simple_tsne(data, labels, path, title):
        print(f"  Computing t-SNE for {title}...")
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        emb = tsne.fit_transform(data)
        
        print(f"  Plotting to {path}...")
        plt.figure(figsize=(10, 8))
        
        # Use tab10 colormap which has 10 distinct colors
        cmap = plt.get_cmap('tab10')
        
        # Plot each class separately to create a proper legend
        unique_labels = sorted(np.unique(labels))
        for lbl in unique_labels:
            mask = labels == lbl
            plt.scatter(
                emb[mask, 0], emb[mask, 1],
                color=cmap(lbl),  # discrete access for tab10 (0-9)
                label=f"Digit {lbl}",
                alpha=0.7, 
                s=20, 
                edgecolors='none'
            )
            
        plt.title(title, fontsize=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Digit Class")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    # Plot Z-Space
    plot_simple_tsne(
        z_all, 
        train_labels, 
        os.path.join(args.output_dir, f'mnist_tsne_phase{phase}_z_class.png'), 
        f'Phase {phase} Z-Space (Digits)'
    )
    
    # Plot P-Space if available
    if p_list:
        p_all = np.concatenate(p_list, axis=0)
        plot_simple_tsne(
            p_all,
            train_labels,
            os.path.join(args.output_dir, f'mnist_tsne_phase{phase}_pspace_class.png'),
            f'Phase {phase} P-Space (Digits)'
        )
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
