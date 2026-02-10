#!/usr/bin/env python3
"""
Training script for ICR (Identify Age-Related Conditions) dataset with PTaRL.

Usage:
    Phase 1 (Representation Learning):
        python train_icr.py --phase 1
    
    Phase 2 (Prototype Learning):
        python train_icr.py --phase 2 --checkpoint /workspace/checkpoints/icr/best_model_phase1.pt
"""
import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.icr_config import get_icr_config, ICRConfig
from src.data.icr_preprocessor import ICRPreprocessor
from src.models.model import PrototypeNetwork
from src.training.trainer import Trainer

# Visualization imports
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str,
    label_type: str = 'alpha'
):
    """
    Plot t-SNE visualization (High Quality).
    
    Args:
        embeddings_2d: (N, 2) t-SNE coordinates
        labels: (N,) Labels for coloring
        save_path: Path to save the plot
        title: Plot title
        label_type: 'alpha' or 'class'
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if label_type == 'alpha':
        # Color scheme for Alpha categories
        colors = {
            'A': '#2ecc71',  # Green - Class 0 (Normal)
            'B': '#e74c3c',  # Red - Class 1 subtype B
            'D': '#9b59b6',  # Purple - Class 1 subtype D
            'G': '#f39c12',  # Orange - Class 1 subtype G
        }
        labels_display = {
            'A': 'A (Class 0 - Normal)',
            'B': 'B (Class 1 - Subtype B)',
            'D': 'D (Class 1 - Subtype D)',
            'G': 'G (Class 1 - Subtype G)',
        }
        unique_labels = ['A', 'B', 'D', 'G']
        
    else:  # 'class'
        # Color scheme for Class
        colors = {
            0: '#2ecc71',  # Green - Class 0 (Normal)
            1: '#e74c3c',  # Red - Class 1 (Age-related condition)
        }
        labels_display = {
            0: 'Class 0 (Normal)',
            1: 'Class 1 (Age-related condition)',
        }
        unique_labels = [0, 1]
    
    # Plot each category
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 0:
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=colors[label],
                label=f"{labels_display[label]} (n={mask.sum()})",
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved high-quality plot to: {save_path}")



def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(
    train_num: np.ndarray,
    train_cat: np.ndarray,
    train_target: np.ndarray,
    batch_size: int,
    val_split: float = 0.2,
    seed: int = 42
):
    """Create train and validation DataLoaders."""
    # Split data
    indices = np.arange(len(train_target))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=val_split, 
        random_state=seed,
        stratify=train_target
    )
    
    # Create tensors
    train_num_t = torch.FloatTensor(train_num[train_idx])
    train_cat_t = torch.LongTensor(train_cat[train_idx])
    train_target_t = torch.FloatTensor(train_target[train_idx])  # Float for BCE loss
    
    val_num_t = torch.FloatTensor(train_num[val_idx])
    val_cat_t = torch.LongTensor(train_cat[val_idx])
    val_target_t = torch.FloatTensor(train_target[val_idx])  # Float for BCE loss
    
    # Create datasets
    train_dataset = TensorDataset(train_num_t, train_cat_t, train_target_t)
    val_dataset = TensorDataset(val_num_t, val_cat_t, val_target_t)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def create_model(config: ICRConfig, device: str):
    """Create PrototypeNetwork model for ICR dataset."""
    model = PrototypeNetwork(
        n_numerical=config.n_numerical,
        categorical_cardinalities=config.get_cardinality_list(),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ffn=config.d_ffn,
        dropout=config.dropout,
        decoder_hidden_dim=config.decoder_hidden_dim,
        n_global_prototypes=config.n_global_prototypes
    ).to(device)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train PTaRL model on ICR dataset')
    parser.add_argument('--phase', type=int, choices=[1, 2], required=True,
                        help='Training phase: 1 (representation) or 2 (prototype)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (required for phase 2)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    args = parser.parse_args()
    
    # Validate arguments
    if args.phase == 2 and args.checkpoint is None:
        print("Warning: Phase 2 training without checkpoint. Starting from scratch.")
    
    # Load config
    config = get_icr_config()
    
    # Override config values if provided
    if args.epochs:
        if args.phase == 1:
            config.phase1_epochs = args.epochs
        else:
            config.phase2_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    
    device = config.device
    
    print("=" * 60)
    print("ICR Dataset - PTaRL Training")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Device: {device}")
    
    # Set seed
    set_seed(config.seed)
    
    # Load and preprocess data
    print("\n[1/5] Loading data...")
    train_df = pd.read_csv(config.train_path)
    
    # Load Greeks for Alpha labels
    if os.path.exists(config.greeks_path):
        greeks_df = pd.read_csv(config.greeks_path)
        # Map Alpha based on Id
        train_df['Alpha'] = train_df['Id'].map(greeks_df.set_index('Id')['Alpha'])
        print(f"  Loaded Alpha labels from {config.greeks_path}")
    else:
        print(f"  Warning: Greeks file not found at {config.greeks_path}. Visualization will fail.")
        # Create dummy for testing if needed or let it fail
        train_df['Alpha'] = 'A'

    print(f"  Samples: {len(train_df)}")
    print(f"  Class distribution: {dict(train_df['Class'].value_counts())}")
    
    # Preprocess
    print("\n[2/5] Preprocessing...")
    preprocessor = ICRPreprocessor(
        numerical_features=config.numerical_features,
        categorical_features=config.categorical_features,
        exclude_columns=config.exclude_columns
    )
    
    train_num, train_cat = preprocessor.fit_transform(train_df)
    train_target = train_df[config.target_column].values.astype(np.int64)
    
    print(f"  Numerical features: {train_num.shape[1]}")
    print(f"  Categorical features: {train_cat.shape[1]}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(config.model_save_path, 'preprocessor.pkl')
    os.makedirs(config.model_save_path, exist_ok=True)
    preprocessor.save(preprocessor_path)
    print(f"  Saved preprocessor to: {preprocessor_path}")
    
    # Create dataloaders
    print("\n[3/5] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_num, train_cat, train_target,
        batch_size=config.batch_size,
        val_split=0.2,
        seed=config.seed
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n[4/5] Creating model...")
    model = create_model(config, device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if unexpected:
            print(f"  Ignored deprecated keys: {len(unexpected)}")
        if args.phase == 2:
            print("  Continuing to Phase 2...")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer (using Trainer's expected signature)
    print("\n[5/5] Training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    epochs = config.phase1_epochs if args.phase == 1 else config.phase2_epochs
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print()
    
    # Train the appropriate phase
    if args.phase == 1:
        trainer.train_phase(phase=1, epochs=epochs, early_stopping_patience=config.early_stopping_patience)
        # Save Phase 1 checkpoint
        checkpoint_path = os.path.join(config.model_save_path, 'best_model_phase1.pt')
        trainer.save_checkpoint(checkpoint_path)
        print(f"\n  Phase 1 checkpoint saved to: {checkpoint_path}")
    else:
        trainer.train_phase(phase=2, epochs=epochs, early_stopping_patience=config.early_stopping_patience)
        # Save Phase 2 checkpoint
        checkpoint_path = os.path.join(config.model_save_path, 'best_model_phase2.pt')
        trainer.save_checkpoint(checkpoint_path)
        print(f"\n  Phase 2 checkpoint saved to: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Checkpoints saved to: {config.model_save_path}")
    
    # Phase 2 specific: save prototype info
    if args.phase == 2:
        print("\nPrototype Information:")
        print(f"  Number of prototypes: {config.n_global_prototypes}")
        prototypes = model.global_prototype_layer.prototypes.detach().cpu().numpy()
        print(f"  Prototype shape: {prototypes.shape}")

    # ---------------------------------------------------------
    # High-Quality Visualization (Post-Training)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating High-Quality t-SNE Visualization")
    print("=" * 60)
    
    # Reload best model for visualization
    final_checkpoint_path = os.path.join(config.model_save_path, f'best_model_phase{args.phase}.pt')
    if os.path.exists(final_checkpoint_path):
        print(f"Loading best model: {final_checkpoint_path}")
        checkpoint = torch.load(final_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.phase == 2:
            model.set_second_phase()
        else:
            model.set_first_phase()
    
    model.eval()
    
    print("Extracting embeddings for visualization...")
    
    # Create sequential loader for full dataset (ordered)
    viz_dataset = TensorDataset(torch.FloatTensor(train_num), torch.LongTensor(train_cat))
    viz_loader = DataLoader(viz_dataset, batch_size=config.batch_size, shuffle=False)
    
    embeddings_list = []
    pspace_list = []
    
    with torch.no_grad():
        for x_n, x_c in viz_loader:
            x_n = x_n.to(device)
            x_c = x_c.to(device)
            outputs = model(x_n, x_c, return_all=True)
            embeddings_list.append(outputs['z'].cpu().numpy())
            if args.phase == 2:
                pspace_list.append(outputs['p_space'].cpu().numpy())
    
    z = np.concatenate(embeddings_list, axis=0)
    alpha_labels = train_df['Alpha'].fillna('A').values
    class_labels = train_df['Class'].values
    
    # Visualizing Z-Space
    print(f"Calculating t-SNE for Z-Space ({z.shape})...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z)
    
    # plot_tsne(z_2d, alpha_labels, os.path.join(config.model_save_path, f'icr_tsne_phase{args.phase}_z_alpha.png'), f"Phase {args.phase} Z-Space (Alpha)", 'alpha')
    # plot_tsne(z_2d, class_labels, os.path.join(config.model_save_path, f'icr_tsne_phase{args.phase}_z_class.png'), f"Phase {args.phase} Z-Space (Class)", 'class')
    
    # Visualizing P-Space (Phase 2 only)
    if args.phase == 2:
        p_space = np.concatenate(pspace_list, axis=0)
        # print(f"Calculating t-SNE for P-Space ({p_space.shape})...")
        # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        # p_2d = tsne.fit_transform(p_space)
        
        # plot_tsne(p_2d, alpha_labels, os.path.join(config.model_save_path, f'icr_tsne_phase{args.phase}_pspace_alpha.png'), f"Phase {args.phase} P-Space (Alpha)", 'alpha')
        # plot_tsne(p_2d, class_labels, os.path.join(config.model_save_path, f'icr_tsne_phase{args.phase}_pspace_class.png'), f"Phase {args.phase} P-Space (Class)", 'class')
        
    print(f"Visualization saved to: {config.model_save_path}")


if __name__ == '__main__':
    main()
