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
    # This automatically downloads and preprocesses (flatten + scale)
    (train_num, train_cat, train_target), _, _ = load_mnist_data(config)
    print(f"  Loaded {len(train_num)} training samples")
    
    # Limit for visualization
    limit = 1000
    indices = np.random.choice(len(train_num), size=min(len(train_num), limit), replace=False)
    
    X_num_train = train_num[indices]
    X_cat_train = train_cat[indices]
    train_labels = train_target[indices]
    
    # Prepare labels dictionary
    labels_dict = {'IsMultipleOf4': train_labels}
        
    # [3/3] Create t-SNE visualization
    print(f"\n[3/3] Creating t-SNE visualizations (Phase {phase})...")
    create_tsne_visualization(
        model, 
        X_num_train, 
        X_cat_train, 
        labels_dict=labels_dict,
        output_dir=args.output_dir,
        file_prefix=f'mnist_tsne_phase{phase}',
        n_samples=limit
    )
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
