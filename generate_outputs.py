#!/usr/bin/env python3
"""
Generate t-SNE visualization from trained model.

Supports both Loan and ICR datasets via --dataset flag.

Usage:
    python generate_outputs.py --dataset icr --checkpoint /workspace/checkpoints/icr/best_model_phase1.pt --phase 1
    python generate_outputs.py --dataset loan --checkpoint /workspace/checkpoints/loan/best_model_phase2.pt --phase 2
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

from src.models.model import create_model_from_config
from src.utils.visualization import create_tsne_visualization


def get_config_for_dataset(dataset: str):
    """Get the appropriate config for the dataset."""
    if dataset == 'loan':
        from src.config import get_default_config
        return get_default_config()
    elif dataset == 'icr':
        from src.icr_config import ICRConfig
        return ICRConfig()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_model_and_preprocessor(
    dataset: str,
    checkpoint_dir: str = None,
    model_path: str = None,
    force_phase: int = None
):
    """Load trained model and preprocessor."""
    config = get_config_for_dataset(dataset)
    
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
    if not os.path.exists(preprocessor_path):
        # Fallback: check subdirectory named after dataset
        preprocessor_path = os.path.join(checkpoint_dir, dataset, 'preprocessor.pkl')
        
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
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
    parser.add_argument('--dataset', type=str, choices=['loan', 'icr'], required=True,
                        help='Dataset to visualize: loan or icr')
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
    print(f"GENERATING T-SNE VISUALIZATION ({args.dataset.upper()})")
    print("=" * 60)
    
    # [1/3] Load model and preprocessor
    print("\n[1/3] Loading model and preprocessor...")
    model, preprocessor, config = load_model_and_preprocessor(
        dataset=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        model_path=args.checkpoint,
        force_phase=args.phase
    )
    phase = args.phase or 2
    print("  Model loaded successfully!")
    
    # [2/3] Load training data
    print("\n[2/3] Loading training data...")
    train_path = config.train_path
    print(f"  Training data path: {train_path}")
    train_df = pd.read_csv(train_path)
    
    # Find target column
    if args.dataset == 'icr':
        target_col = getattr(config, 'target_column', 'Class')
    else:
        target_col = 'loan_status'
    
    train_labels = train_df[target_col].values
    X_num_train, X_cat_train = preprocessor.transform(train_df)
    print(f"  Loaded {len(train_df)} training samples")
    
    # Prepare labels dictionary
    labels_dict = {'Class': train_labels}
    
    # ICR-specific: merge Alpha labels from greeks.csv
    if args.dataset == 'icr':
        if 'Alpha' not in train_df.columns:
            greeks_path = getattr(config, 'greeks_path', '/workspace/data/icr/greeks.csv')
            if os.path.exists(greeks_path):
                print(f"  Loading Greeks from {greeks_path} to get Alpha...")
                greeks_df = pd.read_csv(greeks_path)
                if 'Id' in train_df.columns and 'Id' in greeks_df.columns:
                    train_df = pd.merge(train_df, greeks_df[['Id', 'Alpha']], on='Id', how='left')
        
        if 'Alpha' in train_df.columns:
            print("  Found 'Alpha' column. Adding to visualization labels.")
            labels_dict['Alpha'] = train_df['Alpha'].fillna('N/A').values
        else:
            print("  'Alpha' column not found. Skipping Alpha visualization.")
        
    # [3/3] Create t-SNE visualization
    print(f"\n[3/3] Creating t-SNE visualizations (Phase {phase})...")
    create_tsne_visualization(
        model, 
        X_num_train, 
        X_cat_train, 
        labels_dict=labels_dict,
        output_dir=args.output_dir,
        file_prefix=f'tsne_visualization_phase{phase}',
        n_samples=5000
    )
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
