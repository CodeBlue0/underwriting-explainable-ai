#!/usr/bin/env python3
"""
Evaluate decoder reconstruction quality from a trained model checkpoint.

Supports both Loan and ICR datasets via --dataset flag.

Usage:
    python evaluate_decoder.py --dataset loan
    python evaluate_decoder.py --dataset icr --checkpoint /workspace/checkpoints/icr/best_model_phase1.pt
"""
import argparse
import os
import sys
import pickle
import torch
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.model import create_model_from_config
from src.models.decoder import DecoderEvaluator


def get_config_for_dataset(dataset: str):
    """Get the appropriate config for the dataset."""
    if dataset == 'loan':
        from src.config import get_default_config
        return get_default_config()
    elif dataset == 'icr':
        from src.icr_config import get_icr_config
        return get_icr_config()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_data_for_dataset(dataset: str, config):
    """Load and preprocess data for the given dataset."""
    if dataset == 'loan':
        from src.data.preprocessor import load_and_preprocess_data
        (train_num, train_cat, train_target), (test_num, test_cat), preprocessor = \
            load_and_preprocess_data(
                config.train_path,
                config.test_path,
                config.numerical_features,
                config.categorical_features
            )
        return train_num, train_cat, train_target, preprocessor
    elif dataset == 'icr':
        from src.data.icr_preprocessor import ICRPreprocessor
        import numpy as np
        train_df = pd.read_csv(config.train_path)
        preprocessor = ICRPreprocessor(
            numerical_features=config.numerical_features,
            categorical_features=config.categorical_features,
            exclude_columns=config.exclude_columns
        )
        train_num, train_cat = preprocessor.fit_transform(train_df)
        train_target = train_df[config.target_column].values.astype(np.int64)
        return train_num, train_cat, train_target, preprocessor


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate decoder reconstruction quality')
    parser.add_argument('--dataset', type=str, choices=['loan', 'icr'], required=True,
                        help='Dataset to evaluate: loan or icr')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (auto-detected if not provided)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory containing preprocessor.pkl (auto-detected if not provided)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=None,
                        help='Force model phase (1 or 2)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n" + "=" * 60)
    print(f"DECODER EVALUATION ({args.dataset.upper()})")
    print("=" * 60)
    
    # Get config
    config = get_config_for_dataset(args.dataset)
    config.batch_size = args.batch_size
    
    # Resolve checkpoint and checkpoint-dir
    checkpoint_dir = args.checkpoint_dir or config.model_save_path
    checkpoint = args.checkpoint
    if checkpoint is None:
        # Auto-detect: prefer phase2, fallback to phase1, fallback to best_model.pt
        for name in ['best_model_phase2.pt', 'best_model_phase1.pt', 'best_model.pt']:
            candidate = os.path.join(checkpoint_dir, name)
            if os.path.exists(candidate):
                checkpoint = candidate
                break
        if checkpoint is None:
            print(f"Error: No checkpoint found in {checkpoint_dir}")
            return
    
    # Load preprocessor
    print("\n[1/4] Loading preprocessor...")
    preprocessor_path = os.path.join(checkpoint_dir, 'preprocessor.pkl')
    if not os.path.exists(preprocessor_path):
        preprocessor_path = os.path.join(os.path.dirname(checkpoint), 'preprocessor.pkl')
    
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    config.categorical_cardinalities = preprocessor.get_cardinalities()
    
    # Load checkpoint
    print(f"\n[2/4] Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    
    # Infer model dimensions
    if 'prototype_layer.prototypes' in state_dict:
        config.n_prototypes = state_dict['prototype_layer.prototypes'].shape[0]
    if 'global_prototype_layer.prototypes' in state_dict:
        config.n_global_prototypes = state_dict['global_prototype_layer.prototypes'].shape[0]
    
    # Create and load model
    model = create_model_from_config(config)
    
    phase = args.phase or ckpt.get('phase', 2)
    if phase == 2:
        model.set_second_phase()
    else:
        model.set_first_phase()
    print(f"  Model phase: {phase}")
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load and preprocess data
    print("\n[3/4] Loading validation data...")
    if args.dataset == 'loan':
        from src.data.preprocessor import load_and_preprocess_data
        from src.data.dataset import create_data_loaders
        (train_num, train_cat, train_target), (test_num, test_cat), _ = \
            load_and_preprocess_data(
                config.train_path,
                config.test_path,
                config.numerical_features,
                config.categorical_features
            )
        train_loader, val_loader = create_data_loaders(
            train_num, train_cat, train_target,
            batch_size=config.batch_size,
            val_split=0.1,
            seed=config.seed
        )
    elif args.dataset == 'icr':
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset
        train_df = pd.read_csv(config.train_path)
        train_num, train_cat = preprocessor.transform(train_df)
        train_target = train_df[config.target_column].values.astype(np.int64)
        dataset = TensorDataset(
            torch.FloatTensor(train_num),
            torch.LongTensor(train_cat),
            torch.LongTensor(train_target)
        )
        val_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"  Validation batches: {len(val_loader)}")
    
    # Evaluate decoder
    print("\n[4/4] Evaluating decoder...")
    feature_names = {
        'numerical': config.numerical_features,
        'categorical': config.categorical_features
    }
    evaluator = DecoderEvaluator(feature_names=feature_names)
    metrics = evaluator.evaluate(model, val_loader, device=device)
    evaluator.print_report(metrics)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return metrics


if __name__ == '__main__':
    main()
