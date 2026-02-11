#!/usr/bin/env python3
"""
Evaluate decoder reconstruction quality from a trained model checkpoint.

Usage:
    python evaluate_decoder.py
    python evaluate_decoder.py --checkpoint /workspace/checkpoints/icr/best_model_phase1.pt
"""
import argparse
import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ICRConfig, get_default_config
from src.data.preprocessor import ICRPreprocessor
from src.models.model import create_model_from_config
from src.models.decoder import DecoderEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate decoder reconstruction quality')
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
    print("DECODER EVALUATION (ICR)")
    print("=" * 60)
    
    # Get config
    config = get_default_config()
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
