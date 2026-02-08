#!/usr/bin/env python3
"""
Evaluate decoder reconstruction quality from a trained model checkpoint.

Usage:
    python evaluate_decoder.py
    python evaluate_decoder.py --checkpoint /workspace/checkpoints/best_model_phase1.pt
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

from src.config import get_default_config
from src.data.preprocessor import load_and_preprocess_data
from src.data.dataset import create_data_loaders
from src.models.model import create_model_from_config
from src.models.decoder import DecoderEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate decoder reconstruction quality')
    parser.add_argument('--checkpoint', type=str, default='/workspace/checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='/workspace/checkpoints',
                        help='Directory containing preprocessor.pkl')
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
    print("DECODER EVALUATION")
    print("=" * 60)
    
    # Load preprocessor
    print("\n[1/4] Loading preprocessor...")
    preprocessor_path = os.path.join(args.checkpoint_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Get config
    config = get_default_config()
    config.categorical_cardinalities = preprocessor.get_cardinalities()
    config.batch_size = args.batch_size
    
    # Load checkpoint
    print(f"\n[2/4] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Infer model dimensions
    if 'prototype_layer.prototypes' in state_dict:
        config.n_prototypes = state_dict['prototype_layer.prototypes'].shape[0]
    if 'global_prototype_layer.prototypes' in state_dict:
        config.n_global_prototypes = state_dict['global_prototype_layer.prototypes'].shape[0]
    
    # Create and load model
    model = create_model_from_config(config)
    
    phase = args.phase or checkpoint.get('phase', 2)
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
