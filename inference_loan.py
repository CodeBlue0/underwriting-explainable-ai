#!/usr/bin/env python3
"""
Generate submission file for Loan Default Prediction.

Usage:
    python inference_loan.py --checkpoint /workspace/checkpoints/loan/best_model_phase2.pt --data /workspace/data/loan/test.csv --output /workspace/data/loan/submission.csv
"""
import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_default_config, ModelConfig
from src.data.preprocessor import LoanDataPreprocessor
from src.data.dataset import create_test_loader
from src.models.model import create_model_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='Generate submission for Loan Default Prediction')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='/workspace/data/loan/test.csv',
                        help='Path to test data CSV')
    parser.add_argument('--output', type=str, default='/workspace/data/loan/submission.csv',
                        help='Path to save submission CSV')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    return parser.parse_args()


def load_model_and_preprocessor(checkpoint_path: str, device: str):
    """Load model and preprocessor from checkpoint directory."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # 1. Load Preprocessor
    preprocessor_path = os.path.join(checkpoint_dir, 'preprocessor.pkl')
    if not os.path.exists(preprocessor_path):
        # Try checking if checkpoint is inside a subdirectory
        parent_dir = os.path.dirname(checkpoint_dir)
        preprocessor_path_parent = os.path.join(parent_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path_parent):
            preprocessor_path = preprocessor_path_parent
        else:
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
            
    print(f"Loading preprocessor from {preprocessor_path}...")
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
        
    # 2. Load Config & Model
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    config = get_default_config()
    
    # Update config from preprocessor
    config.categorical_cardinalities = preprocessor.get_cardinalities()
    if hasattr(preprocessor, 'numerical_features'):
        config.numerical_features = preprocessor.numerical_features
        
    # Infer prototype counts from state_dict
    if 'prototype_layer.prototypes' in state_dict:
        config.n_prototypes = state_dict['prototype_layer.prototypes'].shape[0]
    if 'global_prototype_layer.prototypes' in state_dict:
        config.n_global_prototypes = state_dict['global_prototype_layer.prototypes'].shape[0]
    
    model = create_model_from_config(config)
    
    # Set phase
    phase = checkpoint.get('phase', 2)
    if phase == 2:
        model.set_second_phase()
    else:
        model.set_first_phase()
    print(f"Model loaded (Phase {phase})")
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, preprocessor, config


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model
    model, preprocessor, config = load_model_and_preprocessor(args.checkpoint, device)
    
    # 2. Load Data
    print(f"Loading test data from {args.data}...")
    test_df = pd.read_csv(args.data)
    
    # Check ID column
    id_col = 'id'
    if id_col not in test_df.columns:
        # Check if 'Id' (capital I)
        if 'Id' in test_df.columns:
            id_col = 'Id'
        else:
            print("Warning: 'id' column not found. Creating sequential IDs.")
            test_df['id'] = range(len(test_df))
    
    ids = test_df[id_col].values
    
    # 3. Transform Data
    print("Transforming data...")
    try:
        X_num, X_cat = preprocessor.transform(test_df)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        # Check if feature columns match
        expected_cols = preprocessor.numerical_features + preprocessor.categorical_features
        missing = [c for c in expected_cols if c not in test_df.columns]
        if missing:
            print(f"Missing columns: {missing}")
        raise e
        
    # 4. Create DataLoader
    test_loader = create_test_loader(
        X_num, X_cat,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # 5. Inference
    print(f"Running inference on {len(test_df)} samples...")
    all_probs = []
    
    with torch.no_grad():
        for x_num, x_cat in tqdm(test_loader, desc="Inference"):
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            
            outputs = model(x_num, x_cat, return_all=False)
            probs = outputs['probabilities'].cpu().numpy()
            all_probs.extend(probs)
            
    # 6. Save Submission
    submission = pd.DataFrame({
        'id': ids,
        'loan_status': all_probs
    })
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Submission saved to: {args.output}")
    print(submission.head())


if __name__ == '__main__':
    main()
