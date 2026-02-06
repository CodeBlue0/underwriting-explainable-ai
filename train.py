#!/usr/bin/env python3
"""
Main training script for Prototype-based Neuro-Symbolic Model.

Usage:
    python train.py [--epochs 50] [--batch_size 256] [--lr 1e-3]
"""
import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ModelConfig, PROTOTYPE_DESCRIPTIONS, get_default_config
from src.data.preprocessor import LoanDataPreprocessor, load_and_preprocess_data
from src.data.dataset import create_data_loaders, create_test_loader
from src.models.model import PrototypeNetwork, create_model_from_config
from src.training.trainer import train_model, Trainer
from src.explainability.prototype_explainer import PrototypeExplainer
from src.explainability.report_generator import LoanDecisionReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Train Prototype Network for Loan Default Prediction')
    
    # Data paths
    parser.add_argument('--train_path', type=str, default='/workspace/data/train.csv',
                        help='Path to training data CSV')
    parser.add_argument('--test_path', type=str, default='/workspace/data/test.csv',
                        help='Path to test data CSV')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=64,
                        help='Hidden dimension for transformer')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--n_prototypes', type=int, default=10,
                        help='Number of prototype vectors')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Loss weights
    parser.add_argument('--lambda_recon', type=float, default=0.1,
                        help='Weight for reconstruction loss')
    parser.add_argument('--lambda_diversity', type=float, default=0.01,
                        help='Weight for diversity loss')
    parser.add_argument('--lambda_clustering', type=float, default=0.05,
                        help='Weight for clustering loss')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='/workspace/checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick test with 2 epochs on small subset')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create config
    config = get_default_config()
    config.train_path = args.train_path
    config.test_path = args.test_path
    config.d_model = args.d_model
    config.n_heads = args.n_heads
    config.n_layers = args.n_layers
    config.n_prototypes = args.n_prototypes
    config.epochs = args.epochs if not args.quick_test else 2
    config.batch_size = args.batch_size if not args.quick_test else 32
    config.learning_rate = args.lr
    config.early_stopping_patience = args.patience
    config.lambda_reconstruction = args.lambda_recon
    config.lambda_diversity = args.lambda_diversity
    config.lambda_clustering = args.lambda_clustering
    config.device = device
    config.seed = args.seed
    
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("PROTOTYPE-BASED NEURO-SYMBOLIC MODEL TRAINING")
    print("="*60)
    
    # Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    (train_num, train_cat, train_target), (test_num, test_cat), preprocessor = \
        load_and_preprocess_data(
            config.train_path,
            config.test_path,
            config.numerical_features,
            config.categorical_features
        )
    
    # Update cardinalities from data
    config.categorical_cardinalities = preprocessor.get_cardinalities()
    
    print(f"  Training samples: {len(train_num)}")
    print(f"  Test samples: {len(test_num)}")
    print(f"  Numerical features: {config.n_numerical}")
    print(f"  Categorical features: {config.n_categorical}")
    print(f"  Cardinalities: {config.categorical_cardinalities}")
    
    # Quick test: use subset
    if args.quick_test:
        print("\n  [Quick test mode - using 1000 samples]")
        train_num = train_num[:1000]
        train_cat = train_cat[:1000]
        train_target = train_target[:1000]
    
    # Create data loaders
    print("\n[2/5] Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_num, train_cat, train_target,
        batch_size=config.batch_size,
        val_split=0.1,
        seed=config.seed
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n[3/5] Creating model...")
    model = create_model_from_config(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Model summary
    print(f"\n  FT-Transformer Encoder:")
    print(f"    - Hidden dim: {config.d_model}")
    print(f"    - Attention heads: {config.n_heads}")
    print(f"    - Transformer layers: {config.n_layers}")
    print(f"\n  Prototype Layer:")
    print(f"    - Number of prototypes: {config.n_prototypes}")
    print(f"    - Similarity type: {config.similarity_type}")
    
    # Train
    print("\n[4/5] Training model...")
    print("-"*60)
    
    save_path = os.path.join(args.save_dir, 'best_model.pt')
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_path=save_path,
        verbose=True
    )
    
    print("-"*60)
    print(f"\nBest Validation AUC: {max(history['val_auc']):.4f}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(args.save_dir, 'preprocessor.pkl')
    preprocessor.save(preprocessor_path)
    print(f"\nPreprocessor saved to: {preprocessor_path}")
    print(f"Model saved to: {save_path}")
    
    # Generate sample explanation
    print("\n[5/5] Generating sample explanation...")
    print("-"*60)
    
    explainer = PrototypeExplainer(
        model=model,
        preprocessor=preprocessor,
        prototype_descriptions=PROTOTYPE_DESCRIPTIONS,
        device=device
    )
    
    report_generator = LoanDecisionReportGenerator(explainer)
    
    # Explain first validation sample
    sample_idx = 0
    sample_num = train_num[sample_idx]
    sample_cat = train_cat[sample_idx]
    
    explanation = explainer.explain_single(sample_num, sample_cat, top_k=3)
    report = report_generator.generate_report(explanation, applicant_id=f"SAMPLE-{sample_idx}")
    
    print(report)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    return model, history, preprocessor, explainer


if __name__ == '__main__':
    main()
