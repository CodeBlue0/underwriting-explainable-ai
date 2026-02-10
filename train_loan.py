#!/usr/bin/env python3
"""
Training script for Loan Default Prediction dataset with PTaRL.

Usage:
    Phase 1 (Representation Learning):
        python train_loan.py --phase 1
    
    Phase 2 (Prototype Learning):
        python train_loan.py --phase 2 --checkpoint /workspace/checkpoints/loan/best_model_phase1.pt
"""
import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_default_config, ModelConfig, PROTOTYPE_DESCRIPTIONS
from src.data.preprocessor import LoanDataPreprocessor, load_and_preprocess_data
from src.data.dataset import create_data_loaders
from src.models.model import PrototypeNetwork, create_model_from_config
from src.models.decoder import DecoderEvaluator
from src.training.trainer import Trainer
from src.explainability.prototype_explainer import PrototypeExplainer
from src.explainability.report_generator import LoanDecisionReportGenerator


def plot_tsne(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str,
):
    """
    Plot t-SNE visualization for loan data (binary: loan_status 0/1).
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {
        0: '#2ecc71',  # Green - Non-Default
        1: '#e74c3c',  # Red - Default
    }
    labels_display = {
        0: 'Non-Default (loan_status=0)',
        1: 'Default (loan_status=1)',
    }
    
    for label in [0, 1]:
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
    print(f"  Saved plot to: {save_path}")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Train PTaRL model on Loan Default dataset')
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
    config = get_default_config()
    
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
    print("Loan Default Dataset - PTaRL Training")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Device: {device}")
    
    # Set seed
    set_seed(config.seed)
    
    # Load and preprocess data
    print("\n[1/5] Loading data...")
    (train_num, train_cat, train_target), (test_num, test_cat), preprocessor = \
        load_and_preprocess_data(
            config.train_path,
            config.test_path,
            config.numerical_features,
            config.categorical_features
        )
    
    # Update cardinalities from data
    config.categorical_cardinalities = preprocessor.get_cardinalities()
    
    print(f"  Samples: {len(train_num)}")
    print(f"  Class distribution: {dict(pd.Series(train_target).value_counts())}")
    print(f"  Numerical features: {train_num.shape[1]}")
    print(f"  Categorical features: {train_cat.shape[1]}")
    
    # Save preprocessor
    os.makedirs(config.model_save_path, exist_ok=True)
    preprocessor_path = os.path.join(config.model_save_path, 'preprocessor.pkl')
    preprocessor.save(preprocessor_path)
    print(f"  Saved preprocessor to: {preprocessor_path}")
    
    # Create dataloaders
    print("\n[2/5] Creating dataloaders...")
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
    
    # Create trainer
    print("\n[4/5] Training...")
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
        checkpoint_path = os.path.join(config.model_save_path, 'best_model_phase1.pt')
        trainer.save_checkpoint(checkpoint_path)
        print(f"\n  Phase 1 checkpoint saved to: {checkpoint_path}")
    else:
        trainer.train_phase(phase=2, epochs=epochs, early_stopping_patience=config.early_stopping_patience)
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
    # Decoder Evaluation (Post-Training)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Evaluating Decoder Reconstruction")
    print("=" * 60)
    
    # Reload best model
    final_checkpoint_path = os.path.join(config.model_save_path, f'best_model_phase{args.phase}.pt')
    if os.path.exists(final_checkpoint_path):
        print(f"Loading best model: {final_checkpoint_path}")
        checkpoint = torch.load(final_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.phase == 2:
            model.set_second_phase()
        else:
            model.set_first_phase()
    
    model.eval()
    
    feature_names = {
        'numerical': config.numerical_features,
        'categorical': config.categorical_features
    }
    decoder_evaluator = DecoderEvaluator(feature_names=feature_names)
    decoder_metrics = decoder_evaluator.evaluate(model, val_loader, device=device)
    decoder_evaluator.print_report(decoder_metrics)
    
    # ---------------------------------------------------------
    # t-SNE Visualization (Post-Training)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating t-SNE Visualization")
    print("=" * 60)
    
    print("Extracting embeddings for visualization...")
    
    viz_dataset = TensorDataset(torch.FloatTensor(train_num), torch.LongTensor(train_cat))
    viz_loader = DataLoader(viz_dataset, batch_size=config.batch_size, shuffle=False)
    
    embeddings_list = []
    
    with torch.no_grad():
        for x_n, x_c in viz_loader:
            x_n = x_n.to(device)
            x_c = x_c.to(device)
            outputs = model(x_n, x_c, return_all=True)
            embeddings_list.append(outputs['z'].cpu().numpy())
    
    z = np.concatenate(embeddings_list, axis=0)
    class_labels = train_target
    
    # Subsample for t-SNE if too large
    n_samples = min(5000, len(z))
    if len(z) > n_samples:
        print(f"  Subsampling {n_samples} from {len(z)} for t-SNE...")
        indices = np.random.choice(len(z), n_samples, replace=False)
        z_sub = z[indices]
        labels_sub = class_labels[indices]
    else:
        z_sub = z
        labels_sub = class_labels
    
    print(f"Calculating t-SNE for Z-Space ({z_sub.shape})...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z_sub)
    
    plot_tsne(
        z_2d, labels_sub,
        os.path.join(config.model_save_path, f'loan_tsne_phase{args.phase}_z.png'),
        f"Phase {args.phase} Z-Space (Loan Status)"
    )
    
    # P-Space visualization (Phase 2 only)
    if args.phase == 2:
        pspace_list = []
        with torch.no_grad():
            for x_n, x_c in viz_loader:
                x_n = x_n.to(device)
                x_c = x_c.to(device)
                outputs = model(x_n, x_c, return_all=True)
                pspace_list.append(outputs['p_space'].cpu().numpy())
        
        p_space = np.concatenate(pspace_list, axis=0)
        if len(p_space) > n_samples:
            p_sub = p_space[indices]
        else:
            p_sub = p_space
        
        print(f"Calculating t-SNE for P-Space ({p_sub.shape})...")
        tsne2 = TSNE(n_components=2, perplexity=30, random_state=42)
        p_2d = tsne2.fit_transform(p_sub)
        
        plot_tsne(
            p_2d, labels_sub,
            os.path.join(config.model_save_path, f'loan_tsne_phase{args.phase}_pspace.png'),
            f"Phase {args.phase} P-Space (Loan Status)"
        )
    
    # ---------------------------------------------------------
    # Sample Explanation (Phase 2 only)
    # ---------------------------------------------------------
    if args.phase == 2:
        print("\n" + "=" * 60)
        print("Generating Sample Explanation")
        print("=" * 60)
        
        explainer = PrototypeExplainer(
            model=model,
            preprocessor=preprocessor,
            prototype_descriptions=PROTOTYPE_DESCRIPTIONS,
            device=device
        )
        
        report_generator = LoanDecisionReportGenerator(explainer)
        
        # Explain first sample
        sample_num = train_num[0]
        sample_cat = train_cat[0]
        
        explanation = explainer.explain_single(sample_num, sample_cat, top_k=3)
        report = report_generator.generate_report(explanation, applicant_id="SAMPLE-0")
        print(report)
    
    print(f"\nVisualization saved to: {config.model_save_path}")


if __name__ == '__main__':
    main()
