#!/usr/bin/env python3
"""
Inference script for Prototype-based Neuro-Symbolic Model.

Usage:
    python inference.py --model_path checkpoints/best_model.pt --sample_idx 0
    python inference.py --model_path checkpoints/best_model.pt --input_csv custom_data.csv
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
from src.data.preprocessor import LoanDataPreprocessor
from src.data.dataset import create_test_loader
from src.models.model import PrototypeNetwork, create_model_from_config
from src.explainability.prototype_explainer import PrototypeExplainer
from src.explainability.report_generator import LoanDecisionReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Prototype Network')
    
    parser.add_argument('--model_path', type=str, default='/workspace/checkpoints/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--preprocessor_path', type=str, default='/workspace/checkpoints/preprocessor.pkl',
                        help='Path to preprocessor')
    parser.add_argument('--test_path', type=str, default='/workspace/data/test.csv',
                        help='Path to test data')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Index of sample to explain (if set, generates detailed report)')
    parser.add_argument('--output_path', type=str, default='/workspace/predictions.csv',
                        help='Path to save predictions')
    parser.add_argument('--generate_report', action='store_true',
                        help='Generate detailed report for sample')
    parser.add_argument('--language', type=str, default='both', choices=['ko', 'en', 'both'],
                        help='Report language')
    
    return parser.parse_args()


def load_model_and_preprocessor(model_path: str, preprocessor_path: str, device: str):
    """Load trained model and preprocessor."""
    # Load preprocessor
    preprocessor = LoanDataPreprocessor.load(preprocessor_path)
    
    # Create config
    config = get_default_config()
    config.categorical_cardinalities = preprocessor.get_cardinalities()
    config.device = device
    
    # Create model
    model = create_model_from_config(config)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Best validation AUC: {checkpoint.get('best_val_auc', 'N/A')}")
    
    return model, preprocessor, config


def predict_batch(model, x_num: np.ndarray, x_cat: np.ndarray, device: str) -> np.ndarray:
    """Generate predictions for batch of samples."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        x_num_t = torch.from_numpy(x_num).float().to(device)
        x_cat_t = torch.from_numpy(x_cat).long().to(device)
        
        outputs = model(x_num_t, x_cat_t, return_all=False)
        predictions = outputs['probabilities'].cpu().numpy()
    
    return predictions


def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and preprocessor
    print("\nLoading model and preprocessor...")
    model, preprocessor, config = load_model_and_preprocessor(
        args.model_path, args.preprocessor_path, device
    )
    
    # Load test data
    print(f"\nLoading test data from: {args.test_path}")
    test_df = pd.read_csv(args.test_path)
    test_num, test_cat = preprocessor.transform(test_df)
    print(f"Test samples: {len(test_df)}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predict_batch(model, test_num, test_cat, device)
    
    # Save predictions
    if 'id' in test_df.columns:
        result_df = pd.DataFrame({
            'id': test_df['id'],
            'loan_status': predictions
        })
    else:
        result_df = pd.DataFrame({
            'id': range(len(predictions)),
            'loan_status': predictions
        })
    
    result_df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to: {args.output_path}")
    
    # Generate detailed report for sample if requested
    if args.sample_idx is not None or args.generate_report:
        sample_idx = args.sample_idx if args.sample_idx is not None else 0
        
        print(f"\n{'='*60}")
        print(f"DETAILED EXPLANATION FOR SAMPLE {sample_idx}")
        print(f"{'='*60}")
        
        # Create explainer
        explainer = PrototypeExplainer(
            model=model,
            preprocessor=preprocessor,
            prototype_descriptions=PROTOTYPE_DESCRIPTIONS,
            device=device
        )
        
        # Generate explanation
        explanation = explainer.explain_single(
            test_num[sample_idx], 
            test_cat[sample_idx],
            top_k=3
        )
        
        # Generate report
        report_generator = LoanDecisionReportGenerator(explainer)
        report = report_generator.generate_report(
            explanation, 
            applicant_id=f"TEST-{sample_idx}",
            language=args.language
        )
        
        print(report)
        
        # Also print structured summary
        summary = report_generator.generate_summary(explanation)
        print("\nStructured Summary (JSON-like):")
        print(f"  Prediction: {summary['prediction']:.4f}")
        print(f"  Risk Category: {summary['risk_category']}")
        print(f"  Confidence: {summary['confidence']:.4f}")
        print(f"  Top Prototypes:")
        for proto in summary['top_prototypes']:
            print(f"    - {proto['name']}: {proto['similarity']:.3f} similarity, {proto['contribution']:.4f} contribution")
        print(f"  Key Factors:")
        for factor in summary['key_factors']:
            print(f"    - [{factor['impact'].upper()}] {factor['description']}")
    
    print("\nInference complete!")


if __name__ == '__main__':
    main()
