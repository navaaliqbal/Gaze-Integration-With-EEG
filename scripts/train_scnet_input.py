#!/usr/bin/env python3
"""
Train SCNet with input gaze integration
Convenience script for running SCNet model with input integration type
"""
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_scnet import main as train_scnet_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SCNet with input gaze integration')
    
    # Output configuration
    parser.add_argument('--suffix', type=str, default='scnet_input',
                       help='Suffix for output directory (default: scnet_input)')
    
    # Hyperparameter arguments
    parser.add_argument('--lr', '--learning-rate', type=float, default=None,
                       dest='learning_rate',
                       help='Learning rate (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       dest='batch_size',
                       help='Batch size (default: from config)')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience (default: from config)')
    parser.add_argument('--gaze-weight', type=float, default=None,
                       dest='gaze_weight',
                       help='Gaze loss weight (default: from config)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (default: from config)')
    
    args = parser.parse_args()
    
    # Build hyperparameter overrides dictionary
    hyp_overrides = {}
    if args.learning_rate is not None:
        hyp_overrides['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        hyp_overrides['epochs'] = args.epochs
    if args.batch_size is not None:
        hyp_overrides['batch_size'] = args.batch_size
    if args.patience is not None:
        hyp_overrides['patience'] = args.patience
    if args.gaze_weight is not None:
        hyp_overrides['gaze_weight'] = args.gaze_weight
    if args.seed is not None:
        hyp_overrides['seed'] = args.seed
    
    print("=" * 60)
    print("SCNet Input Gaze Integration Training".center(60))
    print("=" * 60)
    print(f"Model: SCNet")
    print(f"Integration Type: Input")
    print(f"Output Suffix: {args.suffix}")
    if hyp_overrides:
        print(f"Custom Hyperparameters: {', '.join(hyp_overrides.keys())}")
    print("=" * 60 + "\n")
    
    # Run training with input integration
    train_scnet_main(
        integration_type='input', 
        output_suffix=args.suffix,
        hyp_overrides=hyp_overrides if hyp_overrides else None
    )