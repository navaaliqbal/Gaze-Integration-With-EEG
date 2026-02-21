"""
Main entry point for training Conformer models with different gaze integration approaches
Supports: baseline (no gaze), input, output, and combined (both) gaze integration
"""
import os
import sys
import torch
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.paths import PathConfig
from config.hyperparameters import get_hyp_for_integration, get_default_hyp
from data.dataloader_builder import get_dataloaders_fixed
from models.model_factory import create_model
from EEGConformer.conformer import Conformer
from training.trainer_conformer_input import train_epoch_conformer_input
from training.trainer_conformer_output import train_epoch_conformer_output
from training.trainer_conformer_combined import train_epoch_conformer_combined
from training.trainer_conformer_baseline import train_epoch_conformer_baseline
from training.metrics import evaluate_model_comprehensive, compute_gaze_loss_scale
from training.early_stopping import EarlyStopping
from utils.debugger import DataDebugger
from utils.statistics_tracker import TrainingStatistics
from sklearn.metrics import classification_report


def print_epoch_summary(epoch, train_stats, eval_stats, integration_type, model=None):
    """Print epoch summary for Conformer training"""
    print("\n" + "=" * 60)
    print(f"Epoch {epoch+1} Summary [Conformer-{integration_type.upper()}]:")
    
    # Print losses
    if 'cls_loss' in train_stats:
        print(f"  Classification Loss = {train_stats['cls_loss']:.4f}")
    if 'gaze_loss' in train_stats:
        print(f"  Gaze Loss = {train_stats['gaze_loss']:.4f}")
    print(f"  Total Loss = {train_stats['loss']:.4f}")
    
    # Print accuracies
    print(f"  Train Accuracy = {train_stats['acc']:.2f}%")
    print(f"  Eval Accuracy = {eval_stats['acc']:.2f}% | "
          f"Balanced Acc = {eval_stats['balanced_acc']:.4f} | "
          f"Macro F1 = {eval_stats['macro_f1']:.4f}")
    
    # Gaze-specific info
    if integration_type in ['input', 'combined']:
        if 'gaze_alpha' in train_stats:
            print(f"  Gaze Alpha: {train_stats['gaze_alpha']:.3f}")
            if 'gaze_alpha_grad' in train_stats:
                print(f"  Gaze Alpha Grad: {train_stats['gaze_alpha_grad']:.6f}")
    
    if 'batches_with_gaze' in train_stats:
        print(f"  Gaze Batches: {train_stats['batches_with_gaze']}")
    
    # Learning rate
    print(f"  LR: {train_stats.get('lr', 0):.2e}")
    if 'gaze_lr' in train_stats:
        print(f"  Gaze LR: {train_stats['gaze_lr']:.2e}")
    
    print("=" * 60)


def train_conformer(integration_type='baseline', output_suffix=None, hyp_overrides=None):
    """
    Train Conformer with specified gaze integration approach
    
    Args:
        integration_type: 'baseline', 'input', 'output', or 'combined'
        output_suffix: Optional suffix for output directory
        hyp_overrides: Optional dictionary to override hyperparameters
    """
    DataDebugger.print_header(f"CONFORMER GAZE INTEGRATION: {integration_type.upper()}", width=80)
    
    # Setup directories
    PathConfig.setup_directories()
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get hyperparameters
    if integration_type == 'baseline':
        hyps = get_default_hyp()
    else:
        hyps = get_hyp_for_integration(integration_type if integration_type != 'combined' else 'both')
    
    # Apply command-line overrides if provided
    if hyp_overrides:
        print(f"\nApplying hyperparameter overrides:")
        for key, value in hyp_overrides.items():
            if hasattr(hyps, key):
                old_value = getattr(hyps, key)
                setattr(hyps, key, value)
                print(f"  {key}: {old_value} → {value}")
            else:
                print(f"  Warning: Unknown hyperparameter '{key}' ignored")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"conformer_{integration_type}_results"
    if output_suffix:
        output_dir = f"{output_dir}_{output_suffix}"
    
    # Initialize statistics tracker
    stats_tracker = TrainingStatistics(output_dir=output_dir)
    print(f"Statistics will be saved to: {stats_tracker.run_dir}")
    
    # Build dataloaders
    # For baseline and output: can use all data (gaze optional)
    # For input and combined: need gaze-matched data only
    only_matched = integration_type in ['input', 'combined']
    
    print("\n" + "=" * 80)
    if only_matched:
        print("DATA LOADING (gaze-matched samples only)")
    else:
        print("DATA LOADING (all samples, gaze optional)")
    print("=" * 80)
    
    try:
        train_loader, eval_loader, gaze_stats = get_dataloaders_fixed(
            data_dir=PathConfig.DATA_DIR,
            batch_size=hyps.batch_size,
            seed=hyps.seed,
            target_length=hyps.target_length,
            gaze_json_dir=PathConfig.GAZE_JSON_DIR,
            only_matched=only_matched,
            suffixes_to_strip=hyps.suffixes_to_strip,
            eeg_sampling_rate=hyps.eeg_sampling_rate,
            train_subdir=PathConfig.TRAIN_SUBDIR,
            eval_subdir=PathConfig.EVAL_SUBDIR
        )
        print(f"✓ Loaded {len(train_loader.dataset)} train samples, {len(eval_loader.dataset)} eval samples")
        if gaze_stats:
            print(f"✓ Gaze data: {gaze_stats.get('matched_samples', 0)} samples with gaze")
    except Exception as e:
        print("Error building dataloaders:", e)
        traceback.print_exc()
        return None
    
    # Record initial class distributions
    print("\nRecording initial class distributions...")
    train_dist = stats_tracker.record_class_distribution(train_loader, "train")
    eval_dist = stats_tracker.record_class_distribution(eval_loader, "eval")
    
    print(f"  Train distribution: {dict(train_dist)}")
    print(f"  Eval distribution: {dict(eval_dist)}")
    
    # Get number of channels from data
    try:
        sample_batch = next(iter(train_loader))
        n_chan = sample_batch['eeg'].shape[1]
        print(f"\nDetected {n_chan} channels from data")
    except:
        n_chan = hyps.n_channels
        print(f"\nUsing default {n_chan} channels")
    
    # Model initialization based on integration type
    emb_size = 30  # Embedding size
    depth = 4      # Transformer depth
    
    if integration_type == 'baseline':
        # Baseline Conformer (no gaze)
        model = Conformer(
            emb_size=emb_size,
            depth=depth,
            n_classes=hyps.n_outputs,
            n_channels=n_chan
        ).to(device)
    else:
        # Use factory for gaze-integrated models
        model_type_map = {
            'input': 'input',
            'output': 'output', 
            'combined': 'both'
        }
        model = create_model(
            model_type='conformer',
            integration_type=model_type_map[integration_type],
            n_channels=n_chan,
            n_classes=hyps.n_outputs,
            original_time_length=hyps.target_length,
            emb_size=emb_size,
            depth=depth,
            dropout_rate=0.2
        ).to(device)
    
    print(f"\nConformer {integration_type.upper()} Configuration:")
    print(f"  Type: {integration_type}")
    print(f"  Channels: {n_chan}")
    print(f"  Embedding size: {emb_size}")
    print(f"  Transformer depth: {depth}")
    print(f"  Outputs: {hyps.n_outputs}")
    if hasattr(model, 'gaze_alpha'):
        print(f"  Initial gaze alpha: {model.gaze_alpha.item():.3f}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup optimizers
    # For input/combined: separate optimizer for gaze_alpha
    if integration_type in ['input', 'combined']:
        model_params = [p for name, p in model.named_parameters() if 'gaze_alpha' not in name]
        gaze_alpha_param = [p for name, p in model.named_parameters() if 'gaze_alpha' in name]
        
        optimizer = torch.optim.Adam(model_params, lr=hyps.learning_rate, weight_decay=2e-4)
        gaze_optimizer = torch.optim.Adam(gaze_alpha_param, lr=hyps.learning_rate * 10) if gaze_alpha_param else None
        
        print(f"  Main model LR: {hyps.learning_rate:.2e}")
        if gaze_optimizer:
            print(f"  Gaze alpha LR: {hyps.learning_rate * 10:.2e} (10x higher)")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=hyps.learning_rate, weight_decay=2e-4)
        gaze_optimizer = None
        print(f"  Learning rate: {hyps.learning_rate:.2e}")
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    gaze_scheduler = None
    if gaze_optimizer:
        gaze_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            gaze_optimizer, mode='max', patience=5, factor=0.5
        )
    
    # Early stopping
    es_path = f"best_conformer_{integration_type}.pth"
    es = EarlyStopping(patience=hyps.patience, path=es_path, verbose=True)
    print(f"\nBest model checkpointing will monitor: {hyps.early_stop_metric}")
    if hyps.use_early_stopping:
        print(f"Early stopping: ENABLED (patience={hyps.patience})")
    else:
        print("Early stopping: DISABLED (will train for all epochs)")
    
    # Test forward pass
    try:
        sample_batch = next(iter(train_loader))
        test_eeg = sample_batch['eeg'].to(device)[:2]
        has_gaze = 'gaze' in sample_batch and sample_batch['gaze'] is not None
        test_gaze = sample_batch['gaze'].to(device)[:2] if has_gaze else None
        
        if integration_type == 'baseline':
            if test_eeg.dim() == 3:
                test_eeg = test_eeg.unsqueeze(1)
            outputs = model(test_eeg)
        elif integration_type == 'input':
            outputs = model(test_eeg, test_gaze)
        elif integration_type in ['output', 'combined']:
            outputs = model(test_eeg, test_gaze if integration_type == 'combined' else None, return_attention=True)
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
            print(f"\nModel forward OK, logits shape: {logits.shape}")
            if 'attention_map' in outputs:
                print(f"Attention map shape: {outputs['attention_map'].shape}")
        elif isinstance(outputs, tuple):
            print(f"\nModel forward OK, output shapes: {[o.shape for o in outputs]}")
        else:
            print(f"\nModel forward OK, output shape: {outputs.shape}")
    except Exception as e:
        print("Model forward error:", e)
        traceback.print_exc()
        return None
    
    # Compute gaze loss scaling if needed
    gaze_loss_scale = 1.0
    if integration_type in ['output', 'combined'] and hyps.use_gaze_loss_scaling:
        gaze_loss_scale, scale_metrics = compute_gaze_loss_scale(
            model, train_loader, device, hyps.gaze_loss_type
        )
        print(f"\n{'='*80}")
        print(f"FIXED SCALING FACTOR: {gaze_loss_scale:.2f}")
        print(f"{'='*80}")
    
    # Training loop setup
    if hyps.early_stop_metric == 'eval_loss':
        best_metric = float('inf')
        metric_name = 'Validation Loss'
    else:
        best_metric = 0.0
        metric_name = 'Balanced Accuracy' if hyps.early_stop_metric == 'balanced_acc' else 'Macro F1'
    
    # Class weights
    class_counts = list(train_dist.values())
    total = sum(class_counts)
    class_weights_raw = [total / c for c in class_counts]
    class_weights_raw[1] = class_weights_raw[1] * 1.5  # Amplify minority class
    class_weights = torch.tensor(class_weights_raw, dtype=torch.float32).to(device)
    print(f"\nClass weights for loss: {class_weights.tolist()}")
    print(f"Best model checkpoint saved when {metric_name} improves")
    
    print(f"\nStarting Conformer {integration_type.upper()} training for {hyps.epochs} epochs...")
    print("=" * 80)
    
    # Training loop
    for epoch in range(hyps.epochs):
        DataDebugger.print_header(f"EPOCH {epoch+1}/{hyps.epochs} [Conformer-{integration_type.upper()}]", 
                                  width=60, char='-')
        
        # Training based on integration type
        if integration_type == 'baseline':
            train_stats = train_epoch_conformer_baseline(
                model, train_loader, optimizer, device,
                class_weights=class_weights,
                stats_tracker=stats_tracker,
                epoch=epoch
            )
        elif integration_type == 'input':
            train_stats = train_epoch_conformer_input(
                model, train_loader, optimizer, device,
                class_weights=class_weights,
                stats_tracker=stats_tracker,
                epoch=epoch,
                gaze_optimizer=gaze_optimizer
            )
        elif integration_type == 'output':
            train_stats = train_epoch_conformer_output(
                model, train_loader, optimizer, device,
                gaze_weight=hyps.gaze_weight,
                gaze_loss_type=hyps.gaze_loss_type,
                class_weights=class_weights,
                stats_tracker=stats_tracker,
                epoch=epoch,
                gaze_loss_scale=gaze_loss_scale,
                use_focal_loss=True,
                focal_gamma=2.0,
                label_smoothing=0.1
            )
        elif integration_type == 'combined':
            train_stats = train_epoch_conformer_combined(
                model, train_loader, optimizer, device,
                gaze_weight=hyps.gaze_weight,
                gaze_loss_type=hyps.gaze_loss_type,
                gaze_loss_scale=gaze_loss_scale,
                class_weights=class_weights,
                stats_tracker=stats_tracker,
                epoch=epoch,
                gaze_optimizer=gaze_optimizer,
                use_focal_loss=True,
                focal_gamma=2.0,
                label_smoothing=0.1
            )
        
        # Evaluation based on integration type
        return_attention = integration_type in ['output', 'combined']
        eval_result = evaluate_model_comprehensive(
            model, eval_loader, device, stats_tracker, "eval",
            return_attention=return_attention,
            gaze_weight=hyps.gaze_weight if integration_type in ['output', 'combined'] else 0.0,
            gaze_loss_type=hyps.gaze_loss_type if integration_type in ['output', 'combined'] else 'mse',
            class_weights=class_weights,
            gaze_loss_scale=gaze_loss_scale if integration_type in ['output', 'combined'] else 1.0
        )
        
        if return_attention and len(eval_result) == 5:
            eval_stats, ev_labels, ev_preds, ev_files, ev_attention = eval_result
        else:
            eval_stats, ev_labels, ev_preds, ev_files = eval_result
        
        # Update schedulers
        if hyps.early_stop_metric == 'eval_loss':
            metric_for_sched = -eval_stats['loss']
        else:
            metric_for_sched = eval_stats['balanced_acc'] if hyps.early_stop_metric == 'balanced_acc' else eval_stats['macro_f1']
        
        scheduler.step(metric_for_sched)
        if gaze_scheduler:
            gaze_scheduler.step(metric_for_sched)
        
        # Record epoch statistics
        epoch_data = stats_tracker.record_epoch(epoch, train_stats, eval_stats, model)
        
        # Print summary
        print_epoch_summary(epoch, train_stats, eval_stats, integration_type, model)
        
        # Classification report
        report = classification_report(ev_labels, ev_preds, digits=4, zero_division=0)
        print(f"\nClassification Report for Epoch {epoch+1}:\n{report}")
        
        # Prepare metric for early stopping
        if hyps.early_stop_metric == 'eval_loss':
            metric_value = -eval_stats['loss']
            current_metric = eval_stats['loss']
            is_better = current_metric < best_metric
        else:
            metric_value = eval_stats['balanced_acc'] if hyps.early_stop_metric == 'balanced_acc' else eval_stats['macro_f1']
            current_metric = metric_value
            is_better = current_metric > best_metric
        
        if is_better:
            best_metric = current_metric
        
        # Early stopping
        es(metric_value, model, save_best_acc=True)
        if hyps.use_early_stopping and es.early_stop:
            print(f"  ⚠ Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    try:
        if os.path.exists(es_path):
            model.load_state_dict(torch.load(es_path))
            print(f"\n✓ Loaded best model from {es_path}")
            print(f"  Best {metric_name}: {best_metric:.4f}")
            if hasattr(model, 'gaze_alpha'):
                print(f"  Final gaze alpha: {model.gaze_alpha.item():.3f}")
    except Exception as e:
        print(f"Could not load best model: {e}")
    
    # Final evaluation
    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION [Conformer-{integration_type.upper()}]")
    print("=" * 80)
    
    final_result = evaluate_model_comprehensive(
        model, eval_loader, device, stats_tracker, "eval_final",
        return_attention=return_attention,
        gaze_weight=hyps.gaze_weight if integration_type in ['output', 'combined'] else 0.0,
        gaze_loss_type=hyps.gaze_loss_type if integration_type in ['output', 'combined'] else 'mse',
        class_weights=class_weights,
        gaze_loss_scale=gaze_loss_scale if integration_type in ['output', 'combined'] else 1.0
    )
    
    if return_attention and len(final_result) == 5:
        final_stats, final_labels, final_preds, final_files, final_attention = final_result
    else:
        final_stats, final_labels, final_preds, final_files = final_result
    
    # Print final results
    print(f"\nFinal Results [Conformer-{integration_type.upper()}]:")
    print(f"  Accuracy: {final_stats['acc']:.2f}%")
    print(f"  Balanced Accuracy: {final_stats['balanced_acc']:.4f}")
    print(f"  Macro F1: {final_stats['macro_f1']:.4f}")
    print(f"  Weighted F1: {final_stats['weighted_f1']:.4f}")
    print(f"  Precision: {final_stats['precision']:.4f}")
    print(f"  Recall: {final_stats['recall']:.4f}")
    if hasattr(model, 'gaze_alpha'):
        print(f"  Final Gaze Alpha: {model.gaze_alpha.item():.3f}")
    
    # Save results
    stats_tracker.save_final_results(model=model)
    
    # Save summary report
    with open(stats_tracker.run_dir / 'training_summary.txt', 'w') as f:
        f.write(f"TRAINING SUMMARY REPORT - CONFORMER {integration_type.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model type: Conformer with {integration_type} gaze integration\n")
        f.write(f"Total epochs: {len(stats_tracker.epoch_stats)}\n")
        f.write(f"Best {metric_name}: {best_metric:.4f}\n")
        if hasattr(model, 'gaze_alpha'):
            f.write(f"Final gaze alpha: {model.gaze_alpha.item():.3f}\n")
        f.write(f"\nHyperparameters:\n")
        f.write(f"  Learning rate: {hyps.learning_rate}\n")
        f.write(f"  Batch size: {hyps.batch_size}\n")
        f.write(f"  Embedding size: {emb_size}\n")
        f.write(f"  Transformer depth: {depth}\n")
        if integration_type in ['output', 'combined']:
            f.write(f"  Gaze weight: {hyps.gaze_weight}\n")
            f.write(f"  Gaze loss type: {hyps.gaze_loss_type}\n")
            f.write(f"  Gaze loss scale: {gaze_loss_scale:.2f}\n")
        f.write(f"\nDataset Statistics:\n")
        f.write(f"  Train samples: {len(train_loader.dataset)}\n")
        f.write(f"  Eval samples: {len(eval_loader.dataset)}\n")
        f.write(f"  Train class distribution: {dict(train_dist)}\n")
        f.write(f"  Eval class distribution: {dict(eval_dist)}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"  Accuracy: {final_stats['acc']:.2f}%\n")
        f.write(f"  Balanced Accuracy: {final_stats['balanced_acc']:.4f}\n")
        f.write(f"  Macro F1: {final_stats['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1: {final_stats['weighted_f1']:.4f}\n")
        f.write(f"\nResults saved to: {stats_tracker.run_dir}\n")
    
    print(f"\n✅ Conformer {integration_type.upper()} training complete!")
    print(f"Best {metric_name}: {best_metric:.4f}")
    print(f"Results saved to: {stats_tracker.run_dir}")
    
    return {
        'model': 'Conformer',
        'integration_type': integration_type,
        'best_metric': best_metric,
        'metric_name': metric_name,
        'final_stats': final_stats,
        'output_dir': str(stats_tracker.run_dir)
    }


def main(integration_type='baseline', output_suffix=None, hyp_overrides=None):
    """
    Main function - trains Conformer with specified gaze integration
    
    Args:
        integration_type: 'baseline', 'input', 'output', or 'combined'
        output_suffix: Optional suffix for output directory
        hyp_overrides: Optional dictionary to override hyperparameters
    """
    if integration_type not in ['baseline', 'input', 'output', 'combined']:
        print(f"Error: integration_type must be 'baseline', 'input', 'output', or 'combined'")
        return None
    
    return train_conformer(integration_type, output_suffix, hyp_overrides)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Conformer with gaze integration')
    
    # Model and integration arguments
    parser.add_argument('--type', type=str, default='baseline',
                       choices=['baseline', 'input', 'output', 'combined'],
                       help='Gaze integration type (default: baseline)')
    parser.add_argument('--suffix', type=str, default=None,
                       help='Suffix for output directory')
    
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
                       help='Gaze loss weight (for output/combined, default: from config)')
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
    
    main(integration_type=args.type, 
         output_suffix=args.suffix,
         hyp_overrides=hyp_overrides if hyp_overrides else None)
