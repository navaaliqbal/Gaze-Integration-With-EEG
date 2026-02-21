"""
Main entry point for training all three gaze integration approaches
"""
import os
import sys
import torch
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.paths import PathConfig
from config.hyperparameters import get_hyp_for_integration
from data.dataloader_builder import get_dataloaders_fixed
from models.model_factory import create_neurogate_model
from training.trainer import train_epoch
from training.metrics import evaluate_model_comprehensive
from training.early_stopping import EarlyStopping
from utils.debugger import DataDebugger
from utils.statistics_tracker import TrainingStatistics
from training.metrics import compute_gaze_loss_scale
from sklearn.metrics import classification_report
def train_integration_approach(integration_type='output', output_suffix=None):
    """
    Train a specific gaze integration approach
    
    Args:
        integration_type: 'input', 'output', or 'both'
        output_suffix: Optional suffix for output directory
    """
    DataDebugger.print_header(f"GAZE INTEGRATION: {integration_type.upper()}", width=80)
    
    # Setup directories
    PathConfig.setup_directories()
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get hyperparameters for this integration type
    hyps = get_hyp_for_integration(integration_type)
    
    # Create output directory with timestamp and integration type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_results_{integration_type}"
    if output_suffix:
        output_dir = f"{output_dir}_{output_suffix}"
    
    # Initialize statistics tracker
    stats_tracker = TrainingStatistics(output_dir=output_dir)
    print(f"Statistics will be saved to: {stats_tracker.run_dir}")
    
    # Build dataloaders
    try:
        train_loader, eval_loader, gaze_stats = get_dataloaders_fixed(
            data_dir=PathConfig.DATA_DIR,
            batch_size=hyps.batch_size,
            seed=hyps.seed,
            target_length=hyps.target_length,
            gaze_json_dir=PathConfig.GAZE_JSON_DIR,
            only_matched=True,
            suffixes_to_strip=hyps.suffixes_to_strip,
            eeg_sampling_rate=hyps.eeg_sampling_rate,
            train_subdir=PathConfig.TRAIN_SUBDIR,
            eval_subdir=PathConfig.EVAL_SUBDIR
        )
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
    
    # Initialize model for this integration type
    try:
        sample_batch = next(iter(train_loader))
        n_chan = sample_batch['eeg'].shape[1]
        print(f"\nDetected {n_chan} channels from data")
    except:
        n_chan = hyps.n_channels
        print(f"\nUsing default {n_chan} channels")
    
    model = create_neurogate_model(
        integration_type=integration_type,
        n_chan=n_chan,
        n_outputs=hyps.n_outputs,
        original_time_length=hyps.original_time_length
    ).to(device)
    
    print(f"\nModel Configuration:")
    print(f"  Type: {integration_type}")
    print(f"  Channels: {n_chan}")
    print(f"  Outputs: {hyps.n_outputs}")
    if hasattr(model, 'gaze_alpha'):
        print(f"  Initial gaze alpha: {model.gaze_alpha.item():.3f}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps.learning_rate)
    # Scheduler: uses 'max' mode because we negate loss (so lower loss = higher negated score)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.1
    )
    
    # Early stopping and best model checkpointing (use same metric for consistency)
    es_path = f"best_model_{integration_type}.pth"
    es = EarlyStopping(patience=hyps.patience, path=es_path, verbose=True)
    print(f"\nBest model checkpointing will monitor: {hyps.early_stop_metric}")
    if hyps.use_early_stopping:
        print(f"Early stopping: ENABLED (patience={hyps.patience})")
    else:
        print("Early stopping: DISABLED (will train for all epochs)")
    if hyps.early_stop_metric == 'eval_loss':
        print("  -> Using validation loss (recommended for detecting overfitting)")
    
    # Test forward pass
    try:
        sample_batch = next(iter(train_loader))
        test_eeg = sample_batch['eeg'].to(device)[:2]
        has_gaze = 'gaze' in sample_batch and sample_batch['gaze'] is not None
        test_gaze = sample_batch['gaze'].to(device)[:2] if has_gaze else None
        
        # Test based on integration type
        if integration_type == 'input':
            logits = model(test_eeg, test_gaze)
        elif integration_type == 'output':
            outputs = model(test_eeg, return_attention=True)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        elif integration_type == 'both':
            outputs = model(test_eeg, test_gaze, return_attention=True)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        print(f"\nModel forward OK, logits shape: {logits.shape}")
    except Exception as e:
        print("Model forward error:", e)
        traceback.print_exc()
        return None
    
    # gaze_loss_scaling - configurable
    if hyps.use_gaze_loss_scaling:
        gaze_loss_scale, scale_metrics = compute_gaze_loss_scale(
            model, train_loader, device, hyps.gaze_loss_type
        )
        print(f"\n" + "=" * 80)
        print(f"FIXED SCALING FACTOR FOR ENTIRE TRAINING: {gaze_loss_scale:.2f}")
        print(f"Effective gaze loss = gaze_weight × gaze_loss_scale × gaze_loss_raw")
        print(f"With gaze_weight={hyps.gaze_weight:.2f}: Effective scale = {hyps.gaze_weight * gaze_loss_scale:.2f}")
        print("=" * 80)
    else:
        gaze_loss_scale = 1.0
        print(f"\n" + "=" * 80)
        print(f"GAZE LOSS SCALING DISABLED - Using default scale: {gaze_loss_scale:.2f}")
        print(f"Effective gaze loss = gaze_weight × gaze_loss_raw")
        print(f"With gaze_weight={hyps.gaze_weight:.2f}")
        print("=" * 80)
    

    # Training loop - Initialize best metric tracking
    # Both early stopping AND best model checkpointing use this same metric
    if hyps.early_stop_metric == 'eval_loss':
        best_metric = float('inf')  # Lower is better for loss
        metric_name = 'Validation Loss'
        print(f"\n{'='*80}")
        print(f"MONITORING: {metric_name} (Primary choice - detects overfitting)")
        print(f"  Goal: Minimize validation loss")
        print(f"  Overfitting signal: Training loss decreases but validation loss increases")
        print(f"{'='*80}")
    elif hyps.early_stop_metric == 'balanced_acc':
        best_metric = 0.0  # Higher is better
        metric_name = 'Balanced Accuracy'
        print(f"\nMonitoring: {metric_name} (Alternative metric)")
    else:  # macro_f1
        best_metric = 0.0  # Higher is better
        metric_name = 'Macro F1'
        print(f"\nMonitoring: {metric_name} (Alternative metric)")
    
    class_counts = list(train_dist.values())
    total = sum(class_counts)
    class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float32).to(device)
    print(f"\nClass weights for loss: {class_weights.tolist()}")
    print(f"Best model checkpoint saved when {metric_name} improves")
     

    print(f"\nStarting training for {hyps.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(hyps.epochs):
        DataDebugger.print_header(f"EPOCH {epoch+1}/{hyps.epochs} [{integration_type.upper()}]", 
                                  width=60, char='-')
        
        # Train with appropriate trainer
        train_stats = train_epoch(
            model, train_loader, optimizer, device,
            gaze_weight=hyps.gaze_weight,
            gaze_loss_type=hyps.gaze_loss_type,
            class_weights=class_weights,
            stats_tracker=stats_tracker,
            epoch=epoch,
            gaze_loss_scale=gaze_loss_scale
        )
        
        # Evaluate
        # Determine if we need attention for evaluation
        return_attention = integration_type in ['output', 'both']
        if return_attention:
            eval_stats, ev_labels, ev_preds, ev_files, ev_attention_maps = evaluate_model_comprehensive(
                model, eval_loader, device, stats_tracker, "eval",
                return_attention=True,
                gaze_weight=hyps.gaze_weight,
                gaze_loss_type=hyps.gaze_loss_type,
                class_weights=class_weights,
                gaze_loss_scale=gaze_loss_scale
            )
        else:
            eval_stats, ev_labels, ev_preds, ev_files = evaluate_model_comprehensive(
                model, eval_loader, device, stats_tracker, "eval",
                return_attention=False,
                gaze_weight=hyps.gaze_weight,
                gaze_loss_type=hyps.gaze_loss_type,
                class_weights=class_weights,
                gaze_loss_scale=gaze_loss_scale
            )
        
        # Update learning rate scheduler based on validation metric
        # Scheduler uses same metric as early stopping for consistency
        if hyps.early_stop_metric == 'eval_loss':
            metric_for_sched = -eval_stats['loss']  # Negate: lower loss = better = higher score
        else:
            metric_for_sched = eval_stats['balanced_acc'] if hyps.early_stop_metric == 'balanced_acc' else eval_stats['macro_f1']
        scheduler.step(metric_for_sched)
        
        # Record epoch statistics
        epoch_data = stats_tracker.record_epoch(epoch, train_stats, eval_stats, model)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary [{integration_type.upper()}]:")
        print(f"  Train: Loss={train_stats['loss']:.4f} | Acc={train_stats['acc']:.2f}%")
        print(f"Cls Loss={train_stats['cls_loss']:.4f}")
        if 'gaze_loss' in train_stats and train_stats['gaze_loss'] > 0:
            print(f"         Gaze Loss={train_stats['gaze_loss']:.4f}")
        if 'gaze_alpha' in train_stats:
            print(f"         Gaze Alpha={train_stats['gaze_alpha']:.3f}")
        
        print(f"  Eval:  Loss={eval_stats['loss']:.4f} | Acc={eval_stats['acc']:.2f}% | "
              f"Balanced Acc={eval_stats['balanced_acc']:.4f} | "
              f"Macro F1={eval_stats['macro_f1']:.4f}")
        print(f"  Gaze:  {train_stats['gaze_samples']}/{train_stats['total_samples']} samples")
        print(f"  LR:    {train_stats['lr']:.2e}")
        report = classification_report(ev_labels, ev_preds, digits=4, zero_division=0)
        print(f"\nClassification Report for Epoch {epoch+1}:\n{report}")
        
        # Prepare validation metric for both early stopping and best model saving
        # IMPORTANT: Both use the SAME metric to ensure consistency
        # EarlyStopping expects "higher is better", so we negate loss
        if hyps.early_stop_metric == 'eval_loss':
            metric_value = -eval_stats['loss']  # Negate: lower loss = higher score
            current_metric = eval_stats['loss']
        elif hyps.early_stop_metric == 'balanced_acc':
            metric_value = eval_stats['balanced_acc']
            current_metric = eval_stats['balanced_acc']
        else:  # macro_f1
            metric_value = eval_stats['macro_f1']
            current_metric = eval_stats['macro_f1']
        
        # Update best metric for tracking (display only)
        if hyps.early_stop_metric == 'eval_loss':
            is_better = current_metric < best_metric
        else:
            is_better = current_metric > best_metric
        
        if is_better:
            best_metric = current_metric
        
        # Early stopping (handles both early stop detection AND best model checkpointing)
        # When validation metric improves, EarlyStopping automatically saves the model
        es(metric_value, model, save_best_acc=True)
        if hyps.use_early_stopping and es.early_stop:
            print(f"  ⚠ Early stopping triggered - validation metric stopped improving")
            break
    
    # Load best model
    try:
        if os.path.exists(es_path):
            model.load_state_dict(torch.load(es_path))
            print(f"\nLoaded best model (weights only)")
            print(f"  Best {metric_name}: {best_metric:.4f}")
    except Exception as e:
        print("Could not load best model:", e)
    
    # Final evaluation
    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION [{integration_type.upper()}]")
    print("=" * 80)
    
    if return_attention:
        final_stats, final_labels, final_preds, final_files, final_attention_maps = evaluate_model_comprehensive(
            model, eval_loader, device, stats_tracker, "eval_final",
            return_attention=True,
            gaze_weight=hyps.gaze_weight,
            gaze_loss_type=hyps.gaze_loss_type,
            class_weights=class_weights,
            gaze_loss_scale=gaze_loss_scale
        )
    else:
        final_stats, final_labels, final_preds, final_files = evaluate_model_comprehensive(
            model, eval_loader, device, stats_tracker, "eval_final",
            return_attention=False,
            gaze_weight=hyps.gaze_weight,
            gaze_loss_type=hyps.gaze_loss_type,
            class_weights=class_weights,
            gaze_loss_scale=gaze_loss_scale
        )
    # Print final results
    print(f"\nFinal Results [{integration_type.upper()}]:")
    print(f"  Accuracy: {final_stats['acc']:.2f}%")
    print(f"  Balanced Accuracy: {final_stats['balanced_acc']:.4f}")
    print(f"  Macro F1: {final_stats['macro_f1']:.4f}")
    print(f"  Weighted F1: {final_stats['weighted_f1']:.4f}")
    print(f"  Precision: {final_stats['precision']:.4f}")
    print(f"  Recall: {final_stats['recall']:.4f}")
    
    # Save all results
    print("\n" + "=" * 80)
    print(f"SAVING RESULTS [{integration_type.upper()}]")
    print("=" * 80)
    
    stats_tracker.save_final_results(model=model)
    
    # Save summary report
    with open(stats_tracker.run_dir / 'training_summary.txt', 'w') as f:
        f.write(f"TRAINING SUMMARY REPORT - {integration_type.upper()} INTEGRATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Integration type: {integration_type}\n")
        f.write(f"Total epochs: {len(stats_tracker.epoch_stats)}\n")
        f.write(f"Best {metric_name}: {best_metric:.4f}\n")
        f.write(f"\nHyperparameters:\n")
        f.write(f"  Learning rate: {hyps.learning_rate}\n")
        f.write(f"  Batch size: {hyps.batch_size}\n")
        f.write(f"  Gaze weight: {hyps.gaze_weight}\n")
        f.write(f"  Gaze loss type: {hyps.gaze_loss_type}\n")
        f.write(f"  Early stop metric: {hyps.early_stop_metric}\n")
        f.write(f"\nDataset Statistics:\n")
        f.write(f"  Train samples: {len(train_loader.dataset)}\n")
        f.write(f"  Eval samples: {len(eval_loader.dataset)}\n")
        f.write(f"  Train class distribution: {dict(train_dist)}\n")
        f.write(f"  Eval class distribution: {dict(eval_dist)}\n")
        f.write(f"\nModel Statistics:\n")
        f.write(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"  Accuracy: {final_stats['acc']:.2f}%\n")
        f.write(f"  Balanced Accuracy: {final_stats['balanced_acc']:.4f}\n")
        f.write(f"  Macro F1: {final_stats['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1: {final_stats['weighted_f1']:.4f}\n")
        f.write(f"\nResults saved to: {stats_tracker.run_dir}\n")
    
    print(f"\n{integration_type.upper()} integration training complete!")
    print(f"Best {metric_name}: {best_metric:.4f}")
    print(f"Results saved to: {stats_tracker.run_dir}")
    
    return {
        'integration_type': integration_type,
        'best_metric': best_metric,
        'metric_name': metric_name,
        'final_stats': final_stats,
        'output_dir': str(stats_tracker.run_dir)
    }

def main(integration_type='output', output_suffix=None):
    """
    Main function - trains specified integration type
    
    Args:
        integration_type: 'input', 'output', or 'both'
        output_suffix: Optional suffix for output directory
    """
    if integration_type not in ['input', 'output', 'both']:
        print(f"Error: integration_type must be 'input', 'output', or 'both'")
        return
    
    return train_integration_approach(integration_type, output_suffix)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NeuroGATE with gaze integration')
    parser.add_argument('--type', type=str, default='output',
                       choices=['input', 'output', 'both'],
                       help='Gaze integration type')
    parser.add_argument('--suffix', type=str, default=None,
                       help='Suffix for output directory')
    
    args = parser.parse_args()
    
    main(integration_type=args.type, output_suffix=args.suffix)