"""
Main entry point for training baseline Conformer model (EEG-only, no gaze integration)
This serves as a baseline comparison for gaze-integrated models.
"""
import os
import sys
import torch
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.paths import PathConfig
from config.hyperparameters import get_default_hyp
from data.dataloader_builder import get_dataloaders_fixed
from EEGConformer.conformer import Conformer
from training.trainer_conformer_baseline import train_epoch_conformer_baseline
from training.early_stopping import EarlyStopping
from utils.debugger import DataDebugger
from utils.statistics_tracker import TrainingStatistics
from sklearn.metrics import classification_report
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                            balanced_accuracy_score)


def evaluate_conformer_baseline(model, eval_loader, device, class_weights=None):
    """
    Evaluation function specifically for Conformer baseline model
    
    Args:
        model: Conformer model
        eval_loader: Evaluation data loader
        device: Device (cuda/cpu)
        class_weights: Optional class weights for loss computation
        
    Returns:
        Tuple of (eval_stats, all_labels, all_preds, all_files)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_files = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)
            
            # Get batch files
            batch_files = []
            if 'file' in batch:
                for f in batch['file']:
                    if isinstance(f, (bytes, bytearray)):
                        try:
                            f = f.decode('utf-8', errors='ignore')
                        except:
                            f = str(f)
                    batch_files.append(os.path.basename(str(f)))
            
            # Add channel dimension if needed
            if eeg.dim() == 3:
                eeg = eeg.unsqueeze(1)
            
            # Forward pass (Conformer returns (features, logits))
            outputs = model(eeg)
            if isinstance(outputs, tuple):
                features, logits = outputs
            else:
                logits = outputs
            
            # Compute loss
            if class_weights is not None:
                loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
            else:
                loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_labels.extend(labels_np)
            all_preds.extend(preds)
            all_files.extend(batch_files)
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    avg_loss = total_loss / max(num_batches, 1)
    acc = (all_preds == all_labels).mean() * 100
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    eval_stats = {
        'loss': avg_loss,
        'acc': acc,
        'balanced_acc': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall
    }
    
    return eval_stats, all_labels, all_preds, all_files


def train_baseline_conformer(output_suffix=None, hyp_overrides=None):
    """
    Train baseline Conformer model (EEG-only)
    
    Args:
        output_suffix: Optional suffix for output directory
        hyp_overrides: Optional dictionary to override hyperparameters
    """
    DataDebugger.print_header("BASELINE CONFORMER (EEG-ONLY)", width=80)
    
    # Setup directories
    PathConfig.setup_directories()
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get default hyperparameters
    hyps = get_default_hyp()
    
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
    output_dir = "conformer_baseline_results"
    if output_suffix:
        output_dir = f"{output_dir}_{output_suffix}"
    
    # Initialize statistics tracker
    stats_tracker = TrainingStatistics(output_dir=output_dir)
    print(f"Statistics will be saved to: {stats_tracker.run_dir}")
    
    # Build dataloaders (use same gaze-matched data for fair comparison)
    print("\n" + "=" * 80)
    print("DATA LOADING (using gaze-matched samples for fair comparison)")
    print("=" * 80)
    try:
        train_loader, eval_loader, gaze_stats = get_dataloaders_fixed(
            data_dir=PathConfig.DATA_DIR,
            batch_size=hyps.batch_size,
            seed=hyps.seed,
            target_length=hyps.target_length,
            gaze_json_dir=PathConfig.GAZE_JSON_DIR,
            only_matched=True,  # Use same samples as gaze models
            suffixes_to_strip=hyps.suffixes_to_strip,
            eeg_sampling_rate=hyps.eeg_sampling_rate,
            train_subdir=PathConfig.TRAIN_SUBDIR,
            eval_subdir=PathConfig.EVAL_SUBDIR
        )
        print(f"✓ Loaded {len(train_loader.dataset)} train samples, {len(eval_loader.dataset)} eval samples")
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
    
    # Initialize baseline Conformer model
    try:
        sample_batch = next(iter(train_loader))
        n_chan = sample_batch['eeg'].shape[1]
        print(f"\nDetected {n_chan} channels from data")
    except:
        n_chan = hyps.n_channels
        print(f"\nUsing default {n_chan} channels")
    
    # Conformer parameters - Reduced to prevent overfitting
    emb_size = 20  # Embedding size (reduced from 40)
    depth = 2    # Transformer depth (reduced from 6)
    
    model = Conformer(
        emb_size=emb_size,
        depth=depth,
        n_classes=hyps.n_outputs,
        n_channels=n_chan
    ).to(device)
    
    print(f"\nBaseline Conformer Configuration:")
    print(f"  Type: EEG-only (no gaze integration)")
    print(f"  Channels: {n_chan}")
    print(f"  Embedding size: {emb_size}")
    print(f"  Transformer depth: {depth}")
    print(f"  Outputs: {hyps.n_outputs}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Lower learning rate and add weight decay for better regularization
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=5e-5,  # Reduced from 1e-4
        weight_decay=1e-4  # Added weight decay for regularization
    )
    # Scheduler: uses 'max' mode because we negate loss (so lower loss = higher negated score)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.1
    )
    
    # Early stopping and best model checkpointing (use same metric for consistency)
    es_path = "best_model_conformer_baseline.pth"
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
        # Add channel dimension
        if test_eeg.dim() == 3:
            test_eeg = test_eeg.unsqueeze(1)
        outputs = model(test_eeg)
        if isinstance(outputs, tuple):
            features, logits = outputs
            print(f"\nModel forward OK, features shape: {features.shape}, logits shape: {logits.shape}")
        else:
            print(f"\nModel forward OK, logits shape: {outputs.shape}")
    except Exception as e:
        print("Model forward error:", e)
        traceback.print_exc()
        return None
    
    # Training loop - Initialize best metric tracking
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
    
    print(f"\nStarting baseline training for {hyps.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(hyps.epochs):
        DataDebugger.print_header(f"EPOCH {epoch+1}/{hyps.epochs} [CONFORMER BASELINE]", 
                                  width=60, char='-')
        
        # Train
        train_stats = train_epoch_conformer_baseline(
            model, train_loader, optimizer, device,
            class_weights=class_weights,
            stats_tracker=stats_tracker,
            epoch=epoch
        )
        
        # Evaluate
        eval_stats, ev_labels, ev_preds, ev_files = evaluate_conformer_baseline(
            model, eval_loader, device, class_weights=class_weights
        )
        
        # Update learning rate scheduler based on validation metric
        if hyps.early_stop_metric == 'eval_loss':
            metric_for_sched = -eval_stats['loss']  # Negate: lower loss = better = higher score
        else:
            metric_for_sched = eval_stats['balanced_acc'] if hyps.early_stop_metric == 'balanced_acc' else eval_stats['macro_f1']
        scheduler.step(metric_for_sched)
        
        # Record epoch statistics
        epoch_data = stats_tracker.record_epoch(epoch, train_stats, eval_stats, model)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary [CONFORMER BASELINE]:")
        print(f"  Train: Loss={train_stats['loss']:.4f} | Acc={train_stats['acc']:.2f}%")
        print(f"  Eval:  Loss={eval_stats['loss']:.4f} | Acc={eval_stats['acc']:.2f}% | "
              f"Balanced Acc={eval_stats['balanced_acc']:.4f} | "
              f"Macro F1={eval_stats['macro_f1']:.4f}")
        print(f"  LR:    {train_stats['lr']:.2e}")
        
        report = classification_report(ev_labels, ev_preds, digits=4, zero_division=0)
        print(f"\nClassification Report for Epoch {epoch+1}:\n{report}")
        
        # Prepare validation metric for both early stopping and best model saving
        if hyps.early_stop_metric == 'eval_loss':
            metric_value = -eval_stats['loss']  # Negate: lower loss = higher score
            current_metric = eval_stats['loss']
        elif hyps.early_stop_metric == 'balanced_acc':
            metric_value = eval_stats['balanced_acc']
            current_metric = eval_stats['balanced_acc']
        else:  # macro_f1
            metric_value = eval_stats['macro_f1']
            current_metric = eval_stats['macro_f1']
        
        # Update best metric for tracking
        if hyps.early_stop_metric == 'eval_loss':
            is_better = current_metric < best_metric
        else:
            is_better = current_metric > best_metric
        
        if is_better:
            best_metric = current_metric
        
        # Early stopping (handles both early stop detection AND best model checkpointing)
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
    print(f"FINAL EVALUATION [CONFORMER BASELINE]")
    print("=" * 80)
    
    final_stats, final_labels, final_preds, final_files = evaluate_conformer_baseline(
        model, eval_loader, device, class_weights=class_weights
    )
    
    # Print final results
    print(f"\nFinal Results [CONFORMER BASELINE]:")
    print(f"  Accuracy: {final_stats['acc']:.2f}%")
    print(f"  Balanced Accuracy: {final_stats['balanced_acc']:.4f}")
    print(f"  Macro F1: {final_stats['macro_f1']:.4f}")
    print(f"  Weighted F1: {final_stats['weighted_f1']:.4f}")
    print(f"  Precision: {final_stats['precision']:.4f}")
    print(f"  Recall: {final_stats['recall']:.4f}")
    
    # Save all results
    print("\n" + "=" * 80)
    print(f"SAVING RESULTS [CONFORMER BASELINE]")
    print("=" * 80)
    
    stats_tracker.save_final_results(model=model)
    
    # Save summary report
    with open(stats_tracker.run_dir / 'training_summary.txt', 'w') as f:
        f.write(f"TRAINING SUMMARY REPORT - BASELINE CONFORMER (EEG-ONLY)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model type: Baseline Conformer (no gaze integration)\n")
        f.write(f"Total epochs: {len(stats_tracker.epoch_stats)}\n")
        f.write(f"Best {metric_name}: {best_metric:.4f}\n")
        f.write(f"\nHyperparameters:\n")
        f.write(f"  Learning rate: 5e-5 (reduced for stability)\n")
        f.write(f"  Weight decay: 1e-4 (L2 regularization)\n")
        f.write(f"  Batch size: {hyps.batch_size}\n")
        f.write(f"  Early stop metric: {hyps.early_stop_metric}\n")
        f.write(f"  Embedding size: {emb_size}\n")
        f.write(f"  Transformer depth: {depth}\n")
        f.write(f"  Dropout: 0.6 (increased from 0.5)\n")
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
    
    print(f"\nCONFORMER BASELINE training complete!")
    print(f"Best {metric_name}: {best_metric:.4f}")
    print(f"Results saved to: {stats_tracker.run_dir}")
    
    return {
        'model_type': 'conformer_baseline',
        'best_metric': best_metric,
        'metric_name': metric_name,
        'final_stats': final_stats,
        'output_dir': str(stats_tracker.run_dir)
    }


def main(output_suffix=None, hyp_overrides=None):
    """
    Main function - trains baseline Conformer model
    
    Args:
        output_suffix: Optional suffix for output directory
        hyp_overrides: Optional dictionary to override hyperparameters
    """
    return train_baseline_conformer(output_suffix, hyp_overrides)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline Conformer model (EEG-only)')
    
    # Output arguments
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
    if args.seed is not None:
        hyp_overrides['seed'] = args.seed
    
    main(output_suffix=args.suffix,
         hyp_overrides=hyp_overrides if hyp_overrides else None)
