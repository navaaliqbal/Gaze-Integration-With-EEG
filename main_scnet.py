"""
Main entry point for training SCNet with gaze integration
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
from models.model_factory import create_scnet_model
from training.trainer import train_epoch
from training.metrics import evaluate_model_comprehensive
from training.early_stopping import EarlyStopping
from utils.debugger import DataDebugger
from utils.statistics_tracker import TrainingStatistics
from sklearn.metrics import classification_report

def print_epoch_summary(epoch, train_stats, eval_stats, integration_type, model=None):
    """Print epoch summary"""
    print("\n" + "=" * 60)
    print(f"Epoch {epoch+1} Summary [SCNet-{integration_type.upper()}]:")
    
    # Print individual losses
    train_str = f"  Train: Loss={train_stats['loss']:.4f}"
    if 'cls_loss' in train_stats:
        print(f"  Classification Loss = {train_stats['cls_loss']:.4f}")
    if 'gaze_loss' in train_stats:
        print(f"  Gaze Loss = {train_stats['gaze_loss']:.4f}")
    print(f"  Total Loss = {train_stats['loss']:.4f}")
    
    # Print accuracies
    print(f"  Train Accuracy = {train_stats['acc']:.2f}%")
    
    # Evaluation metrics
    print(f"  Eval Accuracy = {eval_stats['acc']:.2f}% | "
          f"Balanced Acc = {eval_stats['balanced_acc']:.4f} | "
          f"Macro F1 = {eval_stats['macro_f1']:.4f}")
    
    # Gaze sample info
    print(f"  Gaze Samples: {train_stats.get('gaze_samples', 0)}/{train_stats.get('total_samples', 0)}")
  
    # Learning rate
    print(f"  LR: {train_stats.get('lr', 0):.2e}")
    print("=" * 60)


def train_scnet_gaze(integration_type='input', output_suffix=None, hyp_overrides=None):
    """
    Train SCNet with gaze integration
    """
    DataDebugger.print_header(f"SCNet GAZE INTEGRATION: {integration_type.upper()}", width=80)
    
    # Setup directories
    PathConfig.setup_directories()
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get hyperparameters
    hyps = get_hyp_for_integration(integration_type)
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
    output_dir = f"scnet_results_{integration_type}"
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
    
    # Initialize SCNet model
    try:
        sample_batch = next(iter(train_loader))
        n_chan = sample_batch['eeg'].shape[1]  # Get actual number of channels
        print(f"\nSCNet using {n_chan} channels from data")
    except:
        n_chan = 22
        print(f"\nSCNet using default {n_chan} channels")
    
    model = create_scnet_model(
        integration_type=integration_type,
        n_chan=n_chan,
        n_outputs=hyps.n_outputs,
        original_time_length=hyps.original_time_length
    ).to(device)
    
    print(f"\nSCNet Configuration:")
    print(f"  Type: {integration_type}")
    print(f"  Channels: {n_chan}")
    print(f"  Outputs: {hyps.n_outputs}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.1
    )
    
    # Early stopping
    es_path = f"best_scnet_{integration_type}.pth"
    es = EarlyStopping(patience=hyps.patience, path=es_path, verbose=True)
    
    # Test forward pass
    try:
        sample_batch = next(iter(train_loader))
        test_eeg = sample_batch['eeg'].to(device)[:2]
        has_gaze = 'gaze' in sample_batch and sample_batch['gaze'] is not None
        outputs = None
        if integration_type == 'input':
            test_gaze = sample_batch['gaze'].to(device)[:2] if has_gaze else None
            logits = model(test_eeg, test_gaze)
        elif integration_type == 'output':
            outputs = model(test_eeg, return_attention=True)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        elif integration_type == 'both':
            outputs = model(test_eeg, test_gaze, return_attention=True)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        print(f"\nSCNet forward OK, logits shape: {logits.shape}")
        if outputs is not None and isinstance(outputs, dict) and 'attention_map' in outputs:
            print(f"Attention map shape: {outputs['attention_map'].shape}")
    except Exception as e:
        print("SCNet forward error:", e)
        traceback.print_exc()
        return None
    
    # Training loop
    best_acc = 0.0
    
    # Calculate class weights from actual distribution
    # Ensure we have weights for all classes (n_outputs)
    class_counts = [train_dist.get(i, 1) for i in range(hyps.n_outputs)]
    total = sum(class_counts)
    class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float32)
    print(f"\nClass distribution: {dict(train_dist)}")
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    
    print(f"\nStarting SCNet training for {hyps.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(hyps.epochs):
        print("\n" + "-" * 60)
        print(f"EPOCH {epoch+1}/{hyps.epochs} [SCNet-{integration_type.upper()}]".center(60))
        print("-" * 60)
        
        # Train with SCNet trainer
        train_stats = train_epoch(
            model, train_loader, optimizer, device,
            gaze_weight=hyps.gaze_weight,
            gaze_loss_type=hyps.gaze_loss_type,
            class_weights=class_weights,
            stats_tracker=stats_tracker,
            epoch=epoch
        )
        
        # Evaluate
        return_attention = integration_type in ['output', 'both']  # For future output/combined integration support
        if return_attention:
            eval_stats, ev_labels, ev_preds, ev_files, ev_attention_maps = evaluate_model_comprehensive(
                model, eval_loader, device, stats_tracker, "eval",
                return_attention=True
            )
        else:
            eval_stats, ev_labels, ev_preds, ev_files = evaluate_model_comprehensive(
                model, eval_loader, device, stats_tracker, "eval",
                return_attention=False
            )

        from sklearn.metrics import classification_report
        report = classification_report(ev_labels, ev_preds, digits=4)
        print(f"\nClassification Report for Epoch {epoch+1}:\n{report}")
        
        # Update scheduler
        metric_for_sched = eval_stats['balanced_acc'] if hyps.early_stop_metric == 'balanced_acc' else eval_stats['macro_f1']
        scheduler.step(metric_for_sched)
        
        # Record epoch statistics
        epoch_data = stats_tracker.record_epoch(epoch, train_stats, eval_stats, model)
        
        # Print epoch summary
        print_epoch_summary(epoch, train_stats, eval_stats, integration_type)
        
        # Save best model
        if eval_stats['acc'] > best_acc:
            best_acc = eval_stats['acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'accuracy': eval_stats['acc'],
                'balanced_acc': eval_stats['balanced_acc'],
                'macro_f1': eval_stats['macro_f1'],
                'integration_type': integration_type,
                'n_chan': n_chan,
                'model_config': model.get_config()
            }, es_path)
            
            print(f"  ✓ Saved best SCNet model at epoch {epoch+1} (acc {eval_stats['acc']:.2f}%)")
        
        # Early stopping
        metric_value = eval_stats['balanced_acc'] if hyps.early_stop_metric == 'balanced_acc' else eval_stats['macro_f1']
        es(metric_value, model, save_best_acc=True)
        if es.early_stop:
            print(f"  ⚠ Early stopping triggered")
            break
    
    # Load best model
    try:
        if os.path.exists(es_path):
            checkpoint = torch.load(es_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("\n" + "=" * 60)
            print(f"✓ Loaded best SCNet model from epoch {checkpoint['epoch']+1}")
            print(f"  Best accuracy: {checkpoint['accuracy']:.2f}%")
            print(f"  Balanced accuracy: {checkpoint.get('balanced_acc', 0):.4f}")
            print("=" * 60)
    except Exception as e:
        print("Could not load best model:", e)
    
    # Final evaluation
    print("\n" + "=" * 80)
    print(f"FINAL SCNet EVALUATION [{integration_type.upper()}]".center(80))
    print("=" * 80)
    
    if return_attention:
        final_stats, final_labels, final_preds, final_files, final_attention_maps = evaluate_model_comprehensive(
            model, eval_loader, device, stats_tracker, "eval_final",
            return_attention=True
        )
    else:
        final_stats, final_labels, final_preds, final_files = evaluate_model_comprehensive(
            model, eval_loader, device, stats_tracker, "eval_final",
            return_attention=False
        )
    
    # Print final results
    print(f"\nFinal SCNet Results [{integration_type.upper()}]:")
    print(f"  Accuracy: {final_stats['acc']:.2f}%")
    print(f"  Balanced Accuracy: {final_stats['balanced_acc']:.4f}")
    print(f"  Macro F1: {final_stats['macro_f1']:.4f}")
    print(f"  Weighted F1: {final_stats['weighted_f1']:.4f}")
    
    # Save all results
    stats_tracker.save_final_results(model=model)
    
    print(f"\n✅ SCNet {integration_type.upper()} integration training complete!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {stats_tracker.run_dir}")
    
    return {
        'model': 'SCNet',
        'integration_type': integration_type,
        'best_accuracy': best_acc,
        'final_stats': final_stats,
        'output_dir': str(stats_tracker.run_dir)
    }

def main(integration_type='input', output_suffix=None, hyp_overrides=None):
    """
    Main function - trains SCNet with gaze integration
    """
    if integration_type not in ['input', 'output', 'both']:
        print(f"Error: integration_type must be 'input', 'output', or 'both'")
        return
    
    return train_scnet_gaze(integration_type, output_suffix, hyp_overrides)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SCNet with gaze integration')
    
    # Model and integration arguments
    parser.add_argument('--type', type=str, default='input',
                       choices=['input', 'output', 'both'],
                       help='Gaze integration type')
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
    
    main(integration_type=args.type, 
         output_suffix=args.suffix,
         hyp_overrides=hyp_overrides if hyp_overrides else None)