"""
Main script to run all SCNet variants with different gaze integration levels
"""
import os
import sys
import torch
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.paths import PathConfig
from config.hyperparameters import get_hyp_for_integration
from data.dataloader_builder import get_dataloaders_fixed
from models.model_factory import create_scnet_model, get_model_config
from training.trainer import train_epoch
from training.metrics import evaluate_model_comprehensive
from training.early_stopping import EarlyStopping
from utils.debugger import DataDebugger
from utils.statistics_tracker import TrainingStatistics
from sklearn.metrics import classification_report


def train_scnet(integration_type='none', output_suffix=None):
    """
    Train SCNet with specified gaze integration
    """
    # Setup
    DataDebugger.print_header(f"SCNET GAZE INTEGRATION: {integration_type.upper()}", width=80)
    PathConfig.setup_directories()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Hyperparameters
    hyps = get_hyp_for_integration('output' if integration_type == 'none' else integration_type)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"scnet_results_{integration_type}_{timestamp}"
    if output_suffix:
        output_dir = f"{output_dir}_{output_suffix}"
    
    stats_tracker = TrainingStatistics(output_dir=output_dir)
    print(f"Results saved to: {stats_tracker.run_dir}")
    
    # Load data
    print("\nLoading data...")
    train_loader, eval_loader, _ = get_dataloaders_fixed(
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
    
    # Record class distributions
    train_dist = stats_tracker.record_class_distribution(train_loader, "train")
    eval_dist = stats_tracker.record_class_distribution(eval_loader, "eval")
    print(f"\nTrain distribution: {dict(train_dist)}")
    print(f"Eval distribution: {dict(eval_dist)}")
    
    # Get input dimensions
    sample_batch = next(iter(train_loader))
    n_chan = sample_batch['eeg'].shape[1]
    print(f"\nUsing {n_chan} channels from data")
    
    # Create model
    print(f"\nCreating SCNet with {integration_type} gaze integration...")
    model = create_scnet_model(
        integration_type=integration_type,
        n_chan=n_chan,
        n_outputs=hyps.n_outputs,
        original_time_length=hyps.original_time_length
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model config: {get_model_config(model)}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.1
    )
    
    # Early stopping
    es_path = f"best_scnet_{integration_type}.pth"
    es = EarlyStopping(patience=hyps.patience, path=es_path, verbose=True)
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_batch = next(iter(train_loader))
    test_eeg = test_batch['eeg'].to(device)[:2]
    test_gaze = test_batch['gaze'].to(device)[:2] if 'gaze' in test_batch else None
    
    if integration_type == 'none':
        logits = model(test_eeg)
    elif integration_type == 'input':
        logits = model(test_eeg, test_gaze)
    elif integration_type == 'output':
        outputs = model(test_eeg, return_attention=True)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
    else:  # both
        outputs = model(test_eeg, test_gaze, return_attention=True)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
    
    print(f"✓ Forward pass OK, logits shape: {logits.shape}")
    
    # Class weights
    class_counts = list(train_dist.values())
    total = sum(class_counts)
    class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float32).to(device)
    
    # Training loop
    print(f"\nStarting training for {hyps.epochs} epochs...")
    print("=" * 80)
    
    best_metric = 0.0
    metric_name = 'balanced_acc'
    
    for epoch in range(hyps.epochs):
        print(f"\nEpoch {epoch+1}/{hyps.epochs} [SCNet-{integration_type.upper()}]")
        print("-" * 60)
        
        # Train
        train_stats = train_epoch(
            model, train_loader, optimizer, device,
            gaze_weight=hyps.gaze_weight if integration_type in ['output', 'both'] else 0,
            gaze_loss_type=hyps.gaze_loss_type,
            class_weights=class_weights,
            stats_tracker=stats_tracker,
            epoch=epoch
        )
        
        # Evaluate
        return_attention = integration_type in ['output', 'both']
        if return_attention:
            eval_stats, ev_labels, ev_preds, ev_files, ev_att = evaluate_model_comprehensive(
                model, eval_loader, device, stats_tracker, "eval",
                return_attention=True
            )
        else:
            eval_stats, ev_labels, ev_preds, ev_files = evaluate_model_comprehensive(
                model, eval_loader, device, stats_tracker, "eval",
                return_attention=False
            )
        
        # Update scheduler
        scheduler.step(eval_stats['balanced_acc'])
        
        # Record epoch
        stats_tracker.record_epoch(epoch, train_stats, eval_stats, model)
        
        # Print summary
        print(f"\nTrain Loss: {train_stats['loss']:.4f} | Train Acc: {train_stats['acc']:.2f}%")
        if 'gaze_loss' in train_stats and train_stats['gaze_loss'] > 0:
            print(f"Gaze Loss: {train_stats['gaze_loss']:.4f}")
        print(f"Eval Loss: {eval_stats['loss']:.4f} | Eval Acc: {eval_stats['acc']:.2f}%")
        print(f"Balanced Acc: {eval_stats['balanced_acc']:.4f} | Macro F1: {eval_stats['macro_f1']:.4f}")
        
        # Classification report
        report = classification_report(ev_labels, ev_preds, digits=4)
        print(f"\nClassification Report:\n{report}")
        
        # Save best model
        if eval_stats['balanced_acc'] > best_metric:
            best_metric = eval_stats['balanced_acc']
            torch.save(model.state_dict(), es_path)
            print(f"✓ New best model saved! (Balanced Acc: {best_metric:.4f})")
        
        # Early stopping
        es(eval_stats['balanced_acc'], model)
        if es.early_stop:
            print("Early stopping triggered")
            break
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    if os.path.exists(es_path):
        model.load_state_dict(torch.load(es_path))
        print(f"Loaded best model from {es_path}")
    
    if return_attention:
        final_stats, final_labels, final_preds, final_files, final_att = evaluate_model_comprehensive(
            model, eval_loader, device, stats_tracker, "eval_final",
            return_attention=True
        )
    else:
        final_stats, final_labels, final_preds, final_files = evaluate_model_comprehensive(
            model, eval_loader, device, stats_tracker, "eval_final",
            return_attention=False
        )
    
    print(f"\nFinal Results [SCNet-{integration_type.upper()}]:")
    print(f"  Accuracy: {final_stats['acc']:.2f}%")
    print(f"  Balanced Accuracy: {final_stats['balanced_acc']:.4f}")
    print(f"  Macro F1: {final_stats['macro_f1']:.4f}")
    print(f"  Weighted F1: {final_stats['weighted_f1']:.4f}")
    
    # Save results
    stats_tracker.save_final_results(model=model)
    
    print(f"\nTraining complete! Best balanced accuracy: {best_metric:.4f}")
    print(f"Results saved to: {stats_tracker.run_dir}")
    
    return {
        'integration_type': integration_type,
        'best_metric': best_metric,
        'final_stats': final_stats,
        'output_dir': str(stats_tracker.run_dir)
    }


def main():
    parser = argparse.ArgumentParser(description='Train SCNet with gaze integration')
    parser.add_argument('--type', type=str, default='none',
                        choices=['none', 'input', 'output', 'both'],
                        help='Gaze integration type')
    parser.add_argument('--suffix', type=str, default=None,
                        help='Suffix for output directory')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print(f"SCNet TRAINING WITH GAZE INTEGRATION: {args.type.upper()}")
    print("=" * 80)
    
    return train_scnet(args.type, args.suffix)


if __name__ == "__main__":
    main()