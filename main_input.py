"""
Main script for multi-task NeuroGATE training
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import sys

# Import your modules
from models.neurogate_gaze_features_mt import NeuroGATE_MultiTask
from data.dataset import EEGGazeFeatureDataset
from training.trainer_input import train_epoch_multitask, evaluate_multitask
from sklearn.metrics import classification_report, confusion_matrix

def train_multitask_neurogate(config):
    """
    Train multi-task NeuroGATE model
    """
    print("=" * 80)
    print("TRAINING MULTI-TASK NEUROGATE")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Set device
    device = torch.device(f"cuda:{config.get('gpu', 0)}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ============================================
    # 1. LOAD DATA
    # ============================================
    print("\n1. Loading data...")
    
    dataset_kwargs = {
        'target_length': config.get('target_length', 15000),
        'eeg_sampling_rate': config.get('eeg_sampling_rate', 50.0),
        'gaze_feature_dim': config.get('gaze_feature_dim', 5),
        'debug': config.get('debug', False),
    }
    
    train_dataset = EEGGazeFeatureDataset(
        data_dir=config['train_data_dir'],
        gaze_json_dir=config['gaze_json_dir'],
        **dataset_kwargs
    )
    
    val_dataset = EEGGazeFeatureDataset(
        data_dir=config['val_data_dir'],
        gaze_json_dir=config['gaze_json_dir'],
        **dataset_kwargs
    )
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    # ============================================
    # 2. CREATE MULTI-TASK MODEL
    # ============================================
    print("\n2. Creating multi-task model...")
    
    model = NeuroGATE_MultiTask(
        n_chan=config.get('n_chan', 22),
        n_outputs=config.get('n_outputs', 2),
        original_time_length=config.get('target_length', 15000),
        gaze_feature_dim=config.get('gaze_feature_dim', 5),
        fusion_method=config.get('fusion_method', 'early'),
        task_weight=config.get('task_weight', 0.5)  # λ in loss function
    )
    
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Task weight (λ): {model.task_weight}")
    
    # ============================================
    # 3. SET UP OPTIMIZER
    # ============================================
    print("\n3. Setting up optimizer...")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config.get('lr_patience', 5),
        verbose=True
    )
    
    # ============================================
    # 4. TRAINING LOOP
    # ============================================
    print("\n4. Starting multi-task training...")
    
    class StatsTracker:
        def __init__(self):
            self.stats = []
        def record_batch(self, batch_idx, batch_stats):
            self.stats.append(batch_stats)
        def save(self, path):
            import pandas as pd
            df = pd.DataFrame(self.stats)
            df.to_csv(path, index=False)
    
    stats_tracker = StatsTracker()
    best_val_acc = 0.0
    best_model_path = None
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Training
        train_stats = train_epoch_multitask(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            stats_tracker=stats_tracker,
            epoch=epoch
        )
        
        # Validation
        val_stats = evaluate_multitask(model, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_stats['total_loss'])
        
        # Save best model
        if val_stats['acc'] > best_val_acc:
            best_val_acc = val_stats['acc']
            best_model_path = os.path.join(
                config['output_dir'],
                f"best_model_epoch{epoch+1}_acc{val_stats['acc']:.2f}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_stats['acc'],
                'val_class_loss': val_stats['class_loss'],
                'val_gaze_loss': val_stats['gaze_loss'],
                'task_weight': model.task_weight
            }, best_model_path)
            print(f"   ✓ Saved best model to: {best_model_path}")
        
        # Print epoch summary
        print(f"\n   Epoch Summary:")
        print(f"     Total Loss: {train_stats['total_loss']:.4f} "
              f"(Class: {train_stats['class_loss']:.4f}, "
              f"Gaze: {train_stats['gaze_loss']:.4f})")
        print(f"     Train Acc: {train_stats['acc']:.2f}%")
        print(f"     Val Loss: {val_stats['total_loss']:.4f} "
              f"(Class: {val_stats['class_loss']:.4f}, "
              f"Gaze: {val_stats['gaze_loss']:.4f})")
        print(f"     Val Acc: {val_stats['acc']:.2f}%")
        print(f"     Best Val Acc: {best_val_acc:.2f}%")
        
        # Print gaze reconstruction metrics
        if val_stats['gaze_metrics']:
            print(f"\n   Gaze Reconstruction Metrics:")
            print(f"     Total MSE: {val_stats['gaze_metrics']['mse_total']:.4f}")
            print(f"     MAE: {val_stats['gaze_metrics']['mae']:.4f}")
            print(f"     Correlation: {val_stats['gaze_metrics']['correlation']:.4f}")
    
    # ============================================
    # 5. FINAL EVALUATION
    # ============================================
    print("\n5. Final evaluation...")
    
    # Load best model
    if best_model_path:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Generate comprehensive report
    model.eval()
    all_preds = []
    all_labels = []
    all_gaze_pred = []
    all_gaze_true = []
    
    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)
            gaze_true = batch['gaze'].to(device) if 'gaze' in batch else None
            
            outputs = model(eeg, gaze_true)
            preds = torch.argmax(outputs['classification'], dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if gaze_true is not None:
                all_gaze_pred.extend(outputs['gaze_predictions'].cpu().numpy())
                all_gaze_true.extend(gaze_true.cpu().numpy())
    
    # Classification report
    print("\n" + "="*80)
    print("FINAL CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(all_labels, all_preds,
                                target_names=[f'Class {i}' for i in range(config['n_outputs'])]))
    
    # Gaze reconstruction report
    if all_gaze_pred:
        print("\n" + "="*80)
        print("GAZE FEATURE RECONSTRUCTION REPORT")
        print("="*80)
        
        all_gaze_pred = np.array(all_gaze_pred)
        all_gaze_true = np.array(all_gaze_true)
        
        gaze_feature_names = [
            "Fixation Count",
            "Mean Duration", 
            "Total Duration",
            "Saccade Velocity",
            "Gaze Dispersion"
        ]
        
        for i in range(all_gaze_pred.shape[1]):
            mse = np.mean((all_gaze_pred[:, i] - all_gaze_true[:, i]) ** 2)
            mae = np.mean(np.abs(all_gaze_pred[:, i] - all_gaze_true[:, i]))
            corr = np.corrcoef(all_gaze_pred[:, i], all_gaze_true[:, i])[0, 1]
            
            print(f"  {gaze_feature_names[i]}:")
            print(f"    MSE: {mse:.4f}, MAE: {mae:.4f}, Correlation: {corr:.4f}")
        
        print(f"\n  Overall Reconstruction:")
        print(f"    Total MSE: {np.mean((all_gaze_pred - all_gaze_true) ** 2):.4f}")
        print(f"    Total MAE: {np.mean(np.abs(all_gaze_pred - all_gaze_true)):.4f}")
    
    # Save results
    results = {
        'classification': {
            'accuracy': np.mean(np.array(all_preds) == np.array(all_labels)),
            'predictions': all_preds,
            'labels': all_labels
        },
        'gaze_reconstruction': {
            'predictions': all_gaze_pred.tolist() if all_gaze_pred else [],
            'true_values': all_gaze_true.tolist() if all_gaze_true else []
        }
    }
    
    results_path = os.path.join(config['output_dir'], 'multitask_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Multi-task results saved to: {results_path}")
    return model


def main():
    """Main execution"""
    config = {
        # Data paths
        'train_data_dir': "/kaggle/input/results1/results_1/data1/data_processed1/results0/train1",
        'val_data_dir': "/kaggle/input/results1/results_1/data1/data_processed1/results0/eval1",
        'gaze_json_dir': "/kaggle/input/results1/results_1/gaze_data",
        
        # Model
        'n_chan': 22,
        'n_outputs': 2,
        'target_length': 15000,
        'gaze_feature_dim': 5,
        'fusion_method': 'early',
        'task_weight': 0.4,  # λ: weight for gaze reconstruction loss
        
        # Training
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 1e-5,
        'weight_decay': 1e-4,
        'lr_patience': 5,
        
        # Misc
        'seed': 42,
        'gpu': 0,
        'debug': False,
        
        # Output
        'output_dir': r"C:\Users\S.S.T\Documents\VsCode\eeg models\results\multitask",
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Train
    model = train_multitask_neurogate(config)
    print(f"\nTraining completed! Results saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()