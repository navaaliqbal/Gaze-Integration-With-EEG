"""
Comprehensive tracker for all training metrics and statistics
"""
import json
import pickle
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch

class TrainingStatistics:
    """Comprehensive tracker for all training metrics and statistics."""
    
    def __init__(self, output_dir='training_stats'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize storage containers
        self.epoch_stats = []
        self.batch_stats = []
        self.attention_maps = defaultdict(list)
        self.model_weights = []
        self.class_distributions = []
        self.gaze_stats = []
        self.confusion_matrices = []
        self.predictions = defaultdict(list)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
    
    def record_epoch(self, epoch, train_stats, eval_stats, model=None):
        """Record comprehensive epoch statistics."""
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_stats.get('loss', 0),
            'train_cls_loss': train_stats.get('cls_loss', 0),
            'train_gaze_loss': train_stats.get('gaze_loss', 0),
            'train_acc': train_stats.get('acc', 0),
            'eval_acc': eval_stats.get('acc', 0),
            'eval_macro_f1': eval_stats.get('macro_f1', 0),
            'eval_balanced_acc': eval_stats.get('balanced_acc', 0),
            'eval_weighted_f1': eval_stats.get('weighted_f1', 0),
            'eval_precision': eval_stats.get('precision', 0),
            'eval_recall': eval_stats.get('recall', 0),
            'lr': train_stats.get('lr', 0),
            'gaze_batches': train_stats.get('gaze_batches', 0),
            'gaze_samples': train_stats.get('gaze_samples', 0),
            'timestamp': datetime.now().isoformat()
        }
        self.epoch_stats.append(epoch_data)
        
        # Store model weights if requested
        if model is not None:
            self._record_model_weights(model, epoch)
        
        # Save intermediate results
        self._save_intermediate_results()
        
        return epoch_data
    
    def record_batch(self, batch_idx, batch_stats):
        """Record batch-level statistics."""
        batch_data = {
            'batch_idx': batch_idx,
            **batch_stats
        }
        self.batch_stats.append(batch_data)
    
    def record_attention_maps(self, batch_files, attention_maps, labels, predictions, gaze_maps=None):
        """Store attention maps for later analysis."""
        batch_data = []
        for i, file in enumerate(batch_files):
            att_map = attention_maps[i].cpu().numpy() if torch.is_tensor(attention_maps[i]) else attention_maps[i]
            gaze_map = gaze_maps[i].cpu().numpy() if gaze_maps is not None and i < len(gaze_maps) else None
            
            sample_data = {
                'file': file,
                'attention_map': att_map,
                'label': labels[i].item() if torch.is_tensor(labels[i]) else labels[i],
                'prediction': predictions[i].item() if torch.is_tensor(predictions[i]) else predictions[i],
                'gaze_map': gaze_map,
                'timestamp': datetime.now().isoformat()
            }
            batch_data.append(sample_data)
        
        self.attention_maps['samples'].extend(batch_data)
    
    def record_class_distribution(self, dataloader, name="train"):
        """Record class distribution in a dataloader."""
        all_labels = []
        for batch in dataloader:
            labels = batch['label'].numpy()
            all_labels.extend(labels.tolist())
        
        distribution = Counter(all_labels)
        self.class_distributions.append({
            'dataset': name,
            'distribution': dict(distribution),
            'total_samples': len(all_labels),
            'timestamp': datetime.now().isoformat()
        })
        
        return distribution
    
    def record_confusion_matrix(self, y_true, y_pred, name="eval"):
        """Record confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrices.append({
            'dataset': name,
            'matrix': cm.tolist(),
            'labels': np.unique(np.concatenate([y_true, y_pred])).tolist(),
            'timestamp': datetime.now().isoformat()
        })
        return cm
    
    def record_predictions(self, files, true_labels, predictions, probabilities=None, dataset="eval"):
        """Store predictions for detailed analysis."""
        for i, file in enumerate(files):
            # Handle probability conversion
            prob = probabilities[i] if probabilities is not None else None
            if prob is not None:
                if hasattr(prob, 'tolist'):  # numpy array
                    prob = prob.tolist()
            
            pred_data = {
                'file': file,
                'true_label': true_labels[i],
                'predicted_label': predictions[i],
                'probability': prob,
                'dataset': dataset,
                'correct': true_labels[i] == predictions[i],
                'timestamp': datetime.now().isoformat()
            }
            self.predictions[dataset].append(pred_data)
    
    def _record_model_weights(self, model, epoch):
        """Record model weights and gradients for analysis."""
        weight_data = {'epoch': epoch, 'weights': {}, 'gradients': {}}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_data['weights'][name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item()
                }
                if param.grad is not None:
                    weight_data['gradients'][name] = {
                        'mean': param.grad.mean().item(),
                        'std': param.grad.std().item(),
                        'norm': param.grad.norm().item()
                    }
        
        self.model_weights.append(weight_data)
    
    def _save_intermediate_results(self):
        """Save intermediate results to disk."""
        # Save epoch stats
        if self.epoch_stats:
            pd.DataFrame(self.epoch_stats).to_csv(self.run_dir / 'epoch_stats.csv', index=False)
        
        # Save batch stats (sampled to avoid huge files)
        if self.batch_stats and len(self.batch_stats) > 1000:
            # Save only every 10th batch
            sampled_batch_stats = self.batch_stats[::10]
            pd.DataFrame(sampled_batch_stats).to_csv(self.run_dir / 'batch_stats_sampled.csv', index=False)
        
        # Save class distributions
        if self.class_distributions:
            with open(self.run_dir / 'class_distributions.pkl', 'wb') as f:
                pickle.dump(self.class_distributions, f)
    
    def save_final_results(self, model=None, attention_maps=None):
        """Save all collected statistics to disk."""
        print(f"\nSaving training statistics to: {self.run_dir}")
        
        # 1. Save epoch statistics
        epoch_df = pd.DataFrame(self.epoch_stats)
        epoch_df.to_csv(self.run_dir / 'epoch_statistics.csv', index=False)
        
        # 2. Save training history plots
        self._create_training_plots()
        
        # 3. Save attention maps
        if attention_maps:
            with open(self.run_dir / 'attention_maps.pkl', 'wb') as f:
                pickle.dump(attention_maps, f)
            print(f"  - Saved attention maps")
        
        # 4. Save model weights history
        if self.model_weights:
            with open(self.run_dir / 'model_weights_history.pkl', 'wb') as f:
                pickle.dump(self.model_weights, f)
        
        # 5. Save class distributions
        if self.class_distributions:
            class_dist_df = pd.DataFrame(self.class_distributions)
            class_dist_df.to_csv(self.run_dir / 'class_distributions.csv', index=False)
        
        # 6. Save confusion matrices
        if self.confusion_matrices:
            with open(self.run_dir / 'confusion_matrices.pkl', 'wb') as f:
                pickle.dump(self.confusion_matrices, f)
        
        # 7. Save predictions
        for dataset_name, pred_list in self.predictions.items():
            if pred_list:
                pred_df = pd.DataFrame(pred_list)
                pred_df.to_csv(self.run_dir / f'predictions_{dataset_name}.csv', index=False)
        
        # 8. Save gaze statistics
        if self.gaze_stats:
            gaze_df = pd.DataFrame(self.gaze_stats)
            gaze_df.to_csv(self.run_dir / 'gaze_statistics.csv', index=False)
        
        # 9. Save configuration and metadata
        metadata = {
            'timestamp': self.timestamp,
            'total_epochs': len(self.epoch_stats),
            'total_batches': len(self.batch_stats),
            'total_attention_maps': len(self.attention_maps.get('samples', [])),
            'run_directory': str(self.run_dir)
        }
        with open(self.run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 10. Save model architecture and final state
        if model is not None:
            torch.save(model.state_dict(), self.run_dir / 'final_model.pth')
            # Save model summary
            with open(self.run_dir / 'model_summary.txt', 'w') as f:
                f.write(str(model))
                f.write(f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
                f.write(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        print(f"Training statistics saved successfully!")
        print(f"  - CSV files: epoch_statistics.csv, class_distributions.csv")
        print(f"  - Attention maps: attention_maps.pkl")
        print(f"  - Plots: training_*.png")
        print(f"  - Model: final_model.pth")
    
    def _create_training_plots(self):
        """Create visualization plots from training statistics."""
        if not self.epoch_stats:
            return
        
        epochs = [s['epoch'] for s in self.epoch_stats]
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training losses
        ax = axes[0, 0]
        ax.plot(epochs, [s['train_loss'] for s in self.epoch_stats], label='Total Loss', linewidth=2)
        ax.plot(epochs, [s['train_cls_loss'] for s in self.epoch_stats], label='Classification Loss', linestyle='--')
        ax.plot(epochs, [s['train_gaze_loss'] for s in self.epoch_stats], label='Gaze Loss', linestyle=':')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax = axes[0, 1]
        ax.plot(epochs, [s['train_acc'] for s in self.epoch_stats], label='Train Accuracy', linewidth=2)
        ax.plot(epochs, [s['eval_acc'] for s in self.epoch_stats], label='Eval Accuracy', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training vs Evaluation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 scores
        ax = axes[1, 0]
        ax.plot(epochs, [s['eval_macro_f1'] for s in self.epoch_stats], label='Macro F1', linewidth=2)
        ax.plot(epochs, [s['eval_weighted_f1'] for s in self.epoch_stats], label='Weighted F1', linestyle='--')
        ax.plot(epochs, [s['eval_balanced_acc'] for s in self.epoch_stats], label='Balanced Acc', linestyle=':')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Evaluation Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision-Recall
        ax = axes[1, 1]
        ax.plot(epochs, [s['eval_precision'] for s in self.epoch_stats], label='Precision', linewidth=2)
        ax.plot(epochs, [s['eval_recall'] for s in self.epoch_stats], label='Recall', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Confusion matrix from last epoch
        if self.confusion_matrices:
            last_cm = self.confusion_matrices[-1]
            cm = np.array(last_cm['matrix'])
            labels = last_cm['labels']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
            
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=labels, yticklabels=labels,
                   title='Confusion Matrix',
                   ylabel='True label',
                   xlabel='Predicted label')
            
            plt.tight_layout()
            plt.savefig(self.run_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()