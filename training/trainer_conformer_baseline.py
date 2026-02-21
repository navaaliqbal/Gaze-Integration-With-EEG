"""
Training function for baseline Conformer model (EEG-only, no gaze integration)
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os


def train_epoch_conformer_baseline(model, train_loader, optimizer, device, 
                                   class_weights=None, stats_tracker=None, epoch=0):
    """
    Training epoch for baseline Conformer model (EEG-only)
    
    Args:
        model: Baseline Conformer model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device (cuda/cpu)
        class_weights: Optional class weights for imbalanced data
        stats_tracker: Statistics tracker
        epoch: Current epoch number
        
    Returns:
        Training statistics dictionary
    """
    model.train()
    total_loss = 0.0
    correct = total = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in pbar:
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)
        
        # Get batch files for tracking
        batch_files = []
        if 'file' in batch:
            for f in batch['file']:
                if isinstance(f, (bytes, bytearray)):
                    try:
                        f = f.decode('utf-8', errors='ignore')
                    except:
                        f = str(f)
                batch_files.append(os.path.basename(str(f)))
        else:
            batch_files = ["unknown"] * eeg.shape[0]
        
        # Add channel dimension: (batch, channels, time) -> (batch, 1, channels, time)
        if eeg.dim() == 3:
            eeg = eeg.unsqueeze(1)
        
        # Forward pass (Conformer returns (features, logits) tuple)
        outputs = model(eeg)
        if isinstance(outputs, tuple):
            features, logits = outputs
        else:
            logits = outputs
        
        # Classification loss
        if class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct/total*100:.1f}%"
        })
    
    # Calculate epoch averages
    avg_loss = total_loss / max(len(train_loader), 1)
    acc = correct / total * 100 if total > 0 else 0.0
    
    train_stats = {
        'loss': avg_loss,
        'acc': acc,
        'total_samples': total,
        'total_batches': len(train_loader),
        'lr': current_lr
    }
    
    return train_stats
