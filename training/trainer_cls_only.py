"""
Training functions for classification-only models (no gaze integration)
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_epoch_cls_only(model, train_loader, optimizer, device, 
                         class_weights=None, stats_tracker=None, epoch=0,
                         **kwargs):
    """
    Training epoch for classification-only models (no gaze integration)
    """
    model.train()
    total_loss = 0.0
    correct = total = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Epoch {epoch+1} [CLS-Only]")
    
    for batch_idx, batch in pbar:
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        logits = model(eeg)
        
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
        'cls_loss': avg_loss,  # Same as total loss for cls-only
        'gaze_loss': 0.0,
        'acc': acc,
        'gaze_batches': 0,
        'gaze_samples': 0,
        'total_batches': len(train_loader),
        'total_samples': total,
        'lr': current_lr
    }
    
    return train_stats