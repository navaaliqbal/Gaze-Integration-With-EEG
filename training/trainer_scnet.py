"""
Training functions for SCNet models - WITHOUT tqdm progress bar
"""
import torch
import torch.nn.functional as F
import os

def train_epoch_scnet(model, train_loader, optimizer, device, gaze_weight=0.2,
                     gaze_loss_type='cosine', class_weights=None, 
                     stats_tracker=None, epoch=0):
    """
    Training epoch for SCNet with output integration
    """
    model.train()
    total_loss = total_cls = total_gaze = 0.0
    correct = total = 0
    batches_with_gaze = samples_with_gaze = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print epoch start
    print(f"\nTraining Epoch {epoch+1}...")
    
    # Iterate without tqdm
    for batch_idx, batch in enumerate(train_loader):
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
        
        # Check for gaze data
        has_gaze = 'gaze' in batch and batch['gaze'] is not None
        if has_gaze:
            gaze = batch['gaze'].to(device)
            batches_with_gaze += 1
            samples_with_gaze += eeg.shape[0]
        
        # Forward pass with attention
        if has_gaze:
            outputs = model(eeg, return_attention=True)
            if isinstance(outputs, dict):
                logits = outputs['logits']
                attention_map = outputs.get('attention_map', None)
            elif isinstance(outputs, tuple):
                logits, attention_map = outputs
            else:
                logits = outputs
                attention_map = None
        else:
            logits = model(eeg, return_attention=False)
            attention_map = None
        
        # Classification loss
        if class_weights is not None:
            cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            cls_loss = F.cross_entropy(logits, labels)
        
        # Gaze loss if available (output integration)
        if has_gaze and attention_map is not None:
            from training.losses import compute_gaze_attention_loss
            gaze_loss = compute_gaze_attention_loss(attention_map, gaze, labels, gaze_loss_type)
            loss = cls_loss + gaze_weight * gaze_loss
        else:
            gaze_loss = torch.tensor(0.0).to(device)
            loss = cls_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_gaze += gaze_loss.item() if has_gaze else 0.0
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Print batch progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            batch_acc = correct / total * 100 if total > 0 else 0.0
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                  f"Loss={loss.item():.4f}, Acc={batch_acc:.1f}%")
        
        # Record batch statistics
        if stats_tracker:
            batch_stats = {
                'epoch': epoch,
                'batch_loss': loss.item(),
                'batch_cls_loss': cls_loss.item(),
                'batch_gaze_loss': gaze_loss.item() if has_gaze else 0.0,
                'batch_accuracy': (preds == labels).float().mean().item(),
                'has_gaze': has_gaze,
                'lr': current_lr
            }
            stats_tracker.record_batch(batch_idx, batch_stats)
    
    # Calculate epoch averages
    avg_loss = total_loss / max(len(train_loader), 1)
    avg_cls = total_cls / max(len(train_loader), 1)
    avg_gaze = total_gaze / max(len(train_loader), 1)
    acc = correct / total * 100 if total > 0 else 0.0
    
    train_stats = {
        'loss': avg_loss,
        'cls_loss': avg_cls,
        'gaze_loss': avg_gaze,
        'acc': acc,
        'gaze_batches': batches_with_gaze,
        'gaze_samples': samples_with_gaze,
        'total_batches': len(train_loader),
        'total_samples': total,
        'lr': current_lr
    }
    
    return train_stats