"""
Training functions for output gaze integration
This is your existing trainer - just renamed
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from training.losses import compute_gaze_attention_loss


def train_epoch_output(model, train_loader, optimizer, device, gaze_weight=1, 
                         gaze_loss_type='cosine', class_weights=None, stats_tracker=None, epoch=0,gaze_loss_scale=1.0):
    """Enhanced training epoch with comprehensive statistics tracking."""
    model.train()
    total_loss = total_cls = total_gaze = 0.0
    correct = total = 0
    batches_with_gaze = samples_with_gaze = 0

    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    print("Class weights used in loss computation.", class_weights)

    
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
        
        # Check for gaze data
        has_gaze = 'gaze' in batch and batch['gaze'] is not None
        if has_gaze:
            gaze = batch['gaze'].to(device)
            batches_with_gaze += 1
            samples_with_gaze += eeg.shape[0]
        
        # Forward pass
        if has_gaze:
            outputs = model(eeg, return_attention=True)
            if isinstance(outputs, tuple):
                logits, attention_map = outputs
            else:
                logits = outputs['logits']
                attention_map = outputs['attention_map']
        else:
            logits = model(eeg, return_attention=False)
            attention_map = None
        
        # Classification loss
        if class_weights is not None:
            cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            cls_loss = F.cross_entropy(logits, labels)
        
        # Gaze loss if available
        if has_gaze and attention_map is not None:
            gaze_loss_raw = compute_gaze_attention_loss(attention_map, gaze, labels, gaze_loss_type)
            gaze_loss_scaled = gaze_loss_scale * gaze_loss_raw
            loss = (1 - gaze_weight) * cls_loss + gaze_weight * gaze_loss_scaled
        else:
            gaze_loss_raw = torch.tensor(0.0).to(device)
            gaze_loss_scaled = torch.tensor(0.0).to(device)
            loss = cls_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_gaze += gaze_loss_scaled.item() if has_gaze else 0.0

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # No batch-level storage during training (only epoch-level metrics)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct/total*100:.1f}%",
            'gaze': 'Y' if has_gaze else 'N'
        })
    
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
        'lr': current_lr,
        'gaze_loss_scale': gaze_loss_scale
    }
    
    return train_stats
