"""
Training functions for combined gaze integration
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

def train_epoch_combined(model, train_loader, optimizer, device, gaze_weight=0.2,
                        gaze_loss_type='cosine', class_weights=None, 
                        stats_tracker=None, epoch=0):
    """
    Training epoch for combined model
    """
    model.train()
    total_loss = total_cls = total_gaze = 0.0
    correct = total = 0
    batches_with_gaze = samples_with_gaze = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Epoch {epoch+1} [Combined]")
    
    for batch_idx, batch in pbar:
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
        else:
            gaze = None
        
        # Forward pass with both input and output integration
        if has_gaze:
            outputs = model(eeg, gaze, return_attention=True)
            if isinstance(outputs, dict):
                logits = outputs['logits']
                attention_map = outputs.get('attention_map', None)
            else:
                logits = outputs
                attention_map = None
        else:
            # If no gaze, still get attention map but don't use it for loss
            outputs = model(eeg, None, return_attention=True)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            attention_map = outputs.get('attention_map', None) if isinstance(outputs, dict) else None
        
        # Classification loss
        if class_weights is not None:
            cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            cls_loss = F.cross_entropy(logits, labels)
        
        # Gaze loss for output integration (if gaze available)
        if has_gaze and attention_map is not None:
            from .losses import compute_gaze_attention_loss
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
        
        # Record batch statistics
        if stats_tracker:
            batch_stats = {
                'epoch': epoch,
                'batch_loss': loss.item(),
                'batch_cls_loss': cls_loss.item(),
                'batch_gaze_loss': gaze_loss.item() if has_gaze else 0.0,
                'batch_accuracy': (preds == labels).float().mean().item(),
                'has_gaze': has_gaze,
                'lr': current_lr,
                'gaze_alpha': model.gaze_alpha.item() if hasattr(model, 'gaze_alpha') else 0.0
            }
            stats_tracker.record_batch(batch_idx, batch_stats)
        
        # Update progress bar
        postfix = {
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct/total*100:.1f}%",
            'gaze': 'Y' if has_gaze else 'N'
        }
        if hasattr(model, 'gaze_alpha'):
            postfix['alpha'] = f"{model.gaze_alpha.item():.3f}"
        pbar.set_postfix(postfix)
    
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
    
    if hasattr(model, 'gaze_alpha'):
        train_stats['gaze_alpha'] = model.gaze_alpha.item()
    
    return train_stats