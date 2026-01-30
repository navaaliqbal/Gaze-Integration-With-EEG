"""
Training functions for input gaze integration
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

def train_epoch_input(model, train_loader, optimizer, device, 
                     class_weights=None, stats_tracker=None, epoch=0):
    """
    Training epoch for input integration model
    """
    model.train()
    total_loss = 0.0
    correct = total = 0
    batches_with_gaze = samples_with_gaze = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Epoch {epoch+1} [Input]")
    
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
        
        # Check for gaze data (REQUIRED for input integration)
        has_gaze = 'gaze' in batch and batch['gaze'] is not None
        if has_gaze:
            gaze = batch['gaze'].to(device)
            batches_with_gaze += 1
            samples_with_gaze += eeg.shape[0]
        else:
            # For input integration, we need gaze data
            # Use baseline attention if not available
            gaze = torch.ones_like(eeg).to(device) * 0.01
        
        # Forward pass (input integration)
        logits = model(eeg, gaze)
        
        # Classification loss only (no gaze loss for input integration)
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
        
        # Record batch statistics
        if stats_tracker:
            batch_stats = {
                'epoch': epoch,
                'batch_loss': loss.item(),
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
    acc = correct / total * 100 if total > 0 else 0.0
    
    train_stats = {
        'loss': avg_loss,
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