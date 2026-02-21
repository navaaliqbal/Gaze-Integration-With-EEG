"""
Training functions for combined gaze integration
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from training.losses import compute_gaze_attention_loss


def train_epoch_combined(model, train_loader, optimizer, device, gaze_weight=0.1,
                        gaze_loss_type='cosine', gaze_loss_scale=1.0, class_weights=None, 
                        stats_tracker=None, epoch=0):
    """Training epoch for combined model with gaze loss scaling."""
    model.train()
    total_loss = total_cls = total_gaze = 0.0
    correct = total = 0
    batches_with_gaze = samples_with_gaze = 0
    
    # Track gaze_alpha values
    gaze_alpha_values = []
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Epoch {epoch+1} [Combined]")
    
    for batch_idx, batch in pbar:
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)
        
        # Check for gaze data
        has_gaze = 'gaze' in batch and batch['gaze'] is not None
        if has_gaze:
            gaze = batch['gaze'].to(device)
            batches_with_gaze += 1
            samples_with_gaze += eeg.shape[0]
        else:
            gaze = None
        
        # Forward pass with attention
        if has_gaze:
            outputs = model(eeg, gaze, return_attention=True)
        else:
            outputs = model(eeg, return_attention=True)
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
            attention_map = outputs.get('attention_map', None)
        else:
            logits = outputs
            attention_map = None
        
        # Get gaze_alpha if available
        if hasattr(model, 'gaze_alpha'):
            gaze_alpha_values.append(model.gaze_alpha.item())
        
        # Classification loss
        if class_weights is not None:
            cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            cls_loss = F.cross_entropy(logits, labels)
        
        # Gaze loss if available
        if has_gaze and attention_map is not None:
            gaze_loss_raw = compute_gaze_attention_loss(attention_map, gaze, labels, gaze_loss_type)
            gaze_loss_scaled = gaze_loss_raw * gaze_loss_scale
            loss = cls_loss + gaze_weight * gaze_loss_scaled
        else:
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
        total_gaze += gaze_loss_scaled.item() if has_gaze and attention_map is not None else 0.0
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        postfix = {
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct/total*100:.1f}%",
            'gaze': 'Y' if has_gaze else 'N'
        }
        if has_gaze and attention_map is not None:
            postfix['gaze_loss'] = f"{gaze_loss_scaled.item():.4f}"
        if gaze_alpha_values:
            postfix['alpha'] = f"{gaze_alpha_values[-1]:.3f}"
        pbar.set_postfix(postfix)
    
    # Calculate epoch averages
    avg_loss = total_loss / max(len(train_loader), 1)
    avg_cls = total_cls / max(len(train_loader), 1)
    avg_gaze = total_gaze / max(len(train_loader), 1)
    avg_alpha = sum(gaze_alpha_values) / len(gaze_alpha_values) if gaze_alpha_values else 0.0
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
    
    if hasattr(model, 'gaze_alpha'):
        train_stats['gaze_alpha'] = avg_alpha
    
    return train_stats