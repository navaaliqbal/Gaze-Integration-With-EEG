"""
Training function for Conformer with input gaze integration
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os


def focal_loss(logits, labels, alpha=None, gamma=2.0):
    """
    Focal Loss for handling class imbalance
    Focuses on hard-to-classify examples
    
    Args:
        logits: [B, C] model predictions
        labels: [B] ground truth labels
        alpha: [C] class weights (optional)
        gamma: focusing parameter (default: 2.0)
    """
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)  # probability of true class
    focal_term = (1 - pt) ** gamma
    
    loss = focal_term * ce_loss
    
    if alpha is not None:
        alpha_t = alpha[labels]
        loss = alpha_t * loss
    
    return loss.mean()


def train_epoch_conformer_input(model, train_loader, optimizer, device, 
                                class_weights=None, stats_tracker=None, epoch=0,
                                gaze_weight=None, gaze_loss_type=None, gaze_loss_scale=None, 
                                gaze_optimizer=None, **kwargs):
    """
    Training epoch for Conformer input integration model
    
    Note: gaze_weight, gaze_loss_type, and gaze_loss_scale are accepted but ignored
    because input integration doesn't use gaze loss (gaze is integrated at input level).
    
    Args:
        model: Conformer_Gaze_Input model
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
    batches_with_gaze = samples_with_gaze = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    current_lr = optimizer.param_groups[0]['lr']
    gaze_lr = gaze_optimizer.param_groups[0]['lr'] if gaze_optimizer else current_lr
    
    # Track gaze alpha gradients
    gaze_alpha_grads = []
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Epoch {epoch+1} [Conformer Input]")
    
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
        
        # Check for gaze data (REQUIRED for input integration)
        has_gaze = 'gaze' in batch and batch['gaze'] is not None
        if has_gaze:
            gaze = batch['gaze'].to(device)
            batches_with_gaze += 1
            samples_with_gaze += eeg.shape[0]
        else:
            # For input integration, we need gaze data
            # Use baseline attention if not available (small constant value)
            gaze = torch.ones_like(eeg).to(device) * 0.01
        
        # Forward pass (Conformer returns (features, logits) tuple)
        outputs = model(eeg, gaze)
        if isinstance(outputs, tuple):
            features, logits = outputs
        else:
            logits = outputs
        
        # Classification loss with focal loss for better class 1 learning
        # Focal loss helps model focus on minority class (class 1)
        if class_weights is not None:
            # Use focal loss with class weights - better for imbalanced data
            loss = focal_loss(logits, labels, alpha=class_weights, gamma=2.0)
        else:
            # Standard cross-entropy with label smoothing
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        
        # Backward pass
        optimizer.zero_grad()
        if gaze_optimizer:
            gaze_optimizer.zero_grad()
        
        loss.backward()
        
        # Track gaze alpha gradient before clipping
        if hasattr(model, 'gaze_alpha') and model.gaze_alpha.grad is not None:
            gaze_alpha_grads.append(model.gaze_alpha.grad.item())
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update both optimizers
        optimizer.step()
        if gaze_optimizer:
            gaze_optimizer.step()
        
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
        'cls_loss': avg_loss,  # For input models, cls_loss = loss (no separate gaze loss)
        'acc': acc,
        'gaze_batches': batches_with_gaze,
        'gaze_samples': samples_with_gaze,
        'total_batches': len(train_loader),
        'total_samples': total,
        'lr': current_lr,
        'gaze_lr': gaze_lr
    }
    
    if hasattr(model, 'gaze_alpha'):
        train_stats['gaze_alpha'] = model.gaze_alpha.item()
        # Add average gradient magnitude
        if gaze_alpha_grads:
            train_stats['gaze_alpha_grad'] = sum(abs(g) for g in gaze_alpha_grads) / len(gaze_alpha_grads)
        else:
            train_stats['gaze_alpha_grad'] = None
    
    return train_stats
