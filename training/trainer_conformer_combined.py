"""
Training functions for Conformer with COMBINED gaze integration (both input and output)

This trainer handles BOTH:
1. INPUT: Gaze modulation of EEG signals (gaze-as-gate)
2. OUTPUT: Attention map generation and alignment loss

Similar to trainer_combined.py but adapted for Conformer architecture.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from training.losses import compute_gaze_attention_loss


def focal_loss(logits, labels, alpha=None, gamma=2.0, label_smoothing=0.1):
    """
    Focal loss to address class imbalance by focusing on hard examples.
    
    Args:
        logits: [B, n_classes] model predictions
        labels: [B] ground truth labels
        alpha: [n_classes] class weights tensor or None
        gamma: Focusing parameter (higher = more focus on hard examples)
        label_smoothing: Label smoothing factor
    """
    # Convert to probabilities
    probs = F.softmax(logits, dim=1)
    
    # Get probability of true class
    labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()
    
    # Apply label smoothing
    if label_smoothing > 0:
        labels_one_hot = labels_one_hot * (1 - label_smoothing) + label_smoothing / logits.shape[1]
    
    # Compute cross entropy
    ce_loss = -torch.log(probs + 1e-8) * labels_one_hot
    
    # Apply focal weight: (1 - p_t)^gamma
    pt = (probs * labels_one_hot).sum(dim=1)
    focal_weight = (1 - pt) ** gamma
    
    # Apply class weights if provided
    if alpha is not None:
        alpha_t = alpha[labels]
        focal_weight = focal_weight * alpha_t
    
    # Final loss
    loss = (focal_weight.unsqueeze(1) * ce_loss).sum(dim=1).mean()
    
    return loss


def train_epoch_conformer_combined(model, train_loader, optimizer, device, 
                                   gaze_weight=0.5, gaze_loss_type='cosine', 
                                   gaze_loss_scale=1.0, class_weights=None, 
                                   stats_tracker=None, epoch=0, 
                                   gaze_optimizer=None, use_focal_loss=True, 
                                   focal_gamma=2.0, label_smoothing=0.1):
    """
    Training epoch for Conformer combined model (both input and output gaze integration).
    
    Args:
        model: Conformer_Combined model
        train_loader: Training data loader
        optimizer: Main optimizer
        device: Device (cuda/cpu)
        gaze_weight: Weight for gaze loss in [0, 1] (0.5 = equal weight)
        gaze_loss_type: Type of gaze loss ('cosine', 'mse', 'combined')
        gaze_loss_scale: Additional scaling for gaze loss
        class_weights: Optional class weights for imbalanced data
        stats_tracker: Statistics tracker
        epoch: Current epoch number
        gaze_optimizer: Optional separate optimizer for gaze_alpha
        use_focal_loss: Whether to use focal loss instead of CE
        focal_gamma: Gamma parameter for focal loss
        label_smoothing: Label smoothing factor
        
    Returns:
        Training statistics dictionary
    """
    model.train()
    total_loss = total_cls = total_gaze = 0.0
    correct = total = 0
    batches_with_gaze = samples_with_gaze = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    current_lr = optimizer.param_groups[0]['lr']
    gaze_lr = gaze_optimizer.param_groups[0]['lr'] if gaze_optimizer else current_lr
    
    # Track gaze alpha gradients
    gaze_alpha_grads = []
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Epoch {epoch+1} [Conformer Combined]")
    
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
        else:
            gaze = None
        
        # Forward pass with both input and output integration
        # - INPUT: gaze modulates EEG before feature extraction
        # - OUTPUT: attention map is generated and compared with gaze
        if has_gaze:
            outputs = model(eeg, gaze, return_attention=True)
            # Handle dict output
            if isinstance(outputs, dict):
                logits = outputs['logits']
                attention_map = outputs['attention_map']
            else:
                # Fallback for tuple
                if len(outputs) == 3:
                    _, logits, attention_map = outputs
                else:
                    features, logits = outputs
                    attention_map = None
        else:
            # No gaze: still get attention map but don't use for loss
            outputs = model(eeg, None, return_attention=True)
            if isinstance(outputs, dict):
                logits = outputs['logits']
                attention_map = outputs['attention_map']
            else:
                if len(outputs) == 2:
                    features, logits = outputs
                    attention_map = None
                else:
                    _, logits, attention_map = outputs
        
        # Classification loss - use focal loss or standard CE
        if use_focal_loss:
            cls_loss = focal_loss(
                logits, labels, 
                alpha=class_weights, 
                gamma=focal_gamma,
                label_smoothing=label_smoothing
            )
        else:
            if class_weights is not None:
                cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
            else:
                cls_loss = F.cross_entropy(logits, labels)
        
        # Gaze loss for OUTPUT integration (if gaze available) WITH SCALING
        if has_gaze and attention_map is not None:
            gaze_loss_raw = compute_gaze_attention_loss(
                attention_map, gaze, labels, gaze_loss_type
            )
            gaze_loss_scaled = gaze_loss_raw * gaze_loss_scale  # Apply scaling factor
            # Combined loss
            loss = (1 - gaze_weight) * cls_loss + gaze_weight * gaze_loss_scaled
        else:
            gaze_loss_raw = torch.tensor(0.0).to(device)
            gaze_loss_scaled = torch.tensor(0.0).to(device)
            loss = cls_loss
        
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
        total_cls += cls_loss.item()
        total_gaze += gaze_loss_scaled.item() if has_gaze else 0.0
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Record batch statistics
        if stats_tracker:
            batch_stats = {
                'epoch': epoch,
                'batch_loss': loss.item(),
                'cls_loss': cls_loss.item(),
                'gaze_loss': gaze_loss_scaled.item() if has_gaze else 0.0,
                'batch_accuracy': (preds == labels).float().mean().item(),
                'has_gaze': has_gaze,
                'lr': current_lr,
                'gaze_alpha': model.gaze_alpha.item() if hasattr(model, 'gaze_alpha') else 0.0
            }
            stats_tracker.record_batch(batch_stats, batch_files)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls': f"{cls_loss.item():.4f}",
            'gaze': f"{gaze_loss_scaled.item():.4f}" if has_gaze else '0.0',
            'acc': f"{correct/total*100:.1f}%",
            'α': f"{model.gaze_alpha.item():.3f}" if hasattr(model, 'gaze_alpha') else '0.0',
            'gaze_data': 'Y' if has_gaze else 'N'
        })
    
    # Calculate epoch averages
    avg_loss = total_loss / max(len(train_loader), 1)
    avg_cls = total_cls / max(len(train_loader), 1)
    avg_gaze = total_gaze / max(batches_with_gaze, 1)
    acc = correct / total * 100 if total > 0 else 0.0
    
    # Gaze alpha gradient statistics
    avg_gaze_grad = sum(gaze_alpha_grads) / len(gaze_alpha_grads) if gaze_alpha_grads else 0.0
    
    train_stats = {
        'loss': avg_loss,
        'cls_loss': avg_cls,
        'gaze_loss': avg_gaze,
        'acc': acc,
        'batches_with_gaze': batches_with_gaze,
        'samples_with_gaze': samples_with_gaze,
        'gaze_alpha': model.gaze_alpha.item() if hasattr(model, 'gaze_alpha') else 0.0,
        'gaze_alpha_grad': avg_gaze_grad,
        'lr': current_lr,
        'gaze_lr': gaze_lr
    }
    
    return train_stats


def evaluate_conformer_combined(model, eval_loader, device, class_weights=None, 
                                gaze_loss_type='cosine', gaze_weight=0.5, 
                                gaze_loss_scale=1.0):
    """
    Evaluation function for Conformer combined model.
    
    Args:
        model: Conformer_Combined model
        eval_loader: Evaluation data loader
        device: Device (cuda/cpu)
        class_weights: Optional class weights
        gaze_loss_type: Type of gaze loss
        gaze_weight: Weight for gaze loss
        gaze_loss_scale: Scaling for gaze loss
        
    Returns:
        Evaluation statistics dictionary
    """
    model.eval()
    total_loss = total_cls = total_gaze = 0.0
    correct = total = 0
    batches_with_gaze = samples_with_gaze = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_loader:
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
            
            # Forward pass
            if has_gaze:
                outputs = model(eeg, gaze, return_attention=True)
            else:
                outputs = model(eeg, None, return_attention=True)
            
            # Extract logits and attention map
            if isinstance(outputs, dict):
                logits = outputs['logits']
                attention_map = outputs.get('attention_map', None)
            else:
                if len(outputs) == 2:
                    _, logits = outputs
                    attention_map = None
                else:
                    _, logits, attention_map = outputs
            
            # Classification loss
            if class_weights is not None:
                cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
            else:
                cls_loss = F.cross_entropy(logits, labels)
            
            # Gaze loss
            if has_gaze and attention_map is not None:
                gaze_loss_raw = compute_gaze_attention_loss(
                    attention_map, gaze, labels, gaze_loss_type
                )
                gaze_loss_scaled = gaze_loss_raw * gaze_loss_scale
                loss = (1 - gaze_weight) * cls_loss + gaze_weight * gaze_loss_scaled
            else:
                gaze_loss_scaled = torch.tensor(0.0).to(device)
                loss = cls_loss
            
            # Statistics
            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_gaze += gaze_loss_scaled.item() if has_gaze else 0.0
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / max(len(eval_loader), 1)
    avg_cls = total_cls / max(len(eval_loader), 1)
    avg_gaze = total_gaze / max(batches_with_gaze, 1)
    acc = correct / total * 100 if total > 0 else 0.0
    
    eval_stats = {
        'loss': avg_loss,
        'cls_loss': avg_cls,
        'gaze_loss': avg_gaze,
        'acc': acc,
        'batches_with_gaze': batches_with_gaze,
        'samples_with_gaze': samples_with_gaze,
        'predictions': all_preds,
        'labels': all_labels
    }
    
    return eval_stats
