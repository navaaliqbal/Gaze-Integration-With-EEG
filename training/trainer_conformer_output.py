"""
Training functions for Conformer with OUTPUT gaze integration.
Adapted from trainer_output.py with Conformer-specific handling.
Includes focal loss for class imbalance.
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


def train_epoch_conformer_output(model, train_loader, optimizer, device, gaze_weight=0.5, 
                                   gaze_loss_type='cosine', class_weights=None, stats_tracker=None, 
                                   epoch=0, gaze_loss_scale=1.0, use_focal_loss=True, focal_gamma=2.0,
                                   label_smoothing=0.1):
    """
    Enhanced training epoch for Conformer output integration with focal loss.
    
    Args:
        model: ConformerGazeOutput model
        train_loader: Training data loader
        optimizer: Optimizer (supports dual optimizer if needed)
        device: torch device
        gaze_weight: Weight for gaze loss (0-1), 0.5 = equal weight
        gaze_loss_type: 'cosine', 'mse', or 'combined'
        class_weights: Class weights for imbalanced dataset
        stats_tracker: Statistics tracker object
        epoch: Current epoch number
        gaze_loss_scale: Additional scaling for gaze loss
        use_focal_loss: Whether to use focal loss instead of CE
        focal_gamma: Gamma parameter for focal loss (higher = more focus on hard)
        label_smoothing: Label smoothing factor (0-1)
    """
    model.train()
    total_loss = total_cls = total_gaze = 0.0
    correct = total = 0
    batches_with_gaze = samples_with_gaze = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"Class weights used in loss computation: {class_weights}")
    
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
        
        # Forward pass - Conformer returns tuple or dict
        if has_gaze:
            outputs = model(eeg, return_attention=True)
            # Handle dict output
            if isinstance(outputs, dict):
                logits = outputs['logits']
                attention_map = outputs['attention_map']
            else:
                # Fallback for tuple (features, logits, attention_map)
                if len(outputs) == 3:
                    _, logits, attention_map = outputs
                else:
                    logits, attention_map = outputs
        else:
            outputs = model(eeg, return_attention=False)
            # Handle tuple output (features, logits)
            if isinstance(outputs, tuple):
                _, logits = outputs
            else:
                logits = outputs
            attention_map = None
        
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
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls': f"{cls_loss.item():.4f}",
            'gaze': f"{gaze_loss_scaled.item():.4f}" if has_gaze else "N/A",
            'acc': f"{correct/total*100:.1f}%"
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
        'gaze_loss_scale': gaze_loss_scale,
        'use_focal_loss': use_focal_loss,
        'focal_gamma': focal_gamma if use_focal_loss else None
    }
    
    return train_stats
