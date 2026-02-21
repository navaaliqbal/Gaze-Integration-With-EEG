"""
Generic trainer that dispatches to specific trainers based on model type
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from training.trainer_input import train_epoch_input
from training.trainer_output import train_epoch_output
from training.trainer_combined import train_epoch_combined
from training.trainer_cls_only import train_epoch_cls_only
from training.trainer_scnet import train_epoch_scnet


def train_epoch_eegnet_base(model, train_loader, optimizer, device, 
                           class_weights=None, stats_tracker=None, epoch=0,
                           **kwargs):
    """
    Specialized trainer for base EEGNet (no gaze integration)
    """
    model.train()
    total_loss = 0.0
    correct = total = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Epoch {epoch+1} [EEGNet-Only]")
    
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
        'cls_loss': avg_loss,
        'gaze_loss': 0.0,
        'acc': acc,
        'total_batches': len(train_loader),
        'total_samples': total,
        'lr': current_lr,
        'gaze_samples': 0,
        'gaze_batches': 0
    }
    
    return train_stats


def train_epoch(model, train_loader, optimizer, device, **kwargs):
    """
    Dispatch to appropriate trainer based on model class name
    """
    model_name = model.__class__.__name__.lower()
    
    # Debug print to see what model we're dealing with
    print(f"\nDEBUG - Model class: {model.__class__.__name__}")
    print(f"DEBUG - Model name (lower): {model_name}")
    
    # Check for SCNet first
    if 'scnet' in model_name:
        print("DEBUG -> Using SCNet trainer")
        return train_epoch_scnet(model, train_loader, optimizer, device, **kwargs)
    
    # Check for specific EEGNet variants
    if 'eegnet' in model_name:
        if 'input' in model_name:
            print("DEBUG -> Using EEGNet Input trainer")
            return train_epoch_input(model, train_loader, optimizer, device, **kwargs)
        elif 'output' in model_name:
            print("DEBUG -> Using EEGNet Output trainer")
            return train_epoch_output(model, train_loader, optimizer, device, **kwargs)
        elif 'combined' in model_name:
            print("DEBUG -> Using EEGNet Combined trainer")
            return train_epoch_combined(model, train_loader, optimizer, device, **kwargs)
        elif 'base' in model_name or model_name == 'eegnet_base':
            print("DEBUG -> Using EEGNet Base trainer")
            return train_epoch_eegnet_base(model, train_loader, optimizer, device, **kwargs)
        else:
            # Fallback for any other EEGNet variant
            print(f"DEBUG -> Unknown EEGNet variant, using base trainer")
            return train_epoch_eegnet_base(model, train_loader, optimizer, device, **kwargs)
    
    # Check for NeuroGATE variants
    if 'neurogate' in model_name or 'gate' in model_name:
        if 'input' in model_name:
            print("DEBUG -> Using NeuroGATE Input trainer")
            return train_epoch_input(model, train_loader, optimizer, device, **kwargs)
        elif 'output' in model_name:
            print("DEBUG -> Using NeuroGATE Output trainer")
            return train_epoch_output(model, train_loader, optimizer, device, **kwargs)
        elif 'combined' in model_name:
            print("DEBUG -> Using NeuroGATE Combined trainer")
            return train_epoch_combined(model, train_loader, optimizer, device, **kwargs)
    
    # Default fallback
    print(f"DEBUG -> Warning: Unknown model type {model_name}, using cls-only trainer")
    return train_epoch_cls_only(model, train_loader, optimizer, device, **kwargs)