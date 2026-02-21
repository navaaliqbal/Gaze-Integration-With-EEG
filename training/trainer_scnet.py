"""
Training functions for SCNet models - handles all variants (base, input, output, combined)
"""
import torch
import torch.nn.functional as F
import os

def train_epoch_scnet(model, train_loader, optimizer, device, gaze_weight=0.2,
                     gaze_loss_type='cosine', class_weights=None, 
                     stats_tracker=None, epoch=0, gaze_loss_scale=1.0):
    """
    Training epoch for SCNet models - handles all variants
    
    Args:
        model: SCNet model (base, input, output, or combined)
        train_loader: DataLoader
        optimizer: Optimizer
        device: Device
        gaze_weight: Weight for gaze loss (used only for output/combined)
        gaze_loss_type: Type of gaze loss
        class_weights: Class weights for loss
        stats_tracker: Statistics tracker
        epoch: Current epoch
        gaze_loss_scale: Scale factor for gaze loss
    """
    model.train()
    total_loss = total_cls = total_gaze = 0.0
    correct = total = 0
    batches_with_gaze = samples_with_gaze = 0
    
    # Determine model type
    model_name = model.__class__.__name__.lower()
    is_output_model = 'output' in model_name
    is_input_model = 'input' in model_name
    is_combined_model = 'combined' in model_name
    is_base_model = 'base' in model_name or (not is_output_model and not is_input_model and not is_combined_model)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print epoch start with model info
    model_type = "BASE" if is_base_model else \
                 "INPUT" if is_input_model else \
                 "OUTPUT" if is_output_model else \
                 "COMBINED" if is_combined_model else "UNKNOWN"
    print(f"\nTraining Epoch {epoch+1} [SCNet-{model_type}]...")
    
    # Iterate without tqdm
    for batch_idx, batch in enumerate(train_loader):
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)
        
        # Get batch files (optional)
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
        
        # ========== FORWARD PASS BASED ON MODEL TYPE ==========
        
        # CASE 1: BASE MODEL - no gaze, no attention
        if is_base_model:
            logits = model(eeg)
            attention_map = None
        
        # CASE 2: INPUT MODEL - takes gaze as input, no attention
        elif is_input_model:
            logits = model(eeg, gaze)
            attention_map = None
        
        # CASE 3: OUTPUT MODEL - returns attention map
        elif is_output_model:
            if has_gaze:
                # When we have gaze, we want attention for loss
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
                # No gaze, don't need attention
                logits = model(eeg, return_attention=False)
                attention_map = None
        
        # CASE 4: COMBINED MODEL - takes gaze and returns attention
        elif is_combined_model:
            if has_gaze:
                outputs = model(eeg, gaze, return_attention=True)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                    attention_map = outputs.get('attention_map', None)
                elif isinstance(outputs, tuple):
                    logits, attention_map = outputs
                else:
                    logits = outputs
                    attention_map = None
            else:
                # No gaze, still might want attention?
                outputs = model(eeg, None, return_attention=False)
                logits = outputs if not isinstance(outputs, dict) else outputs['logits']
                attention_map = None
        
        else:
            # Fallback - assume base model behavior
            logits = model(eeg)
            attention_map = None
        
        # ========== LOSS COMPUTATION ==========
        
        # Classification loss
        if class_weights is not None:
            cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            cls_loss = F.cross_entropy(logits, labels)
        
        # Gaze loss (only for output/combined models with gaze data)
        gaze_loss = torch.tensor(0.0).to(device)
        if has_gaze and attention_map is not None and (is_output_model or is_combined_model):
            from training.losses import compute_gaze_attention_loss
            gaze_loss_raw = compute_gaze_attention_loss(attention_map, gaze, labels, gaze_loss_type)
            gaze_loss = gaze_loss_scale * gaze_loss_raw
            loss = cls_loss + gaze_weight * gaze_loss
        else:
            loss = cls_loss
        
        # ========== BACKWARD PASS ==========
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # ========== STATISTICS ==========
        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_gaze += gaze_loss.item() if has_gaze and (is_output_model or is_combined_model) else 0.0
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Print batch progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            batch_acc = (preds == labels).float().mean().item() * 100
            gaze_info = f", Gaze Loss={gaze_loss.item():.4f}" if gaze_loss.item() > 0 else ""
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                  f"Loss={loss.item():.4f}, Cls Loss={cls_loss.item():.4f}{gaze_info}, Acc={batch_acc:.1f}%")
        
        # Record batch statistics
        if stats_tracker:
            batch_stats = {
                'epoch': epoch,
                'batch_loss': loss.item(),
                'batch_cls_loss': cls_loss.item(),
                'batch_gaze_loss': gaze_loss.item() if has_gaze and (is_output_model or is_combined_model) else 0.0,
                'batch_accuracy': (preds == labels).float().mean().item(),
                'has_gaze': has_gaze,
                'lr': current_lr,
                'model_type': model_type
            }
            stats_tracker.record_batch(batch_idx, batch_stats)
    
    # ========== EPOCH STATISTICS ==========
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
        'model_type': model_type
    }
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1} Summary [SCNet-{model_type}]:")
    print(f"  Avg Loss: {avg_loss:.4f}, Avg Cls: {avg_cls:.4f}, Avg Gaze: {avg_gaze:.4f}")
    print(f"  Accuracy: {acc:.2f}%, Samples with gaze: {samples_with_gaze}/{total}")
    
    return train_stats