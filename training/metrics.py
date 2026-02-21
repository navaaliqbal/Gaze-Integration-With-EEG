"""
Evaluation metrics and functions
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import (classification_report, f1_score, 
                           precision_score, recall_score, 
                           balanced_accuracy_score)
from training.losses import compute_gaze_attention_loss


def evaluate_model_comprehensive(model, eval_loader, device, stats_tracker=None, dataset_name="eval", 
                                return_attention=False, gaze_weight=0.0, gaze_loss_type='mse', 
                                class_weights=None, gaze_loss_scale=1.0):
    """Enhanced evaluation with comprehensive statistics including loss computation.
       Handles all SCNet variants (base, input, output, combined)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_files = []
    all_probs = []
    all_attention_maps = []
    all_gaze_maps = []
    
    # Determine model type
    model_name = model.__class__.__name__.lower()
    is_output_model = 'output' in model_name
    is_input_model = 'input' in model_name
    is_combined_model = 'combined' in model_name
    is_base_model = 'base' in model_name or (not is_output_model and not is_input_model and not is_combined_model)
    
    model_type = "BASE" if is_base_model else \
                 "INPUT" if is_input_model else \
                 "OUTPUT" if is_output_model else \
                 "COMBINED" if is_combined_model else "UNKNOWN"
    
    print(f"\nEvaluating {dataset_name} dataset with {model_type} model...")
    
    # Loss tracking
    total_loss = 0.0
    total_cls_loss = 0.0
    total_gaze_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)
            
            # Get files
            batch_files = []
            if 'file' in batch:
                for f in batch['file']:
                    if isinstance(f, (bytes, bytearray)):
                        try:
                            f = f.decode('utf-8', errors='ignore')
                        except:
                            f = str(f)
                    batch_files.append(os.path.basename(str(f)))
            
            # Get gaze if available
            gaze = None
            has_gaze = 'gaze' in batch and batch['gaze'] is not None
            if has_gaze:
                gaze = batch['gaze'].to(device)
            
            # ========== FORWARD PASS BASED ON MODEL TYPE ==========
            
            # CASE 1: BASE MODEL - no gaze, no attention
            if is_base_model:
                logits = model(eeg)
                attention_maps = None
            
            # CASE 2: INPUT MODEL - takes gaze as input, no attention
            elif is_input_model:
                logits = model(eeg, gaze)
                attention_maps = None
            
            # CASE 3: OUTPUT MODEL - may return attention
            elif is_output_model:
                if return_attention:
                    outputs = model(eeg, return_attention=True)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                        attention_maps = outputs.get('attention_map', None)
                    elif isinstance(outputs, tuple):
                        logits, attention_maps = outputs
                    else:
                        logits = outputs
                        attention_maps = None
                else:
                    logits = model(eeg, return_attention=False)
                    attention_maps = None
            
            # CASE 4: COMBINED MODEL - takes gaze and may return attention
            elif is_combined_model:
                if return_attention:
                    outputs = model(eeg, gaze, return_attention=True)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                        attention_maps = outputs.get('attention_map', None)
                    elif isinstance(outputs, tuple):
                        logits, attention_maps = outputs
                    else:
                        logits = outputs
                        attention_maps = None
                else:
                    logits = model(eeg, gaze, return_attention=False)
                    attention_maps = None
            
            else:
                # Fallback - assume base model
                logits = model(eeg)
                attention_maps = None
            
            # ========== LOSS COMPUTATION ==========
            
            # Classification loss
            if class_weights is not None:
                cls_loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
            else:
                cls_loss = F.cross_entropy(logits, labels)
            
            # Gaze loss (only for output/combined models with gaze and attention)
            if has_gaze and attention_maps is not None and (is_output_model or is_combined_model) and gaze_weight > 0:
                gaze_loss_raw = compute_gaze_attention_loss(attention_maps, gaze, labels, gaze_loss_type)
                gaze_loss_scaled = gaze_loss_scale * gaze_loss_raw
                batch_loss = cls_loss + gaze_weight * gaze_loss_scaled
                total_gaze_loss += gaze_loss_scaled.item()
            else:
                batch_loss = cls_loss
            
            total_loss += batch_loss.item()
            total_cls_loss += cls_loss.item()
            num_batches += 1
            
            # Get predictions
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Store results
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_files.extend(batch_files)
            all_probs.extend(probs.tolist())
            
            if attention_maps is not None and return_attention:
                all_attention_maps.extend(attention_maps.cpu().numpy())
            
            # Store gaze maps if available
            if has_gaze:
                all_gaze_maps.extend(gaze.cpu().numpy())
            
            # Print progress every 5 batches
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(eval_loader):
                print(f"  Processed {batch_idx + 1}/{len(eval_loader)} batches")
    
    # Calculate metrics
    acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Calculate average losses
    avg_loss = total_loss / max(num_batches, 1)
    avg_cls_loss = total_cls_loss / max(num_batches, 1)
    avg_gaze_loss = total_gaze_loss / max(num_batches, 1)
    
    eval_stats = {
        'acc': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'balanced_acc': balanced_acc,
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'gaze_loss': avg_gaze_loss,
        'total_samples': len(all_labels)
    }
    
    print(f"\n{dataset_name} Results:")
    print(f"  Loss: {avg_loss:.4f} (Cls: {avg_cls_loss:.4f}, Gaze: {avg_gaze_loss:.4f})")
    print(f"  Accuracy: {acc:.2f}%, Balanced Acc: {balanced_acc:.4f}, Macro F1: {macro_f1:.4f}")
    
    # Return based on return_attention flag
    if return_attention and (is_output_model or is_combined_model):
        return eval_stats, all_labels, all_preds, all_files, all_attention_maps
    else:
        return eval_stats, all_labels, all_preds, all_files