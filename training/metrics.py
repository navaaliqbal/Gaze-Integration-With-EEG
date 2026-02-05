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

def evaluate_model_comprehensive(model, eval_loader, device, stats_tracker=None, dataset_name="eval", return_attention=True):
    """Enhanced evaluation with comprehensive statistics."""
    model.eval()
    all_labels = []
    all_preds = []
    all_files = []
    all_probs = []
    all_attention_maps = []
    all_gaze_maps = []
    
    # Check if model is combined type (needs gaze parameter)
    model_name = model.__class__.__name__.lower()
    is_combined = 'combined' in model_name
    is_scnet_input = 'scnet' in model_name and 'input' in model_name
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
            
            # Get gaze if available (for combined models)
            gaze = None
            if 'gaze' in batch and batch['gaze'] is not None:
                gaze = batch['gaze'].to(device)
            
            # Forward pass with attention - handle combined models
            if is_combined:
                outputs = model(eeg, gaze, return_attention=return_attention)
            elif is_scnet_input:
                logits = model(eeg, gaze)
                attention_maps = None
            else:
                outputs = model(eeg, return_attention=return_attention)
            
            if not is_scnet_input:
                if isinstance(outputs, tuple):
                    logits, attention_maps = outputs
                elif isinstance(outputs, dict):
                    logits = outputs['logits']
                    attention_maps = outputs.get('attention_map', None)
                else:
                    logits = outputs
                    attention_maps = None
            
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
            if 'gaze' in batch and batch['gaze'] is not None:
                gaze = batch['gaze'].cpu().numpy()
                all_gaze_maps.extend(gaze)
            
            # Print progress every batch
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(eval_loader):
                print(f"  Processed {batch_idx + 1}/{len(eval_loader)} batches")
    
    # Calculate metrics
    acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    eval_stats = {
        'acc': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'balanced_acc': balanced_acc,
        'total_samples': len(all_labels)
    }
    
    # Return based on return_attention flag
    if return_attention:
        return eval_stats, all_labels, all_preds, all_files, all_attention_maps
    else:
        return eval_stats, all_labels, all_preds, all_files

def collect_all_attention_maps(model, dataloader, device, stats_tracker=None, dataset_name="full"):
    """Collect attention maps for all samples in a dataset."""
    print(f"\nCollecting attention maps for {dataset_name} dataset...")
    model.eval()
    
    all_data = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Collecting {dataset_name}"):
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
            
            # Forward pass with attention
            outputs = model(eeg, return_attention=True)
            if isinstance(outputs, tuple):
                logits, attention_maps = outputs
            elif isinstance(outputs, dict):
                logits = outputs['logits']
                attention_maps = outputs.get('attention_map', None)
            else:
                logits = outputs
                attention_maps = None
            
            # Get predictions
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Store all data
            for i in range(eeg.shape[0]):
                sample_data = {
                    'file': batch_files[i] if i < len(batch_files) else f"unknown_{i}",
                    'eeg': eeg[i].cpu().numpy(),
                    'attention_map': attention_maps[i].cpu().numpy() if attention_maps is not None else None,
                    'label': labels[i].cpu().item(),
                    'prediction': preds[i].cpu().item(),
                    'probability': probs[i].cpu().numpy(),
                    'logits': logits[i].cpu().numpy(),
                    'dataset': dataset_name
                }
                
                # Add gaze data if available
                if 'gaze' in batch and batch['gaze'] is not None:
                    sample_data['gaze_map'] = batch['gaze'][i].numpy()
                
                all_data.append(sample_data)
    
    print(f"Collected {len(all_data)} attention maps from {dataset_name} dataset")
    return all_data

def compute_gaze_loss_scale(model, train_loader, device, gaze_loss_type='mse'):
    """
    Compute a principled scaling factor for gaze_loss so that its gradient contribution
    is comparable to classification loss.
    
    Uses a single training batch, computes the scale once, and returns it fixed for the entire training.
    
    Note: Only applicable for models with attention-based gaze integration (output/combined).
    Input integration models don't use gaze loss, so this returns 1.0 immediately.
    """
    print("\n" + "=" * 80)
    print("COMPUTING GAZE LOSS SCALING FACTOR")
    print("=" * 80)
    
    # Check model type first
    model_name = model.__class__.__name__.lower()
    is_input = 'input' in model_name and 'combined' or 'output' not in model_name
    is_combined = 'combined' in model_name
    
    # Input integration doesn't use gaze loss (gaze is at input level)
    if is_input:
        print("  INFO: Input integration model detected - no gaze loss scaling needed")
        return 1.0, {'info': 'Input model - no gaze loss'}
    
    # Get a batch with gaze data
    for batch in train_loader:
        if 'gaze' in batch and batch['gaze'] is not None:
            break
    else:
        print("  WARNING: No gaze data found in training loader!")
        return 1.0, {'error': 'No gaze data'}
    
    model.eval()  # Just for computation, not training
    
    eeg = batch['eeg'].to(device)
    labels = batch['label'].to(device)
    gaze = batch['gaze'].to(device)
    
    with torch.no_grad():
        # Forward pass with attention (combined model needs gaze for forward)
        if is_combined:
            outputs = model(eeg, gaze, return_attention=True)
        else:
            outputs = model(eeg, return_attention=True)
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
            attention_map = outputs.get('attention_map', None)
        else:
            logits = outputs
            attention_map = None
        
        if attention_map is None:
            print("  WARNING: Model does not return attention map!")
            return 1.0, {'error': 'No attention map'}
        
        # Compute raw losses
        cls_loss = F.cross_entropy(logits, labels).item()
        gaze_loss = compute_gaze_attention_loss(attention_map, gaze, labels, gaze_loss_type).item()
        
        # Compute scaling factor: cls_loss / gaze_loss
        if gaze_loss > 1e-8:
            gaze_loss_scale = cls_loss / gaze_loss
        else:
            gaze_loss_scale = 1.0
        
        # Clip to reasonable range
        gaze_loss_scale = np.clip(gaze_loss_scale, 0.1, 100.0)
        
        # Diagnostic information
        metrics = {
            'cls_loss_raw': cls_loss,
            'gaze_loss_raw': gaze_loss,
            'gaze_loss_scale': gaze_loss_scale,
            'batch_size': eeg.shape[0],
            'has_gaze': True,
            'samples_with_gaze': eeg.shape[0],
            'loss_ratio': cls_loss / gaze_loss if gaze_loss > 0 else float('inf')
        }
        
        print(f"\n  Loss Scaling Analysis:")
        print(f"    Classification loss: {cls_loss:.6f}")
        print(f"    Gaze loss (raw): {gaze_loss:.6f}")
        print(f"    Loss ratio (cls/gaze): {cls_loss/gaze_loss:.2f}:1")
        print(f"    Recommended gaze_loss_scale: {gaze_loss_scale:.2f}")
        print(f"    With gaze_weight=1.0: Effective scale = {gaze_loss_scale:.2f}")
        
        return gaze_loss_scale, metrics