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

def evaluate_model_comprehensive(model, eval_loader, device, stats_tracker=None, 
                                dataset_name="eval", return_attention=False):
    """Enhanced evaluation with comprehensive statistics."""
    model.eval()
    all_labels = []
    all_preds = []
    all_files = []
    all_probs = []
    all_attention_maps = []
    all_gaze_maps = []
    
    print(f"\nEvaluating on {dataset_name} set...")
    
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
            
            # Forward pass - handle based on return_attention parameter
            if return_attention:
                outputs = model(eeg, return_attention=True)
            else:
                outputs = model(eeg, return_attention=False)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                # Handle tuple output (logits, attention_maps)
                logits, attention_maps = outputs
            elif isinstance(outputs, dict):
                # Handle dictionary output
                logits = outputs['logits']
                attention_maps = outputs.get('attention_map', None)
            else:
                # Direct logits output
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