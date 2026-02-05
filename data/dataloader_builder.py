"""
Functions for building dataloaders with proper filtering
"""
import torch
from collections import Counter
import numpy as np
import os

from data.filtered_dataset import FilteredEEGGazeFixationDataset
from data.dataset import EEGGazeFixationDataset
from utils.debugger import DataDebugger

def get_dataloaders_fixed(data_dir, batch_size, seed, target_length=None, indexes=None,
                         gaze_json_dir=None, only_matched=True,
                         suffixes_to_strip=None, **kwargs):
    """
    Build dataloaders using FilteredEEGGazeFixationDataset
    """
    DataDebugger.print_header("BUILD DATALOADERS (FIXED)")
    
    # Use configurable subdirectories
    train_dir = os.path.join(data_dir, kwargs.get('train_subdir', 'train'))
    eval_dir = os.path.join(data_dir, kwargs.get('eval_subdir', 'eval'))
    
    print(f"  Main data_dir: {data_dir}")
    print(f"  Train directory: {train_dir}")
    print(f"  Eval directory: {eval_dir}")
    print(f"  Gaze JSON directory: {gaze_json_dir}")
    
    # Instantiate the filtered wrapper for train and eval
    dataset_kwargs = {
        'indexes': indexes,
        'target_length': target_length,
        'eeg_sampling_rate': kwargs.get('eeg_sampling_rate', 50.0)
    }
    
    # Train dataset
    trainset = FilteredEEGGazeFixationDataset(
        data_dir=train_dir,
        gaze_json_dir=gaze_json_dir,
        dataset_cls=EEGGazeFixationDataset,
        dataset_kwargs=dataset_kwargs,
        suffixes_to_strip=suffixes_to_strip
    )
    
    # Calculate class weights for sampler
    labels = [trainset[i]['label'] for i in range(len(trainset))]
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)
    
    print(f"\n  Training label distribution: {dict(class_counts)}")
    print(f"  Number of classes: {num_classes}, Total training samples: {total_samples}")
    
  
    # Eval dataset
    evalset = FilteredEEGGazeFixationDataset(
        data_dir=eval_dir,
        gaze_json_dir=gaze_json_dir,
        dataset_cls=EEGGazeFixationDataset,
        dataset_kwargs=dataset_kwargs,
        suffixes_to_strip=suffixes_to_strip
    )
    
    # Create worker init function
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=min(batch_size, len(trainset)) if len(trainset) > 0 else 1,
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    eval_loader = torch.utils.data.DataLoader(
        evalset,
        batch_size=min(batch_size, len(evalset)) if len(evalset) > 0 else 1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    print("\nDATALOADER SUMMARY")
    print(f"  Train samples: {len(trainset)} | batches: {len(train_loader)}")
    print(f"  Eval samples:  {len(evalset)} | batches: {len(eval_loader)}")
    
    # Quick dataset diagnostics
    DataDebugger.analyze_dataset(trainset, "Filtered Train Dataset", max_samples=5)
    DataDebugger.analyze_dataset(evalset, "Filtered Eval Dataset", max_samples=5)
    
    return train_loader, eval_loader, {
        'train_filtered': len(trainset),
        'eval_filtered': len(evalset),
        'train_disk_matched': len(trainset.disk_matched_basenames) if hasattr(trainset, 'disk_matched_basenames') else 0,
        'eval_disk_matched': len(evalset.disk_matched_basenames) if hasattr(evalset, 'disk_matched_basenames') else 0
    }