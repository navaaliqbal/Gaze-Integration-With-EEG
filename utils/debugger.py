"""
Data debugging helpers for comprehensive dataset analysis
"""
import traceback
from collections import Counter
import numpy as np
import torch
import os

class DataDebugger:
    """Data debugging helpers for comprehensive dataset analysis."""
    
    @staticmethod
    def print_header(title, width=80, char="="):
        """Print a formatted header."""
        print("\n" + char * width)
        print(title.center(width))
        print(char * width)
    
    @staticmethod
    def analyze_dataset(dataset, name="Dataset", max_samples=20):
        """Analyze dataset content and statistics."""
        DataDebugger.print_header(f"ANALYZE {name}")
        print(f"  Total samples: {len(dataset)}")
        
        if len(dataset) == 0:
            print("  WARNING: Dataset is empty!")
            return None, None
        
        sample_count = min(max_samples, len(dataset))
        all_labels = []
        all_files = []
        
        for i in range(sample_count):
            try:
                sample = dataset[i]
                if isinstance(sample, dict):
                    label = sample.get('label', None)
                    eeg = sample.get('eeg', None)
                    gaze = sample.get('gaze', None)
                    
                    f = sample.get('file', None)
                    if f is not None:
                        if isinstance(f, (bytes, bytearray)):
                            try:
                                f = f.decode('utf-8', errors='ignore')
                            except:
                                f = str(f)
                        basename = os.path.splitext(os.path.basename(str(f)))[0]
                        all_files.append(basename)
                    
                    all_labels.append(label)
                    
                    print(f"  Sample {i}: Label={label}")
                    if eeg is not None:
                        print(f"    EEG shape: {eeg.shape}, dtype: {eeg.dtype}")
                    if gaze is not None:
                        print(f"    Gaze shape: {gaze.shape}, dtype: {gaze.dtype}")
                    if f is not None:
                        print(f"    File: {basename[:50]}")
                else:
                    print(f"  Sample {i}: Not a dict, type: {type(sample)}")
                    
            except Exception as e:
                print(f"  Sample {i}: Error reading sample: {e}")
                traceback.print_exc()
        
        if all_labels:
            counter = Counter(all_labels)
            print(f"\n  Label distribution (sampled): {dict(counter)}")
            print(f"  Number of unique labels: {len(counter)}")
        
        # Check data types
        if sample_count > 0:
            try:
                sample = dataset[0]
                if isinstance(sample, dict):
                    print(f"\n  Data types in first sample:")
                    for key, value in sample.items():
                        if torch.is_tensor(value):
                            print(f"    {key}: Tensor shape={value.shape}, dtype={value.dtype}")
                        elif isinstance(value, np.ndarray):
                            print(f"    {key}: Numpy shape={value.shape}, dtype={value.dtype}")
                        elif isinstance(value, (list, tuple)):
                            print(f"    {key}: {type(value).__name__} length={len(value)}")
                        else:
                            print(f"    {key}: {type(value).__name__}")
            except:
                pass
        
        return all_labels, all_files
    
    @staticmethod
    def analyze_dataloader(dataloader, name="Dataloader", max_batches=3):
        """Analyze dataloader batches."""
        DataDebugger.print_header(f"ANALYZE {name}")
        print(f"  Total batches: {len(dataloader)}")
        
        if len(dataloader) == 0:
            print("  WARNING: Dataloader is empty!")
            return None, None
        
        all_labels = []
        all_files = []
        
        for bidx, batch in enumerate(dataloader):
            if bidx >= max_batches:
                break
            
            print(f"\n  Batch {bidx+1}:")
            
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                        if key == 'label':
                            all_labels.extend(value.numpy().tolist())
                            print(f"      labels: {value.numpy().tolist()}")
                        if key == 'file':
                            files = []
                            for f in value:
                                if isinstance(f, (bytes, bytearray)):
                                    try:
                                        f = f.decode('utf-8', errors='ignore')
                                    except:
                                        f = str(f)
                                files.append(os.path.splitext(os.path.basename(str(f)))[0])
                            all_files.extend(files)
                            print(f"      files: {files[:3]}...")
                    elif isinstance(value, list):
                        print(f"    {key}: list length={len(value)}")
                    else:
                        print(f"    {key}: {type(value)}")
            
            elif isinstance(batch, (list, tuple)):
                print(f"    Batch is {type(batch).__name__} with {len(batch)} elements")
                for i, item in enumerate(batch):
                    if torch.is_tensor(item):
                        print(f"      [{i}]: shape={item.shape}, dtype={item.dtype}")
        
        if all_labels:
            counter = Counter(all_labels)
            print(f"\n  Label distribution in seen batches: {dict(counter)}")
            print(f"  Total samples seen: {len(all_labels)}")
        
        return all_labels, all_files