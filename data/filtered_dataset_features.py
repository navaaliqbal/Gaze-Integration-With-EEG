"""
Filtered dataset wrapper that only includes samples with both EEG and gaze data
"""
import os
import glob
from typing import Optional, List
from collections import defaultdict

# Import from utils
from utils.file_utils import list_files_recursive, normalize_basename

class FilteredEEGGazeFixationDataset:
    """
    Wraps EEGGazeFixationDataset and filters it to only include indices whose
    normalized basename exists both as a .npz and a .json
    """
    
    def __init__(self, 
                 data_dir: str,
                 gaze_json_dir: str,
                 dataset_cls,
                 dataset_kwargs=None,
                 suffixes_to_strip=None):
        
        self.data_dir = data_dir
        self.gaze_json_dir = gaze_json_dir
        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dataset_kwargs or {}
        self.suffixes_to_strip = suffixes_to_strip or []
        
        # Build on-disk maps
        self._build_on_disk_maps()
        
        # Ensure gaze_json_dir is in dataset_kwargs
        if 'gaze_json_dir' in self.dataset_kwargs:
            del self.dataset_kwargs['gaze_json_dir']
        
        # Instantiate underlying dataset
        try:
            self.original_dataset = self.dataset_cls(
                data_dir=self.data_dir,
                gaze_json_dir=self.gaze_json_dir,
                **self.dataset_kwargs
            )
        except Exception as e:
            print(f"Error creating dataset: {e}")
            raise
        
        # Build dataset index map
        self._build_dataset_index_map()
        
        # Compute filtered indices
        self.filtered_indices = self._compute_filtered_indices()
        
        # Print diagnostics
        debug_mode = self.dataset_kwargs.get('debug', False)
        if debug_mode:
            self._print_diagnostics()
    
    def _build_on_disk_maps(self):
        """Build maps of files on disk"""
        npz_paths = list_files_recursive(self.data_dir, '.npz')
        json_paths = list_files_recursive(self.gaze_json_dir, '.json')
        
        self.npz_map = defaultdict(list)
        for p in npz_paths:
            nb = normalize_basename(p, self.suffixes_to_strip)
            self.npz_map[nb].append(p)
        
        self.json_map = defaultdict(list)
        for p in json_paths:
            nb = normalize_basename(p, self.suffixes_to_strip)
            self.json_map[nb].append(p)
        
        self.disk_npz_basenames = set(self.npz_map.keys())
        self.disk_json_basenames = set(self.json_map.keys())
        self.disk_matched_basenames = self.disk_npz_basenames & self.disk_json_basenames
    
    def _build_dataset_index_map(self):
        """Map dataset indices to normalized basenames"""
        self.dataset_index_to_base = {}
        self.dataset_base_to_indices = defaultdict(list)
        
        n = len(self.original_dataset)
        for idx in range(n):
            try:
                sample = self.original_dataset[idx]
                f = None
                
                if isinstance(sample, dict) and 'file' in sample:
                    f = sample['file']
                elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
                    f = sample[2]
                
                if f is not None:
                    nb = normalize_basename(f, self.suffixes_to_strip)
                else:
                    nb = None
                    
            except Exception as e:
                nb = None
            
            self.dataset_index_to_base[idx] = nb
            if nb:
                self.dataset_base_to_indices[nb].append(idx)
    
    def _compute_filtered_indices(self):
        """Compute which indices to keep based on disk matches"""
        kept = []
        for nb in sorted(self.disk_matched_basenames):
            idxs = self.dataset_base_to_indices.get(nb, [])
            if idxs:
                kept.extend(idxs)
        return sorted(set(kept))
    
    def _print_diagnostics(self):
        """Print diagnostic information"""
        print("\n" + "=" * 70)
        print("FILTEREDEEGGAZEFIXATIONDATASET DIAGNOSTICS".center(70))
        print("=" * 70)
        print(f"  data_dir: {self.data_dir}")
        print(f"  gaze_json_dir: {self.gaze_json_dir}")
        print(f"\n  Disk: {len(self.npz_map)} unique npz basenames, "
              f"{len(self.json_map)} unique json basenames")
        print(f"  Disk matched basenames: {len(self.disk_matched_basenames)}")
        
        if self.disk_matched_basenames:
            print(f"    Examples (first 5): {sorted(list(self.disk_matched_basenames))[:5]}")
        
        dataset_bases = set(k for k in self.dataset_base_to_indices.keys() if k)
        print(f"\n  Dataset reported basenames: {len(dataset_bases)}")
        
        on_disk_not_in_dataset = sorted(list(self.disk_matched_basenames - dataset_bases))
        in_dataset_not_on_disk = sorted(list(dataset_bases - self.disk_matched_basenames))
        
        print(f"\n  Normalized diff counts:")
        print(f"    On-disk matched but NOT in dataset: {len(on_disk_not_in_dataset)}")
        print(f"    In dataset but NOT matched on-disk: {len(in_dataset_not_on_disk)}")
        
        print(f"\n  Filtered indices kept: {len(self.filtered_indices)} "
              f"(out of original {len(self.original_dataset)})")
        print("=" * 70)
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        orig_idx = self.filtered_indices[idx]
        return self.original_dataset[orig_idx]