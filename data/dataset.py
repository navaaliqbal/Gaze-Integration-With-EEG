"""
Dataset classes for EEG and gaze data - USING ORIGINAL VERSION
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import json
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import signal, interpolate
from dataclasses import dataclass
import matplotlib.pyplot as plt
import re

@dataclass
class Fixation:
    """Represents a single gaze fixation"""
    start_time: float
    end_time: float
    duration: float
    x: float           # EEG TIME in seconds
    y: float           # Position
    channels: List[str]
    num_points: int
    
    @property
    def attention_weight(self) -> float:
        """Calculate attention weight based on fixation characteristics"""
        duration_weight = min(1.0, self.duration * 2)
        density_weight = min(1.0, self.num_points / 50)
        return duration_weight * density_weight * 0.8 + 0.2
    
    def __post_init__(self):
        """Validate after initialization"""
        if not isinstance(self.channels, list):
            self.channels = [self.channels] if self.channels else []

class EEGGazeFixationDataset(Dataset):
    """
    Dataset that works with your get_datanpz() pipeline AND gaze JSON fixation data.
    
    KEY FIX: Uses ONLY x coordinate (EEG time in seconds) for alignment.
    Ignores start_time/end_time (they're in time.time() units).
    """
    
    def __init__(self, data_dir: str, 
                 indexes: Optional[List[int]] = None,
                 target_length: Optional[int] = None,
                 gaze_json_dir: Optional[str] = None,
                 gaze_json_pattern: str = "*.json",
                 eeg_sampling_rate: float = 100.0,
                 channel_mapping: Optional[Dict[str, int]] = None,
                 debug: bool = False,
                 suffixes_to_strip: List[str] = None,
                 debug_sample: str = None,  # NEW: Specific sample to debug
                 **kwargs):
        """
        Args:
            data_dir: Directory with EEG .npz files (from get_datanpz)
            indexes: Subset indices to use
            target_length: Target time length for EEG (e.g., 15000 for 5 mins)
            gaze_json_dir: Directory with gaze JSON files
            gaze_json_pattern: Pattern to match gaze files
            eeg_sampling_rate: EEG sampling rate (from data_description)
            channel_mapping: Map from channel names to indices
            debug: Enable debugging output and plots
            debug_sample: Specific sample filename to debug (e.g., "0000006_P0_S2.json")
        """
        super().__init__()
        DEFAULT_SUFFIXES_TO_STRIP = [
            '_clean', '_interp', '_filtered', '_fix', '_fixations', 
            '_epochs', '_epoch', 
            '_p0', '_p1', '_p2', '_p3', '_p4', '_p5',  # Handle various subject numbers
            '_p0_clean', '_p0_filtered',  # Handle combined suffixes
            '_session1', '_session2',  # Handle session numbers
            '_run1', '_run2'  # Handle run numbers
        ]
        self.data_dir = data_dir
        self.target_length = target_length
        self.eeg_sampling_rate = eeg_sampling_rate
        self.debug = debug
        self.debug_sample = debug_sample  # Store the specific sample to debug
        self.suffixes_to_strip = kwargs.get('suffixes_to_strip', DEFAULT_SUFFIXES_TO_STRIP)
        # Calculate EEG duration in seconds
        self.eeg_duration_seconds = target_length / eeg_sampling_rate if target_length else None
        
        if self.debug:
            print(f"\n=== INITIALIZING EEGGazeFixationDataset ===")
            print(f"Target length: {target_length} samples")
            print(f"EEG sampling rate: {eeg_sampling_rate} Hz")
            print(f"EEG duration: {self.eeg_duration_seconds:.1f} seconds ({self.eeg_duration_seconds/60:.1f} minutes)")
            if self.debug_sample:
                print(f"DEBUG MODE: Will trace sample: {self.debug_sample}")
        
        # Default channel mapping (from your pipeline)
        if channel_mapping is None:
            self.channel_mapping = self._create_nmt_channel_mapping()
        else:
            self.channel_mapping = channel_mapping
        self.channel_mapping = [ch.upper() for ch in self.channel_mapping]
        self.channel_to_index = {ch: idx for idx, ch in enumerate(self.channel_mapping)}
        
        # Load data.csv to get file list and labels
        # csv_file = "/kaggle/input/results1/results_1/data1/data_processed1/results0/data_1.csv"
        csv_file = os.path.join(data_dir, 'data.csv')
        if os.path.exists(csv_file):
            import pandas as pd
            self.df = pd.read_csv(csv_file)

        else:
            # Fallback: list all .npz files
            self.df = None
            print("Warning: data.csv not found, using .npz files directly")
        
        # Get list of EEG files
        self.eeg_files = self._get_eeg_files()
        
        # Apply index filtering
        if indexes is not None:
            self.eeg_files = [self.eeg_files[i] for i in indexes]
            if self.debug:
                print(f"Using {len(self.eeg_files)} files after index filtering")
        
        # Find gaze JSON files
        gaze_dir = gaze_json_dir if gaze_json_dir else data_dir
        self.gaze_files = sorted(glob.glob(os.path.join(gaze_dir, gaze_json_pattern)))
        
        if self.debug:
            print(f"Found {len(self.gaze_files)} gaze JSON files")
            if self.debug_sample:
                debug_path = os.path.join(gaze_dir, self.debug_sample)
                if os.path.exists(debug_path):
                    print(f"✓ Debug sample found: {debug_path}")
                else:
                    print(f"✗ Debug sample NOT found: {debug_path}")
                    # Try to find similar files
                    similar = [f for f in self.gaze_files if self.debug_sample in f]
                    if similar:
                        print(f"  Similar files found: {[os.path.basename(f) for f in similar[:3]]}")
        
        # Create mapping from EEG files to gaze files
        self.eeg_to_gaze = self._create_file_mappings()
        
        # Preload gaze data
        self.gaze_cache = {}
        self._preload_gaze_data()
    
    def _create_nmt_channel_mapping(self) -> List[str]:
        """Create channel mapping for NMT dataset (from your pipeline)"""
        return [
            "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
            "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
            "FZ", "CZ", "PZ", "A1", "A2", "EKG"
        ]
    
    def _get_eeg_files(self) -> List[str]:
        """Get list of EEG .npz files"""
        files = []
        
        if self.df is not None and 'File' in self.df.columns:
            for idx, csv_file in enumerate(self.df['File']):
                file_path = str(csv_file).strip()
                
                if os.path.isabs(file_path):
                    if os.path.exists(file_path):
                        files.append(file_path)
                    else:
                        filename = os.path.basename(file_path)
                        alt_path = os.path.join(self.data_dir, filename)
                        if os.path.exists(alt_path):
                            files.append(alt_path)
                else:
                    rel_path = os.path.join(self.data_dir, file_path)
                    if os.path.exists(rel_path):
                        files.append(rel_path)
                    else:
                        filename = os.path.basename(file_path)
                        alt_path = os.path.join(self.data_dir, filename)
                        if os.path.exists(alt_path):
                            files.append(alt_path)
        
        # Fallback: list all .npz files in directory
        if not files:
            npz_files = sorted(glob.glob(os.path.join(self.data_dir, '*.npz')))
            files = npz_files
        
        return files
    
    def normalize_basename(self, path_or_name):
        """Use the SAME normalization as FilteredEEGGazeFixationDataset"""
        if path_or_name is None:
            return None

        # If bytes, decode
        if isinstance(path_or_name, (bytes, bytearray)):
            try:
                path_or_name = path_or_name.decode('utf-8', errors='ignore')
            except:
                path_or_name = str(path_or_name)

        # Get base name without extension, lowercase
        base = os.path.splitext(os.path.basename(str(path_or_name)))[0]
        b = base.lower()
        
        # Strip specified suffixes
        if self.suffixes_to_strip:
            # Sort by length (longest first) to handle complex suffixes
            sorted_suffixes = sorted(self.suffixes_to_strip, key=len, reverse=True)
            for s in sorted_suffixes:
                s_l = s.lower()
                if b.endswith(s_l):
                    b = b[: -len(s_l)]
                    break
        
        return b
    
    def _create_file_mappings(self) -> Dict[str, Optional[str]]:
        """Map EEG files to corresponding gaze JSON files - FIXED"""
        mappings = {}
        
        # Build normalized maps
        eeg_map = {}
        for eeg_file in self.eeg_files:
            nb = self.normalize_basename(eeg_file)
            eeg_map[nb] = eeg_file
        
        gaze_map = {}
        for gaze_file in self.gaze_files:
            nb = self.normalize_basename(gaze_file)
            gaze_map[nb] = gaze_file
        
        # Match using normalized names
        for eeg_nb, eeg_file in eeg_map.items():
            gaze_file = gaze_map.get(eeg_nb)
            mappings[eeg_file] = gaze_file
            
            if self.debug and self.debug_sample:
                # Debug specific sample
                if self.debug_sample in eeg_file or self.debug_sample in str(gaze_file):
                    print(f"\nDEBUG: File matching for {self.debug_sample}")
                    print(f"  EEG: {os.path.basename(eeg_file)} → normalized: {eeg_nb}")
                    if gaze_file:
                        print(f"  Gaze: {os.path.basename(gaze_file)} → normalized: {self.normalize_basename(gaze_file)}")
                        print(f"  Match: {eeg_nb == self.normalize_basename(gaze_file)}")
                    else:
                        print(f"  No gaze match found")
                        print(f"  Available gaze normalized names: {list(gaze_map.keys())[:5]}...")
        
        matches = sum(1 for g in mappings.values() if g is not None)
        
        if self.debug:
            print(f"Matched {matches}/{len(mappings)} EEG files with gaze data")
        
        return mappings
    
    def _preload_gaze_data(self):
        """Preload all gaze JSON files into memory"""
        for gaze_file in self.gaze_files:
            try:
                # CHECK IF THIS IS THE DEBUG SAMPLE
                is_debug_sample = self.debug_sample and self.debug_sample in gaze_file
                
                if is_debug_sample and self.debug:
                    print(f"\n{'='*80}")
                    print(f"DEBUG: PRELOADING GAZE FILE: {os.path.basename(gaze_file)}")
                    print(f"Full path: {gaze_file}")
                    print(f"{'='*80}")
                
                with open(gaze_file, 'r') as f:
                    gaze_json = json.load(f)
                
                all_fixations = []
                
                # ========== FIX 1: Handle different JSON formats ==========
                fixations_list = []
                
                # Format 1: Direct 'fixations' key (from your debug output)
                if 'fixations' in gaze_json and isinstance(gaze_json['fixations'], list):
                    fixations_list = gaze_json['fixations']
                    if is_debug_sample and self.debug:
                        print(f"✓ Found 'fixations' key with {len(fixations_list)} fixations")
                
                # Format 2: Nested in sessions (your original assumption)
                elif 'sessions' in gaze_json:
                    for session in gaze_json['sessions']:
                        if 'fixations' in session and isinstance(session['fixations'], list):
                            fixations_list.extend(session['fixations'])
                    if is_debug_sample and self.debug:
                        print(f"✓ Found 'sessions->fixations' with {len(fixations_list)} fixations")
                
                # Format 3: Try to find any list that looks like fixations
                else:
                    if is_debug_sample and self.debug:
                        print(f"✗ No 'fixations' or 'sessions' key found. Searching...")
                    for key, value in gaze_json.items():
                        if isinstance(value, list) and len(value) > 0:
                            first_item = value[0]
                            if isinstance(first_item, dict) and 'x' in first_item and 'y' in first_item:
                                fixations_list = value
                                if is_debug_sample and self.debug:
                                    print(f"✓ Found fixations in key '{key}' with {len(fixations_list)} fixations")
                                break
                
                if is_debug_sample and self.debug and not fixations_list:
                    print(f"✗ No fixations found in JSON file!")
                    print(f"Available keys: {list(gaze_json.keys())}")
                
                # ========== FIX 2: Create Fixation objects ==========
                for fix_idx, fix_dict in enumerate(fixations_list):
                    # Handle 'channel' vs 'channels' field
                    channels = []
                    if 'channel' in fix_dict and fix_dict['channel']:
                        if isinstance(fix_dict['channel'], list):
                            channels = fix_dict['channel']
                        elif isinstance(fix_dict['channel'], str):
                            channels = [fix_dict['channel']]
                    elif 'channels' in fix_dict:
                        if isinstance(fix_dict['channels'], list):
                            channels = fix_dict['channels']
                    
                    # DEBUG: Show first few fixations
                    if is_debug_sample and self.debug and fix_idx < 3:
                        print(f"\n  Fixation {fix_idx}:")
                        print(f"    x: {fix_dict.get('x', 'MISSING')}")
                        print(f"    y: {fix_dict.get('y', 'MISSING')}")
                        print(f"    duration: {fix_dict.get('duration', 'MISSING')}")
                        print(f"    channels: {channels}")
                        print(f"    num_points: {fix_dict.get('num_points', 'MISSING')}")
                    
                    # Create fixation object
                    fixation = Fixation(
                        start_time=fix_dict.get('start_time', 0),
                        end_time=fix_dict.get('end_time', 0),
                        duration=fix_dict.get('duration', 0),
                        x=fix_dict.get('x', 0),  # EEG time in seconds
                        y=fix_dict.get('y', 0),
                        channels=channels,  # FIXED: Use correct field name
                        num_points=fix_dict.get('num_points', 1)
                    )
                    all_fixations.append(fixation)
                
                self.gaze_cache[gaze_file] = all_fixations
                
                if is_debug_sample and self.debug and all_fixations:
                    print(f"\n✓ Successfully loaded {len(all_fixations)} fixations")
                    x_values = [f.x for f in all_fixations]
                    print(f"  x range: {min(x_values):.1f}s to {max(x_values):.1f}s")
                    print(f"  First 3 x values: {x_values[:3]}")
                    
            except Exception as e:
                print(f"  Error loading {gaze_file}: {e}")
                import traceback
                traceback.print_exc()
                self.gaze_cache[gaze_file] = []
    
    def _load_eeg_from_npz(self, file_path: str) -> np.ndarray:
        """Load EEG from .npz file (compatible with get_datanpz output)"""
        data = np.load(file_path)
        
        # Your pipeline saves EEG as numpy arrays
        eeg_keys = ['eeg', 'data', 'X', 'signal']
        for key in eeg_keys:
            if key in data:
                eeg = data[key]
                break
        else:
            # Look for any array
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    eeg = arr
                    break
            else:
                raise ValueError(f"No EEG data found in {file_path}")
        
        # Ensure shape is (channels, time)
        if eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T
        
        return eeg.astype(np.float32)
    
    def _get_label(self, file_path: str) -> int:
        """Get label for EEG file"""
        if self.df is not None and 'File' in self.df.columns and 'Label' in self.df.columns:
            filename = os.path.basename(file_path)
            
            for idx, row in self.df.iterrows():
                csv_file = str(row['File'])
                
                if (filename in csv_file or
                    os.path.basename(csv_file) == filename or
                    file_path in csv_file or
                    csv_file in file_path):
                    
                    return int(row['Label'])
        
        # Fallback: check .npz file
        try:
            data = np.load(file_path)
            for key in ['label', 'y', 'target']:
                if key in data:
                    return int(data[key])
        except:
            pass
        
        # Default
        return 0
    
    def _filter_fixations_by_eeg_time(self, fixations: List[Fixation]) -> List[Fixation]:
        """
        REMOVED: Filter fixations based on x coordinate (EEG time in seconds)
        Now returns ALL fixations without filtering by EEG duration
        """
        if not fixations:
            return []
        
        # Simply return all fixations without filtering
        # Remove the EEG duration check completely
        
        if self.debug:
            print(f"\n  NOT filtering by EEG time:")
            print(f"    Keeping ALL {len(fixations)} fixations regardless of x value")
            
            x_values = [f.x for f in fixations]
            if x_values:
                print(f"    x range: {min(x_values):.1f}s - {max(x_values):.1f}s")
            
            # Show example
            if fixations:
                print(f"    Example fixation:")
                print(f"      x: {fixations[0].x}s (EEG time)")
                print(f"      start_time: {fixations[0].start_time} (time.time())")
        
        # Return all fixations as-is
        return fixations
    
    def _convert_fixations_to_gaze_map(self, 
                                      fixations: List[Fixation],
                                      eeg_duration: float,
                                      n_channels: int,
                                      debug_context: str = "") -> np.ndarray:
        """
        Convert fixations to 2D gaze map matching EEG shape
        Uses x coordinate as EEG time reference
        """
        n_time = int(eeg_duration * self.eeg_sampling_rate)  # Should be target_length
        
        if self.debug:
            print(f"\n{debug_context}")
            print(f"  Creating gaze map from {len(fixations)} fixations:")
            print(f"    EEG duration: {eeg_duration:.1f}s")
            print(f"    Time samples: {n_time}")
            print(f"    Channels: {n_channels}")
        
        # Create empty gaze map
        gaze_map = np.zeros((n_channels, n_time), dtype=np.float32)
        
        if not fixations:
            if self.debug:
                print("  No fixations, returning baseline attention")
            return gaze_map + 0.01  # Baseline attention
        
        # DEBUG: Show all fixation x values
        if self.debug:
            print(f"  All fixation x values: {[f.x for f in fixations]}")
        
        # Process each fixation
        total_weight_added = 0
        for fixation_idx, fixation in enumerate(fixations):
            # CRITICAL: Use fixation.x as EEG time in seconds
            eeg_time_seconds = fixation.x
            
            # Convert EEG time to sample index
            time_normalized = eeg_time_seconds / eeg_duration  # Should be 0 to 1
            center_idx = int(time_normalized * n_time)
            
            # Clamp to valid range
            center_idx = max(0, min(center_idx, n_time - 1))
            
            # Calculate spread based on fixation duration
            # Longer fixations = wider attention spread
            duration_samples = int(fixation.duration * self.eeg_sampling_rate)
            half_duration = max(1, duration_samples // 2)
            
            # Define attention window
            start_idx = max(0, center_idx - half_duration)
            end_idx = min(n_time, center_idx + half_duration)
            
            # Get channel indices
            channel_indices = []
            for ch_name in fixation.channels:
                if ch_name.upper() in self.channel_to_index:
                    channel_indices.append(self.channel_to_index[ch_name.upper()])
            
            if not channel_indices:
                # If no specific channels, distribute lightly
                channel_indices = list(range(n_channels))
                weight = fixation.attention_weight * 0.3
                if self.debug and fixation_idx < 3:
                    print(f"  Fixation {fixation_idx}: No channel match. Using all channels.")
            else:
                weight = fixation.attention_weight
            
            # Apply Gaussian attention around center_idx
            if end_idx > start_idx:
                center = (start_idx + end_idx) / 2
                sigma = max(1.0, (end_idx - start_idx) / 4.0)
                
                time_indices = np.arange(start_idx, end_idx)
                gaussian = np.exp(-0.5 * ((time_indices - center) / sigma) ** 2)
                
                weight_for_fixation = gaussian.sum() * weight * len(channel_indices)
                total_weight_added += weight_for_fixation
                
                for ch_idx in channel_indices:
                    gaze_map[ch_idx, start_idx:end_idx] += gaussian * weight
            
            if self.debug and fixation_idx < 3:  # Debug first 3 fixations
                print(f"\n    Fixation {fixation_idx}:")
                print(f"      EEG time (x): {eeg_time_seconds:.2f}s")
                print(f"      Normalized time: {time_normalized:.4f} -> idx: {center_idx}")
                print(f"      Duration: {fixation.duration:.2f}s -> {duration_samples} samples")
                print(f"      Channels: {fixation.channels} -> indices: {channel_indices}")
                print(f"      Weight: {weight:.3f}")
                print(f"      Window: {start_idx}-{end_idx} ({end_idx-start_idx} samples)")
                if end_idx > start_idx:
                    print(f"      Gaussian sum: {gaussian.sum():.3f}")
        
        if self.debug:
            print(f"\n  Total weight added to gaze map: {total_weight_added:.6f}")
            print(f"  Gaze map max before normalization: {gaze_map.max():.6f}")
            print(f"  Gaze map min before normalization: {gaze_map.min():.6f}")
        
        # Normalize
        if gaze_map.max() > 0:
            # can use gaze_map.sum() if you want to preserve total attention, but max normalization is simpler
            gaze_map = gaze_map / gaze_map.max()
            if self.debug:
                print(f"  ✓ Normalized gaze map (max was {gaze_map.max():.6f})")
        else:
            if self.debug:
                print(f"  ✗ WARNING: gaze_map.max() = {gaze_map.max():.6f} ≤ 0!")
        
        # Add baseline
        # gaze_map = gaze_map * 0.9 + 0.1
        # gaze_map = gaze_map / (gaze_map.max() + 1e-6)  # Normalize 0-1
        # #apply non linear scaling
        # gaze_map = gaze_map ** 2  # amplify the differencev

        
        if self.debug:
            print(f"\n  Final gaze map statistics:")
            print(f"    Min: {gaze_map.min():.6f}, Max: {gaze_map.max():.6f}")
            print(f"    Mean: {gaze_map.mean():.6f}, Std: {gaze_map.std():.6f}")
            
            # Check if any attention beyond EEG duration
            gaze_temporal = gaze_map.mean(axis=0)
            significant_idx = np.where(gaze_temporal > 0.15)[0]
            if len(significant_idx) > 0:
                max_idx = significant_idx[-1]
                max_time = max_idx / self.eeg_sampling_rate
                print(f"    Last significant gaze at: {max_time:.1f}s")
        
        return gaze_map
    
    def _process_eeg(self, eeg: np.ndarray) -> np.ndarray:
        """Process EEG: normalize and crop/pad to target_length"""
        # Z-score normalize each channel
        eeg_processed = np.zeros_like(eeg)
        for ch in range(eeg.shape[0]):
            channel_data = eeg[ch]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 0:
                eeg_processed[ch] = (channel_data - mean) / std
            else:
                eeg_processed[ch] = channel_data - mean
        
        # Crop or pad to target_length
        if self.target_length is not None:
            current_length = eeg_processed.shape[1]
            
            if self.debug:
                print(f"\n  Processing EEG:")
                print(f"    Original shape: {eeg.shape}")
                print(f"    Target length: {self.target_length}")
                print(f"    Current length: {current_length}")
            
            if current_length > self.target_length:
                # CROP: Take first target_length samples (FIRST 5 MINUTES)
                eeg_processed = eeg_processed[:, :self.target_length]
                if self.debug:
                    print(f"    Cropped to first {self.target_length} samples")
            elif current_length < self.target_length:
                # PAD
                pad_width = self.target_length - current_length
                eeg_processed = np.pad(
                    eeg_processed, 
                    ((0, 0), (0, pad_width)), 
                    mode='edge'
                )
                if self.debug:
                    print(f"    Padded with {pad_width} samples")
        
        return eeg_processed
    
    def __len__(self) -> int:
        return len(self.eeg_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample with PROPER TIME ALIGNMENT"""
        if self.debug:
            print(f"\n{'='*80}")
            print(f"LOADING SAMPLE {idx}")
            print(f"{'='*80}")
        
        # Get EEG file
        eeg_file = self.eeg_files[idx]
        
        # ========== EXTRACT SEGMENT OFFSET ==========
        import re
        basename = os.path.basename(eeg_file)
        segment_num = 0
        if '_S' in basename:
            match = re.search(r'_S(\d+)', basename)
            if match:
                segment_num = int(match.group(1))
        segment_offset = segment_num * 120  # 0, 300, 600, ... seconds
        
        if self.debug:
            print(f"\n1. EEG File: {basename}")
            print(f"   Segment number: {segment_num}, Offset: {segment_offset}s")
        
        # CHECK IF THIS IS THE DEBUG SAMPLE'S MATCHING EEG FILE
        is_debug_related = False
        if self.debug_sample:
            # Try to find matching EEG file for debug sample
            debug_base = self.debug_sample.replace('.json', '')
            if debug_base in basename:
                is_debug_related = True
                print(f"\n{'='*80}")
                print(f"DEBUG: FOUND MATCHING EEG FILE FOR {self.debug_sample}")
                print(f"EEG: {basename} -> JSON: {self.debug_sample}")
                print(f"{'='*80}")
        
        # Load EEG
        eeg_raw = self._load_eeg_from_npz(eeg_file)
        
        if self.debug:
            print(f"\n2. Raw EEG:")
            print(f"   Shape: {eeg_raw.shape}")
            print(f"   Duration: {eeg_raw.shape[1]/self.eeg_sampling_rate:.1f}s")
        
        # Get label
        label = self._get_label(eeg_file)
        
        # Process EEG to target length (FIRST 5 MINUTES)
        eeg = self._process_eeg(eeg_raw)
        
        # Calculate actual EEG duration (should match target)
        eeg_duration = eeg.shape[1] / self.eeg_sampling_rate
        
        if self.debug:
            print(f"\n3. Processed EEG:")
            print(f"   Shape: {eeg.shape}")
            print(f"   Duration: {eeg_duration:.1f}s")
            print(f"   Label: {label}")
        
        # Get gaze fixations
        gaze_file = self.eeg_to_gaze.get(eeg_file)
        all_fixations = self.gaze_cache.get(gaze_file, []) if gaze_file else []
        
        if self.debug:
            print(f"\n4. Gaze Data:")
            print(f"   Gaze file: {os.path.basename(gaze_file) if gaze_file else 'None'}")
            print(f"   All fixations: {len(all_fixations)}")
        
        # ========== CONVERT ABSOLUTE TIMES TO SEGMENT-RELATIVE ==========
        if self.debug and all_fixations:
            print(f"\n5. Timestamp Conversion (Segment {segment_num}):")
            print(f"   Segment offset: {segment_offset}s")
            print(f"   EEG duration: {self.eeg_duration_seconds}s")
        
        segment_fixations = []
        for fixation_idx, fixation in enumerate(all_fixations):
            # Convert absolute time to segment-relative time
            relative_x = fixation.x - segment_offset
            
            # Only keep fixations within this segment (0-300s)
            if 0 <= relative_x <= self.eeg_duration_seconds:
                # Create new fixation with relative time
                new_fix = Fixation(
                    start_time=fixation.start_time,
                    end_time=fixation.end_time,
                    duration=fixation.duration,
                    x=relative_x,  # Now 0-300s for this segment
                    y=fixation.y,
                    channels=fixation.channels,
                    num_points=fixation.num_points
                )
                segment_fixations.append(new_fix)
                
                if self.debug and fixation_idx < 3:
                    print(f"   Fixation {fixation_idx}:")
                    print(f"     Absolute x: {fixation.x:.2f}s")
                    print(f"     Relative x: {relative_x:.2f}s")
                    print(f"     In segment: ✓")
        
        if self.debug:
            print(f"\n6. Conversion Result:")
            print(f"   Converted {len(segment_fixations)}/{len(all_fixations)} fixations")
            if segment_fixations:
                x_values = [f.x for f in segment_fixations]
                print(f"   Relative x range: {min(x_values):.1f}s - {max(x_values):.1f}s")
        
        # Use segment-relative fixations for gaze map
        filtered_fixations = segment_fixations
        
        # Convert filtered fixations to gaze map
        debug_context = ""
        if is_debug_related:
            debug_context = f"\n{'='*80}\nDEBUG: CREATING GAZE MAP FOR {self.debug_sample}\n{'='*80}"
        
        gaze_map = self._convert_fixations_to_gaze_map(
            fixations=filtered_fixations,
            eeg_duration=eeg_duration,
            n_channels=eeg.shape[0],
            debug_context=debug_context
        )
        
        # ========== DEBUG: VISUALIZE ALIGNMENT ==========
        if self.debug and idx == 0:  # Only for first sample
            self._debug_visualize_alignment(eeg, gaze_map, eeg_duration, filtered_fixations)
        
        # Resize if needed (shouldn't be needed if we did everything right)
        if gaze_map.shape[1] != eeg.shape[1]:
            if self.debug:
                print(f"  WARNING: Resizing gaze map from {gaze_map.shape[1]} to {eeg.shape[1]}")
            
            original_times = np.linspace(0, 1, gaze_map.shape[1])
            new_times = np.linspace(0, 1, eeg.shape[1])
            
            gaze_resized = np.zeros((gaze_map.shape[0], eeg.shape[1]))
            for ch in range(gaze_map.shape[0]):
                interp_func = interpolate.interp1d(
                    original_times, 
                    gaze_map[ch], 
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                gaze_resized[ch] = interp_func(new_times)
            
            gaze_map = gaze_resized
        
        # Convert to tensors
        eeg_tensor = torch.FloatTensor(eeg)
        gaze_tensor = torch.FloatTensor(gaze_map)
        label_tensor = torch.LongTensor([label]).squeeze()
        
        if self.debug and is_debug_related:
            print(f"\n{'='*80}")
            print(f"DEBUG: FINAL OUTPUT FOR {self.debug_sample}")
            print(f"{'='*80}")
            print(f"EEG shape: {eeg.shape}")
            print(f"Gaze shape: {gaze_map.shape}")
            print(f"Gaze min: {gaze_map.min():.6f}, max: {gaze_map.max():.6f}")
            print(f"Gaze mean: {gaze_map.mean():.6f}, std: {gaze_map.std():.6f}")
            print(f"Number of fixations used: {len(filtered_fixations)}")
            print(f"{'='*80}")
        
        return {
            'eeg': eeg_tensor,
            'gaze': gaze_tensor,
            'label': label,
            'file': eeg_file,
            'num_fixations': len(filtered_fixations),
            'eeg_duration': eeg_duration
        }
    
    def _debug_visualize_alignment(self, eeg: np.ndarray, gaze_map: np.ndarray, 
                                  eeg_duration: float, fixations: List[Fixation]):
        """Create debug visualization for time alignment"""
        os.makedirs('debug_plots', exist_ok=True)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. Plot EEG signal (channel average)
        time_axis = np.arange(eeg.shape[1]) / self.eeg_sampling_rate
        eeg_avg = eeg.mean(axis=0)
        
        axes[0].plot(time_axis, eeg_avg, 'b-', linewidth=0.5, alpha=0.7)
        axes[0].set_title(f'EEG Signal (Channel Average) - Duration: {eeg_duration:.1f}s')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Amplitude (normalized)')
        axes[0].axvline(x=eeg_duration, color='r', linestyle='--', 
                       label=f'EEG end ({eeg_duration:.1f}s)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Plot gaze attention over time
        gaze_temporal = gaze_map.mean(axis=0)
        axes[1].plot(time_axis, gaze_temporal, 'g-', linewidth=2, alpha=0.8)
        axes[1].set_title('Gaze Attention Over Time (from x coordinate)')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Attention Weight')
        axes[1].axvline(x=eeg_duration, color='r', linestyle='--')
        axes[1].set_ylim(0, 1)
        axes[1].fill_between(time_axis, 0, gaze_temporal, alpha=0.3, color='green')
        axes[1].grid(True, alpha=0.3)
        
        # Mark fixation times (x coordinates)
        if fixations:
            fixation_times = [f.x for f in fixations]  # EEG times!
            fixation_weights = [f.attention_weight for f in fixations]
            
            axes[1].scatter(fixation_times, 
                          [0.8] * len(fixation_times),  # Position near top
                          c=fixation_weights, cmap='hot', 
                          s=50, alpha=0.7, edgecolors='black',
                          label=f'Fixations ({len(fixation_times)})')
            axes[1].legend()
        
        # 3. Plot heatmap of gaze attention
        im = axes[2].imshow(gaze_map, aspect='auto', cmap='hot',
                           extent=[0, eeg_duration, 0, gaze_map.shape[0]],
                           vmin=0, vmax=1)
        axes[2].set_title('Gaze Attention Heatmap (Channels × Time)')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Channel Index')
        axes[2].axvline(x=eeg_duration, color='r', linestyle='--')
        
        # Add channel labels
        channel_labels = self.channel_mapping[:gaze_map.shape[0]]
        axes[2].set_yticks(np.arange(len(channel_labels)))
        axes[2].set_yticklabels(channel_labels, fontsize=8)
        
        plt.colorbar(im, ax=axes[2], label='Attention Weight')
        
        plt.tight_layout()
        plt.savefig(f'debug_plots/time_alignment_sample0.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create a summary text file
        with open('debug_plots/time_alignment_summary.txt', 'w') as f:
            f.write(f"=== EEG-GAZE TIME ALIGNMENT DEBUG ===\n\n")
            f.write(f"EEG Duration: {eeg_duration:.1f} seconds\n")
            f.write(f"EEG Samples: {eeg.shape[1]}\n")
            f.write(f"Gaze Map Shape: {gaze_map.shape}\n")
            f.write(f"Number of Fixations: {len(fixations)}\n\n")
            
            f.write("KEY INSIGHT:\n")
            f.write("  - Using x coordinate as EEG time reference\n")
            f.write("  - Ignoring start_time/end_time (time.time() units)\n")
            f.write("  - x = EEG time in seconds (0-300 for 5-min EEG)\n\n")
            
            f.write("Fixation Details:\n")
            for i, fix in enumerate(fixations[:10]):  # First 10
                f.write(f"  Fixation {i}:\n")
                f.write(f"    EEG time (x): {fix.x:.2f}s\n")
                f.write(f"    start_time: {fix.start_time:.2f} (time.time())\n")
                f.write(f"    Duration: {fix.duration:.2f}s\n")
                f.write(f"    Channels: {fix.channels}\n")
                f.write(f"    Weight: {fix.attention_weight:.3f}\n")
            
            # Check alignment
            max_gaze_idx = np.where(gaze_temporal > 0.1)[0]
            if len(max_gaze_idx) > 0:
                max_gaze_time = max_gaze_idx[-1] / self.eeg_sampling_rate
                f.write(f"\nAlignment Check:\n")
                f.write(f"  Last significant gaze at: {max_gaze_time:.1f}s\n")
                f.write(f"  EEG duration: {eeg_duration:.1f}s\n")
                if max_gaze_time > eeg_duration:
                    f.write(f"  WARNING: Gaze exceeds EEG duration!\n")
                else:
                    f.write(f" OK: Gaze within EEG duration\n")

# Wrapper for backward compatibility
def EEGDataset_with_gaze(data_dir, indexes=None, target_length=None, debug=False, **kwargs):
    """Wrapper that matches your existing EEGDataset interface"""
    return EEGGazeFixationDataset(
        data_dir=data_dir,
        indexes=indexes,
        target_length=target_length,
        debug=debug,
        **kwargs
    )