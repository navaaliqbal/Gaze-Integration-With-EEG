# gaze_feature_extractor.py
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def extract_gaze_features_from_fixation_json(fixation_json_path: str) -> Dict[str, float]:
    """
    Extract 4-5 key gaze features from fixation JSON for EEG-level integration.
    
    Returns:
        Dictionary with 4-5 gaze features
    """
    with open(fixation_json_path, "r") as f:
        data = json.load(f)

    all_fixations = []
    for session in data.get("sessions", []):
        all_fixations.extend(session.get("fixations", []))
    
    if not all_fixations:
        # Return default features if no fixations
        return {
            "fixation_count": 0.0,
            "mean_fixation_duration": 0.0,
            "total_fixation_duration": 0.0,
            "saccade_velocity": 0.0,
            "gaze_dispersion": 0.0
        }

    # Convert to DataFrame for easier calculation
    df = pd.DataFrame(all_fixations)
    
    # ==== KEY FEATURES (4-5 total) ====
    
    # 1. Fixation count (normalized)
    fixation_count = len(df)
    
    # 2. Mean fixation duration
    mean_fixation_duration = np.mean(df['duration'].values) if len(df) > 0 else 0.0
    
    # 3. Total fixation duration
    total_fixation_duration = df['duration'].sum() if len(df) > 0 else 0.0
    
    # 4. Saccade velocity (if there are multiple fixations)
    if len(df) > 1:
        x_coords = df['x'].values
        y_coords = df['y'].values
        durations = df['duration'].values
        
        # Calculate inter-fixation distances (saccades)
        saccade_lengths = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
        saccade_times = np.diff(durations[:-1])  # Approximate timing
        
        if len(saccade_times) > 0 and saccade_times.sum() > 0:
            saccade_velocity = saccade_lengths[:len(saccade_times)].sum() / saccade_times.sum()
        else:
            saccade_velocity = 0.0
    else:
        saccade_velocity = 0.0
    
    # 5. Gaze dispersion (spread of fixations)
    if len(df) > 1:
        x_std = np.std(df['x'].values)
        y_std = np.std(df['y'].values)
        gaze_dispersion = np.sqrt(x_std**2 + y_std**2)
    else:
        gaze_dispersion = 0.0
    
    # Normalize features (optional, can be done in model)
    features = {
        "fixation_count": float(fixation_count),
        "mean_fixation_duration": float(mean_fixation_duration),
        "total_fixation_duration": float(total_fixation_duration),
        "saccade_velocity": float(saccade_velocity),
        "gaze_dispersion": float(gaze_dispersion)
    }
    
    # Alternative: Use only 4 features by combining or removing one
    # features = {
    #     "fixation_count": float(fixation_count),
    #     "mean_fixation_duration": float(mean_fixation_duration),
    #     "saccade_velocity": float(saccade_velocity),
    #     "gaze_dispersion": float(gaze_dispersion)
    # }
    
    return features

def normalize_gaze_features(features: Dict[str, float]) -> np.ndarray:
    """
    Normalize gaze features to [-1, 1] range.
    """
    # Define reasonable ranges for each feature (adjust based on your data)
    ranges = {
        "fixation_count": (0, 200),  # Assuming max 200 fixations
        "mean_fixation_duration": (0, 5),  # 0-5 seconds
        "total_fixation_duration": (0, 300),  # 0-300 seconds
        "saccade_velocity": (0, 500),  # 0-500 pixels/sec
        "gaze_dispersion": (0, 1000),  # 0-1000 pixels
    }
    
    normalized = []
    for key in ["fixation_count", "mean_fixation_duration", 
                "total_fixation_duration", "saccade_velocity", "gaze_dispersion"]:
        if key in features:
            val = features[key]
            min_val, max_val = ranges[key]
            # Normalize to [-1, 1]
            if max_val > min_val:
                norm_val = 2 * ((val - min_val) / (max_val - min_val)) - 1
            else:
                norm_val = 0.0
            normalized.append(norm_val)
    
    return np.array(normalized, dtype=np.float32)