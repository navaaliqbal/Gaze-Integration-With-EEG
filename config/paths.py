"""
Configuration for data paths and directories
"""
import os
from pathlib import Path

class PathConfig:
    # Main data directories
    DATA_DIR = r"C:\Users\S.S.T\Documents\VsCode\eeg models\results\data\data_processed\results1"
    GAZE_JSON_DIR = r"C:\Users\S.S.T\Documents\VsCode\eeg models\results\gaze"
    
    # Subdirectories
    TRAIN_SUBDIR = 'train'
    EVAL_SUBDIR = 'eval'
    
    # CSV file
    DATA_CSV = os.path.join(DATA_DIR, 'data.csv')
    
    # Output directories
    TRAINING_STATS_DIR = 'training_statistics'
    DEBUG_PLOTS_DIR = 'debug_plots'
    MODEL_CHECKPOINTS_DIR = 'checkpoints'
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.TRAINING_STATS_DIR,
            cls.DEBUG_PLOTS_DIR,
            cls.MODEL_CHECKPOINTS_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)