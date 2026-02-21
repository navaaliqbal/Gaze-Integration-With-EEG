"""
Configuration for data paths and directories
"""
import os
from pathlib import Path

class PathConfig:
    #Main data directories
    # DATA_DIR = r"C:\Users\S.S.T\Documents\VsCode\eeg models\results\data\data_processed\results1"
    # GAZE_JSON_DIR = r"C:\Users\S.S.T\Documents\VsCode\eeg models\results\gaze"
    # TRAIN_DIR = r"C:\Users\S.S.T\Documents\VsCode\eeg models\results\data\data_processed\results1\train"
    # EVAL_DIR = r"C:\Users\S.S.T\Documents\VsCode\eeg models\results\data\data_processed\results1\eval"
    DATA_DIR = "/kaggle/input/datasets/ayeshasiddiqa19104/result60-2mins/results/data/data_processed/results0"
    GAZE_JSON_DIR = "/kaggle/input/datasets/ayeshasiddiqa19104/fixations-60-output-segmented/fixations_60_output_segmented"
    TRAIN_DIR = "/kaggle/input/datasets/ayeshasiddiqa19104/npzfiles1/results/data/data_processed/results0/train"

    EVAL_DIR = "/kaggle/input/datasets/ayeshasiddiqa19104/npzfiles1/results/data/data_processed/results0/eval"

    TRAIN_SUBDIR = 'train'
    EVAL_SUBDIR = 'eval'
    # TRAIN_SUBDIR = 'train1'
    # EVAL_SUBDIR = 'eval1'
    
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