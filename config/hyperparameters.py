"""
Hyperparameter configurations for all three gaze integration approaches
"""
from dataclasses import dataclass
from typing import Literal

@dataclass
class Hyperparameters:
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 25
    accumulation_steps: int = 1
    
    # Model parameters
    n_channels: int = 22
    n_outputs: int = 2
    original_time_length: int = 15000  # 5 minutes at 50 Hz
    
    # Gaze integration type
    gaze_integration: Literal['input', 'output', 'both'] = 'output'
    
    # Gaze attention parameters
    gaze_weight: float = 0.2  # Weight for gaze loss (output/both)
    gaze_loss_type: str = 'cosine'  # 'mse', 'cosine', 'combined'
    use_gaze_loss_scaling: bool = True  # If True, compute scaling factor; if False, use 1.0
    
    # Input integration parameters
    input_gaze_alpha: float = 1.0  # Initial gaze alpha for input integration
    
    # Early stopping
    patience: int = 10
    early_stop_metric: str = 'balanced_acc'
    
    # Data parameters
    target_length: int = 15000  # 5 minutes of EEG at 50Hz
    eeg_sampling_rate: float = 50.0
    seed: int = 42
    
    # Dataset parameters
    suffixes_to_strip = [
        '_clean', '_interp', '_filtered', '_fix', '_fixations', 
        '_epochs', '_epoch', 
        '_p0', '_p1', '_p2', '_p3', '_p4', '_p5',
        '_p0_clean', '_p0_filtered',
        '_session1', '_session2',
        '_run1', '_run2'
    ]

def get_hyp_for_integration(integration_type: str = 'output'):
    """Get hyperparameters for specific integration type"""
    hyp = Hyperparameters()
    
    if integration_type == 'input':
        hyp.gaze_integration = 'input'
        hyp.gaze_weight = 0.0  # No gaze loss for input integration
    elif integration_type == 'output':
        hyp.gaze_integration = 'output'
        hyp.gaze_weight = 0.2
    elif integration_type == 'both':
        hyp.gaze_integration = 'both'
        hyp.gaze_weight = 0.2
    
    return hyp

def get_default_hyp():
    """Return default hyperparameters"""
    return Hyperparameters()