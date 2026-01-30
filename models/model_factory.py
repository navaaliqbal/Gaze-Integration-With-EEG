"""
Factory to create NeuroGATE models with different gaze integration approaches
"""
from models.neurogate_gaze_input import NeuroGATE_Gaze_Input
from models.neurogate_gaze_output import NeuroGATE_Gaze_Output
from models.neurogate_combined import NeuroGATE_Combined

def create_neurogate_model(integration_type='output', n_chan=22, n_outputs=2, original_time_length=15000):
    """
    Create NeuroGATE model with specified gaze integration
    
    Args:
        integration_type: 'input', 'output', or 'both'
        n_chan: Number of EEG channels
        n_outputs: Number of output classes
        original_time_length: Original time length of EEG signals
        
    Returns:
        NeuroGATE model instance
    """
    if integration_type == 'input':
        return NeuroGATE_Gaze_Input(
            n_chan=n_chan,
            n_outputs=n_outputs,
            original_time_length=original_time_length
        )
    elif integration_type == 'output':
        return NeuroGATE_Gaze_Output(
            n_chan=n_chan,
            n_outputs=n_outputs,
            original_time_length=original_time_length
        )
    elif integration_type == 'both':
        return NeuroGATE_Combined(
            n_chan=n_chan,
            n_outputs=n_outputs,
            original_time_length=original_time_length
        )
    else:
        raise ValueError(f"Unknown integration type: {integration_type}. "
                        f"Must be 'input', 'output', or 'both'.")

def get_model_config(model):
    """Get configuration of a model"""
    if hasattr(model, 'get_config'):
        return model.get_config()
    else:
        return {'gaze_integration': 'unknown'}