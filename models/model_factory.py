"""
Factory to create all SCNet variants
"""
from models.scnet_base import SCNet_Base
from models.scnet_gaze_input import SCNet_Gaze_Input
from models.scnet_gaze_output import SCNet_Gaze_Output
from models.scnet_gaze_combined import SCNet_Gaze_Combined
"""
Factory to create all EEGNet variants
"""
from models.eegnet_base import EEGNet_Base
from models.eegnet_gaze_input import EEGNet_Gaze_Input
from models.eegnet_gaze_output import EEGNet_Gaze_Output
from models.eegnet_gaze_combined import EEGNet_Gaze_Combined


def create_eegnet_model(integration_type='none', **kwargs):
    """
    Create EEGNet model with specified gaze integration
    
    Args:
        integration_type: 'none', 'input', 'output', or 'both'
        **kwargs: Model parameters (num_input, num_class, channel, signal_length,
                  fs, F1, D, dropout_rate)
    
    Returns:
        EEGNet model instance
    """
    if integration_type == 'none':
        return EEGNet_Base(**kwargs)
    elif integration_type == 'input':
        return EEGNet_Gaze_Input(**kwargs)
    elif integration_type == 'output':
        return EEGNet_Gaze_Output(**kwargs)
    elif integration_type == 'both':
        return EEGNet_Gaze_Combined(**kwargs)
    else:
        raise ValueError(f"Unknown integration type: {integration_type}")




def create_scnet_model(integration_type='none', **kwargs):
    """
    Create SCNet model with specified gaze integration
    
    Args:
        integration_type: 'none', 'input', 'output', or 'both'
        **kwargs: Model parameters (n_chan, n_outputs, original_time_length)
    
    Returns:
        SCNet model instance
    """
    if integration_type == 'none':
        return SCNet_Base(**kwargs)
    elif integration_type == 'input':
        return SCNet_Gaze_Input(**kwargs)
    elif integration_type == 'output':
        return SCNet_Gaze_Output(**kwargs)
    elif integration_type == 'both':
        return SCNet_Gaze_Combined(**kwargs)
    else:
        raise ValueError(f"Unknown integration type: {integration_type}")


def get_model_config(model):
    """Get configuration of a model"""
    if hasattr(model, 'get_config'):
        return model.get_config()
    else:
        return {'model': 'unknown', 'gaze_integration': 'unknown'}