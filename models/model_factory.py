"""
Model Factory - Create all model variants with/without gaze integration

Supported Models:
- EEGNet: create_eegnet_model()
- Conformer: create_conformer_model()  
- SCNet: create_scnet_model()

Integration Types:
- 'none': Baseline model without gaze
- 'input': Gaze modulates input signals (gaze-as-gate)
- 'output': Gaze supervises attention/features at output
- 'both': Combined input and output integration (where available)

Usage:
    from models.model_factory import create_conformer_model
    
    # Baseline Conformer
    model = create_conformer_model(integration_type='none', n_chan=22, n_outputs=2)
    
    # Conformer with input gaze integration
    model = create_conformer_model(integration_type='input', n_chan=22, n_outputs=2)
"""

# SCNet variants
from models.scnet_base import SCNet_Base
from models.scnet_gaze_input import SCNet_Gaze_Input
from models.scnet_gaze_output import SCNet_Gaze_Output
from models.scnet_gaze_combined import SCNet_Gaze_Combined

# EEGNet variants
from models.eegnet_base import EEGNet_Base
from models.eegnet_gaze_input import EEGNet_Gaze_Input
from models.eegnet_gaze_output import EEGNet_Gaze_Output
from models.eegnet_gaze_combined import EEGNet_Gaze_Combined

# Conformer variants
from EEGConformer.conformer import Conformer
from models.conformer_gaze_input import Conformer_Gaze_Input
from models.conformer_gaze_output_v2 import ConformerGazeOutputV2
from models.conformer_combined import Conformer_Combined


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


def create_conformer_model(integration_type='none', **kwargs):
    """
    Create Conformer model with specified gaze integration
    
    Args:
        integration_type: 'none', 'input', 'output', or 'both'
        **kwargs: Model parameters (n_chan, n_outputs, original_time_length, 
                  emb_size, depth)
    
    Returns:
        Conformer model instance
    """
    if integration_type == 'none':
        # Base Conformer expects different parameter names
        # Map from common interface to Conformer's expected params
        n_channels = kwargs.pop('n_chan', kwargs.pop('n_channels', 22))
        n_classes = kwargs.pop('n_outputs', kwargs.pop('n_classes', 2))
        emb_size = kwargs.pop('emb_size', 40)
        depth = kwargs.pop('depth', 4)
        
        return Conformer(
            emb_size=emb_size,
            depth=depth,
            n_classes=n_classes,
            n_channels=n_channels,
            **kwargs
        )
    elif integration_type == 'input':
        return Conformer_Gaze_Input(**kwargs)
    elif integration_type == 'output':
        return ConformerGazeOutputV2(**kwargs)
    elif integration_type == 'both':
        return Conformer_Combined(**kwargs)
    else:
        raise ValueError(f"Unknown integration type: {integration_type}")




def create_scnet_model(integration_type='none', **kwargs):
    """
    Create SCNet model with specified gaze integration
    
    Args:
        integration_type: 'none', 'input', 'output', or 'both'
        **kwargs: Model parameters
    
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