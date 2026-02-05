"""
Factory to create NeuroGATE and SCNet models with different gaze integration approaches
"""
from models.neurogate_gaze_input import NeuroGATE_Gaze_Input
from models.neurogate_gaze_output import NeuroGATE_Gaze_Output
from models.neurogate_combined import NeuroGATE_Combined
from models.scnet_gaze_output import SCNet_Gaze_Output
from models.scnet_gaze_input import SCNet_Gaze_Input

def create_model(model_type='neurogate', integration_type='output', **kwargs):
    """
    Create model based on type and integration
    
    Args:
        model_type: 'neurogate' or 'scnet'
        integration_type: 'input', 'output', or 'both'
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    if model_type == 'neurogate':
        if integration_type == 'input':
            return NeuroGATE_Gaze_Input(**kwargs)
        elif integration_type == 'output':
            return NeuroGATE_Gaze_Output(**kwargs)
        elif integration_type == 'both':
            return NeuroGATE_Combined(**kwargs)
        else:
            raise ValueError(f"Unknown integration type for NeuroGATE: {integration_type}")
    
    elif model_type == 'scnet':
        if integration_type == 'output':
            return SCNet_Gaze_Output(**kwargs)
        elif integration_type == 'input':
            return SCNet_Gaze_Input(**kwargs)
       
        # elif integration_type == 'both':
        #     from models.scnet_gaze_both import SCNet_Gaze_Both
        #     return SCNet_Gaze_Both(**kwargs)
        else:
            raise ValueError(f"Unknown integration type for SCNet: {integration_type}")
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'neurogate' or 'scnet'")

# For backward compatibility
def create_neurogate_model(integration_type='output', **kwargs):
    """Legacy function for backward compatibility"""
    return create_model('neurogate', integration_type, **kwargs)

def create_scnet_model(integration_type='output', **kwargs):
    """Create SCNet model"""
    return create_model('scnet', integration_type, **kwargs)

def get_model_config(model):
    """Get configuration of a model"""
    if hasattr(model, 'get_config'):
        return model.get_config()
    else:
        return {'model_type': 'unknown', 'gaze_integration': 'unknown'}