"""
Generic trainer that dispatches to specific trainers based on model type
"""
from training.trainer_input import train_epoch_input
from training.trainer_output import train_epoch_output
from training.trainer_combined import train_epoch_combined
from training.trainer_scnet import train_epoch_scnet

def train_epoch(model, train_loader, optimizer, device, **kwargs):
    """
    Generic training epoch that routes to appropriate trainer
    
    Args:
        model: Model instance
        **kwargs: Trainer-specific arguments
        
    Returns:
        Training statistics
    """
    # Determine model type
    model_name = model.__class__.__name__.lower()
    
    # Check for SCNet models first
    if 'scnet' in model_name:
        # All SCNet models use the SCNet trainer
        return train_epoch_scnet(model, train_loader, optimizer, device, **kwargs)
    
    # Then check NeuroGATE models
    elif 'input' in model_name:
        return train_epoch_input(model, train_loader, optimizer, device, **kwargs)
    elif 'output' in model_name:
        return train_epoch_output(model, train_loader, optimizer, device, **kwargs)
    elif 'combined' in model_name:
        return train_epoch_combined(model, train_loader, optimizer, device, **kwargs)
    else:
        # Default to output integration
        print(f"Warning: Unknown model type {model_name}, using output trainer")
        return train_epoch_output(model, train_loader, optimizer, device, **kwargs)