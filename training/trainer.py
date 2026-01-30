"""
Generic trainer that dispatches to specific trainers based on model type
"""
from training.trainer_input import train_epoch_input
from training.trainer_output import train_epoch_output
from training.trainer_combined import train_epoch_combined

def train_epoch(model, train_loader, optimizer, device, **kwargs):
    """
    Generic training epoch that routes to appropriate trainer
    
    Args:
        model: NeuroGATE model instance
        **kwargs: Trainer-specific arguments
        
    Returns:
        Training statistics
    """
    # Determine model type
    model_name = model.__class__.__name__.lower()
    
    if 'input' in model_name:
        return train_epoch_input(model, train_loader, optimizer, device, **kwargs)
    elif 'output' in model_name:
        return train_epoch_output(model, train_loader, optimizer, device, **kwargs)
    elif 'combined' in model_name:
        return train_epoch_combined(model, train_loader, optimizer, device, **kwargs)
    else:
        # Default to output integration
        return train_epoch_output(model, train_loader, optimizer, device, **kwargs)