"""
SCNet with gaze integration at INPUT level
Gaze modulates EEG before feature extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.scnet_base import SCNet_Base, MFFMBlock, SILM


class SCNet_Gaze_Input(SCNet_Base):
    """
    SCNet with gaze integration at input level
    Gaze acts as a gate: eeg * (1 + alpha * gaze)
    """
    def __init__(self, n_chan: int = 22, n_outputs: int = 2, original_time_length: int = 6000):
        super(SCNet_Gaze_Input, self).__init__(n_chan, n_outputs, original_time_length)
        
        # Learnable gaze strength parameter
        self.gaze_alpha = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, eeg, gaze=None):
        """
        Forward pass with input gaze integration
        
        Args:
            eeg: [B, n_chan, T] - EEG signals
            gaze: [B, n_chan, T] or None - Gaze attention maps
            
        Returns:
            logits: [B, n_outputs] - Classification logits
        """
        # Apply gaze modulation if available
        if gaze is not None:
            # Ensure same shape
            if eeg.shape != gaze.shape:
                # Interpolate gaze to match EEG time dimension
                gaze = F.interpolate(gaze, size=eeg.shape[-1], mode='linear', align_corners=False)
            
            # Gate mechanism: modulate EEG with gaze
            eeg = eeg * (1.0 + self.gaze_alpha * gaze)
        
        # Forward through base SCNet
        return super().forward(eeg)
    
    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'model': 'SCNet_Gaze_Input',
            'gaze_integration': 'input',
            'gaze_alpha': self.gaze_alpha.item()
        })
        return config