"""
SCNet with gaze integration at BOTH input and output levels
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.scnet_gaze_output import SCNet_Gaze_Output


class SCNet_Gaze_Combined(SCNet_Gaze_Output):
    """
    SCNet with gaze integration at both input and output levels
    Combines input modulation with output attention
    """
    def __init__(self, n_chan: int = 22, n_outputs: int = 2, original_time_length: int = 6000):
        super(SCNet_Gaze_Combined, self).__init__(n_chan, n_outputs, original_time_length)
        
        # Input gaze modulation parameter
        self.gaze_alpha = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, eeg, gaze=None, return_attention=False):
        """
        Forward pass with both input and output gaze integration
        
        Args:
            eeg: [B, n_chan, T] - EEG signals
            gaze: [B, n_chan, T] or None - Gaze attention maps
            return_attention: Whether to return attention map
            
        Returns:
            If return_attention=False: logits only
            If return_attention=True: dict with 'logits' and 'attention_map'
        """
        # ========== INPUT INTEGRATION ==========
        if gaze is not None:
            # Ensure same shape
            if eeg.shape != gaze.shape:
                gaze = F.interpolate(gaze, size=eeg.shape[-1], mode='linear', align_corners=False)
            
            # Input modulation
            eeg = eeg * (1.0 + self.gaze_alpha * gaze)
        
        # ========== OUTPUT INTEGRATION (via parent) ==========
        # Pass through parent class which handles attention
        return super().forward(eeg, return_attention=return_attention)
    
    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'model': 'SCNet_Gaze_Combined',
            'gaze_integration': 'both',
            'gaze_alpha': self.gaze_alpha.item()
        })
        return config