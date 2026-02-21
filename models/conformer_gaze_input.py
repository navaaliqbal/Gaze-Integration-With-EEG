"""
Conformer with gaze integration in INPUT (gaze-as-gate)
Modulates EEG signals with gaze attention before feature extraction
"""
import torch
import torch.nn as nn
from EEGConformer.conformer import Conformer


class Conformer_Gaze_Input(nn.Module):
    """
    Conformer with gaze integration at input level
    Uses gaze to modulate the EEG input directly (gaze-as-gate)
    """
    
    def __init__(self, n_chan: int = 22, n_outputs: int = 2, 
                 original_time_length: int = 6000,
                 emb_size: int = 30, depth: int = 4):
        """
        Args:
            n_chan: Number of EEG channels (default: 22)
            n_outputs: Number of output classes (default: 2)
            original_time_length: Expected time length of input (default: 6000)
            emb_size: Transformer embedding dimension (default: 30)
            depth: Number of transformer layers (default: 4)
        """
        super().__init__()
        
        self.n_chan = n_chan
        self.n_outputs = n_outputs
        self.original_time_length = original_time_length
        self.emb_size = emb_size
        self.depth = depth
        
        # 🔑 Learnable gaze strength for input modulation
        # Initialized to higher value (0.5) to ensure gaze has meaningful effect
        # Will be optimized with separate optimizer for better learning
        self.gaze_alpha = nn.Parameter(torch.tensor(0.5))
        
        # Conformer backbone (same as baseline)
        self.conformer = Conformer(
            emb_size=emb_size,
            depth=depth,
            n_classes=n_outputs,
            n_channels=n_chan
        )
    
    def forward(self, eeg, gaze=None):
        """
        Forward pass with input gaze integration
        
        Args:
            eeg: [B, C, T] - EEG signals  
            gaze: [B, C, T] - Gaze attention maps (optional)
            
        Returns:
            If return_attention=False (default for Sequential):
                (features, logits): Tuple from ClassificationHead
            If return_attention=True:
                Would need to be implemented if needed
        """
        # ====================================================
        # INPUT INTEGRATION: GAZE-AS-GATE
        # ====================================================
        # Modulate EEG with gaze before any processing
        # Formula: eeg_modulated = eeg * (1 + alpha * gaze)
        # - When gaze is high, amplifies that channel/time
        # - When gaze is low, reduces that channel/time
        # - alpha controls the strength of modulation
        if gaze is not None:
            eeg = eeg * (1.0 + self.gaze_alpha * gaze)
        
        # ====================================================
        # CONFORMER ARCHITECTURE
        # ====================================================
        # Add channel dimension: (B, C, T) -> (B, 1, C, T)
        if eeg.dim() == 3:
            eeg = eeg.unsqueeze(1)
        
        # Forward through Conformer
        # Returns (features, logits) tuple from ClassificationHead
        output = self.conformer(eeg)
        
        return output
    
    def get_config(self):
        """Get model configuration"""
        return {
            'model': 'Conformer_Gaze_Input',
            'gaze_integration': 'input',
            'n_chan': self.n_chan,
            'n_outputs': self.n_outputs,
            'original_time_length': self.original_time_length,
            'emb_size': self.emb_size,
            'depth': self.depth,
            'gaze_alpha_init': 0.1
        }
