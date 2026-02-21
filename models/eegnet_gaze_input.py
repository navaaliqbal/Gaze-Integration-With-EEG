"""
EEGNet with gaze integration at INPUT level
Gaze modulates EEG before feature extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.eegnet_base import EEGNet_Base


class EEGNet_Gaze_Input(EEGNet_Base):
    """
    EEGNet with gaze integration at input level
    Gaze acts as a gate: eeg * (1 + alpha * gaze)
    """
    def __init__(self, num_input=1, num_class=2, channel=22, signal_length=6000,
                 fs=200, F1=8, D=3, dropout_rate=0.2):
        super(EEGNet_Gaze_Input, self).__init__(
            num_input, num_class, channel, signal_length,
            fs, F1, D, dropout_rate
        )
        
        # Learnable gaze strength parameter
        self.gaze_alpha = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, eeg, gaze=None):
        """
        Forward pass with input gaze integration
        
        Args:
            eeg: (batch, channel, time) - EEG signals
            gaze: (batch, channel, time) or None - Gaze attention maps
            
        Returns:
            logits: (batch, num_class) - Classification probabilities
        """
        # Apply gaze modulation if available
        if gaze is not None:
            # Ensure same shape
            if eeg.shape != gaze.shape:
                # Interpolate gaze to match EEG time dimension
                gaze = F.interpolate(
                    gaze.unsqueeze(1), 
                    size=eeg.shape[-1], 
                    mode='linear', 
                    align_corners=False
                ).squeeze(1)
            
            # Gate mechanism: modulate EEG with gaze
            eeg = eeg * (1.0 + self.gaze_alpha * gaze)
        
        # Add channel dimension for Conv2d
        x = eeg.unsqueeze(1)
        
        # Forward through base EEGNet
        x = self.bn1(self.conv2d(x))
        x = self.bn2(self.depthwise_conv(x))
        x = self.elu(x)
        x = self.dropout(self.avg_pool1(x))
        x = self.sep_conv_depth(x)
        x = self.bn3(self.sep_conv_point(x))
        x = self.elu(x)
        x = self.dropout(self.avg_pool2(x))
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        
        return x
    
    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'model': 'EEGNet_Gaze_Input',
            'gaze_integration': 'input',
            'gaze_alpha': self.gaze_alpha.item()
        })
        return config