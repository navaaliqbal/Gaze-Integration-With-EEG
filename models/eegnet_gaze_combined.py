"""
EEGNet with gaze integration at BOTH input and output levels
Combines input modulation with output attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.eegnet_gaze_output import EEGNet_Gaze_Output


class EEGNet_Gaze_Combined(EEGNet_Gaze_Output):
    """
    EEGNet with gaze integration at both input and output levels
    """
    def __init__(self, num_input=1, num_class=2, channel=22, signal_length=6000,
                 fs=200, F1=8, D=3, dropout_rate=0.2):
        super(EEGNet_Gaze_Combined, self).__init__(
            num_input, num_class, channel, signal_length,
            fs, F1, D, dropout_rate
        )
        
        # Input gaze modulation parameter
        self.gaze_alpha = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, eeg, gaze=None, return_attention=False):
        """
        Forward pass with both input and output gaze integration
        
        Args:
            eeg: (batch, channel, time) - EEG signals
            gaze: (batch, channel, time) or None - Gaze attention maps
            return_attention: Whether to return attention map
            
        Returns:
            If return_attention=False: logits only
            If return_attention=True: dict with 'logits' and 'attention_map'
        """
        self.features = None
        
        # ========== INPUT INTEGRATION ==========
        if gaze is not None:
            # Ensure same shape
            if eeg.shape != gaze.shape:
                gaze = F.interpolate(
                    gaze.unsqueeze(1),
                    size=eeg.shape[-1],
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
            
            # Input modulation
            eeg = eeg * (1.0 + self.gaze_alpha * gaze)
        
        # Add channel dimension
        x = eeg.unsqueeze(1)
        
        # ========== FORWARD PASS ==========
        x = self.bn1(self.conv2d(x))
        x = self.bn2(self.depthwise_conv(x))
        x = self.elu(x)
        x = self.dropout(self.avg_pool1(x))
        x = self.sep_conv_depth(x)
        x = self.bn3(self.sep_conv_point(x))
        x = self.elu(x)
        x = self.dropout(self.avg_pool2(x))
        
        # Store features for attention
        features = x
        
        # ========== OUTPUT ATTENTION ==========
        attention_map_low = self.attention_conv(features)
        
        # Apply attention to features (optional)
        # weighted_features = features * attention_map_low
        
        # Flatten and classify
        x = self.flatten(x)
        logits = self.fc(x)
        probs = self.softmax(logits)
        
        if return_attention:
            # Upsample attention to original time resolution
            attention_map = F.interpolate(
                attention_map_low,
                size=(1, self.signal_length),
                mode='bilinear',
                align_corners=False
            )
            attention_map = attention_map.squeeze(1)
            attention_map = attention_map.repeat(1, self.channel, 1)
            
            return {
                'logits': probs,
                'attention_map': attention_map,
                'features': features
            }
        else:
            return probs
    
    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'model': 'EEGNet_Gaze_Combined',
            'gaze_integration': 'both',
            'gaze_alpha': self.gaze_alpha.item()
        })
        return config