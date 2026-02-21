"""
EEGNet with gaze integration at OUTPUT level (attention maps)
Generates attention maps from features and uses them for gaze supervision
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.eegnet_base import EEGNet_Base


class EEGNet_Gaze_Output(EEGNet_Base):
    """
    EEGNet with gaze integration at output level
    Generates attention maps from intermediate features
    """
    def __init__(self, num_input=1, num_class=2, channel=22, signal_length=6000,
                 fs=200, F1=8, D=3, dropout_rate=0.2):
        super(EEGNet_Gaze_Output, self).__init__(
            num_input, num_class, channel, signal_length,
            fs, F1, D, dropout_rate
        )
        
        # Calculate the size of feature maps after conv layers
        # After avg_pool2, feature map size: (F2, 1, signal_length/32)
        self.feature_time = round(signal_length / 32)
        
        # Attention generation layer
        # Takes features and produces attention map over time
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.F2, self.F2 // 2, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(self.F2 // 2),
            nn.ELU(),
            nn.Conv2d(self.F2 // 2, 1, kernel_size=(1, 3), padding=(0, 1)),
            nn.Sigmoid()
        )
        
        # Store features for attention
        self.features = None
        self._register_hook()
    
    def _register_hook(self):
        """Register hook to capture features before the final flatten layer"""
        def hook(module, input, output):
            self.features = output
        
        # Hook after the last convolutional layer (before flatten)
        self.avg_pool2.register_forward_hook(hook)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass with attention generation
        
        Args:
            x: (batch, channel, time) - EEG signals
            return_attention: Whether to return attention map
            
        Returns:
            If return_attention=False: logits only
            If return_attention=True: dict with 'logits' and 'attention_map'
        """
        self.features = None
        batch_size = x.shape[0]
        
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Forward through layers (features captured by hook)
        x = self.bn1(self.conv2d(x))
        x = self.bn2(self.depthwise_conv(x))
        x = self.elu(x)
        x = self.dropout(self.avg_pool1(x))
        x = self.sep_conv_depth(x)
        x = self.bn3(self.sep_conv_point(x))
        x = self.elu(x)
        x = self.dropout(self.avg_pool2(x))  # Features captured here
        
        # Store features for attention
        features = self.features if self.features is not None else x
        
        # Generate attention map from features
        attention_map_low = self.attention_conv(features)  # (batch, 1, 1, time/32)
        
        # Apply attention to features (optional - can be used for weighting)
        # For gaze supervision, we'll return the attention map
        
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
            )  # (batch, 1, 1, time)
            
            # Reshape to match gaze map shape (batch, channel, time)
            attention_map = attention_map.squeeze(1)  # (batch, 1, time)
            attention_map = attention_map.repeat(1, self.channel, 1)  # (batch, channel, time)
            
            return {
                'logits': probs,
                'attention_map': attention_map,
                'features': features
            }
        else:
            return probs
    
    def get_attention(self, x):
        """Convenience method to get attention map only"""
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True)
            return outputs['attention_map']
    
    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'model': 'EEGNet_Gaze_Output',
            'gaze_integration': 'output'
        })
        return config