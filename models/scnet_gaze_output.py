"""
SCNet with gaze integration at OUTPUT level (attention maps)
Modified for 22-channel EEG input with proper attention mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.scnet_base import MFFMBlock, SILM


class SCNet_Gaze_Output(nn.Module):
    """
    SCNet with gaze integration at output level
    Generates attention maps from features and uses them to weight features
    """
    
    def __init__(self, n_chan: int = 22, n_outputs: int = 2, original_time_length: int = 6000):
        super().__init__()
        
        self.n_chan = n_chan
        self.n_outputs = n_outputs
        self.original_time_length = original_time_length
        
        # SILM: Input channels = n_chan, Output channels = n_chan + 3
        self.silm = SILM()
        
        # After SILM and pooling: (n_chan + 3) * 2
        silm_out_channels = n_chan + 3
        self.bn1 = nn.BatchNorm1d(silm_out_channels * 2)
        
        # First set of MFFM blocks
        self.mffm_block1 = MFFMBlock(silm_out_channels * 2)
        self.mffm_block2 = MFFMBlock(silm_out_channels * 2)
        
        # After MFFM blocks: each outputs (input_channels + 24)
        mffm_out_channels = (silm_out_channels * 2) + 24
        
        self.conv1 = nn.Conv1d(in_channels=mffm_out_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Attention layer - generates attention from features at T/4 resolution
        self.attention_layer = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, n_chan, kernel_size=3, padding=1),  # Output: n_chan channels
            nn.Sigmoid()
        )
        
        # Upsample attention to original length
        self.upsample = nn.Upsample(size=original_time_length, mode='linear', align_corners=False)
        
        # Second set of MFFM blocks
        self.mffm_block3 = MFFMBlock(64)
        self.mffm_block4 = MFFMBlock(64)
        
        self.conv2 = nn.Conv1d(in_channels=88, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Third MFFM block
        self.mffm_block5 = MFFMBlock(64)
        self.conv3 = nn.Conv1d(in_channels=88, out_channels=64, kernel_size=3, padding=1)
        
        # Final classifier
        self.fc = nn.Linear(64, n_outputs)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize all weights"""
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attention_layer[0].weight)
        nn.init.xavier_uniform_(self.attention_layer[3].weight)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        
        Args:
            x: [B, n_chan, T] - EEG signals
            return_attention: Whether to return attention map
            
        Returns:
            If return_attention=False: logits only [B, n_outputs]
            If return_attention=True: dict with 'logits' and 'attention_map'
        """
        # SILM - add statistical features
        x = self.silm(x)                                         # [B, n_chan+3, T]
        
        # Dual pooling
        x1 = F.avg_pool1d(x, 2, 2)                               # [B, n_chan+3, T/2]
        x2 = F.max_pool1d(x, 2, 2)                               # [B, n_chan+3, T/2]
        x = torch.cat((x1, x2), dim=1)                           # [B, (n_chan+3)*2, T/2]
        x = self.bn1(x)
        
        # First MFFM blocks with residual
        y1 = self.mffm_block1(x)                                 # [B, (n_chan+3)*2+24, T/2]
        y2 = self.mffm_block2(x)                                 # [B, (n_chan+3)*2+24, T/2]
        x = y1 + y2                                               # [B, (n_chan+3)*2+24, T/2]
        x = F.dropout(x, 0.5, training=self.training)
        
        # Pool and conv
        x = F.max_pool1d(x, 2, 2)                                # [B, (n_chan+3)*2+24, T/4]
        x = F.relu(self.bn2(self.conv1(x)))                      # [B, 64, T/4]
        
        # Extract features for attention generation (at T/4 resolution)
        attention_features = x.clone()
        
        # Second MFFM blocks with residual
        y1 = self.mffm_block3(x)                                 # [B, 88, T/4]
        y2 = self.mffm_block4(x)                                 # [B, 88, T/4]
        x = y1 + y2                                               # [B, 88, T/4]
        x = F.relu(self.bn3(self.conv2(x)))                      # [B, 64, T/4]
        
        # Third MFFM block
        x = self.mffm_block5(x)                                   # [B, 88, T/4]
        x = F.max_pool1d(x, 2, 2)                                # [B, 88, T/8]
        x = self.conv3(x)                                         # [B, 64, T/8]
        
        # Generate and apply attention if requested
        attention_map = None
        if return_attention:
            # Generate attention from features at T/4 resolution
            att_map_low = self.attention_layer(attention_features)  # [B, n_chan, T/4]
            
            # Upsample to original resolution
            attention_map = self.upsample(att_map_low)              # [B, n_chan, T]
            
            # Downsample attention to match current feature resolution (T/8)
            att_down = F.avg_pool1d(attention_map, kernel_size=8, stride=8)  # T/8
            
            # Average across channels to get temporal weights
            att_weights = att_down.mean(dim=1, keepdim=True)        # [B, 1, T/8]
            
            # Apply attention to features
            x = x * att_weights
        
        # Global pooling and classification
        x = x.mean(dim=2)                                           # [B, 64]
        logits = self.fc(x)                                          # [B, n_outputs]
        
        if return_attention:
            return {
                'logits': logits,
                'attention_map': attention_map,  # [B, n_chan, T]
                'features': x
            }
        else:
            return logits
    
    def get_attention(self, x):
        """Convenience method to get attention map only"""
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True)
            return outputs['attention_map']
    
    def get_config(self):
        """Get model configuration"""
        return {
            'model': 'SCNet_Gaze_Output',
            'gaze_integration': 'output',
            'n_chan': self.n_chan,
            'n_outputs': self.n_outputs,
            'original_time_length': self.original_time_length
        }