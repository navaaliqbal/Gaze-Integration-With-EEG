"""
Original SCNet architecture without any gaze integration
Based on the original SCNet implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MFFMBlock(nn.Module):
    """Multi-scale Feature Fusion Module"""
    def __init__(self, in_channels):
        super(MFFMBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(16)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1_cat = torch.cat((x1, x), dim=1)
        x2 = F.relu(self.bn2(self.conv2(x1_cat)))
        return torch.cat((x2, x1_cat), dim=1)


class SILM(nn.Module):
    """Statistical Information Layer Module"""
    def __init__(self):
        super(SILM, self).__init__()

    def forward(self, x):
        # x shape: [B, C, L] - EEG channels
        gap = torch.mean(x, dim=1, keepdim=True)  # [B, 1, L]
        gsp = torch.std(x, dim=1, keepdim=True, unbiased=False)  # [B, 1, L]
        gmp, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, L]
        gap = F.dropout(gap, 0.05, training=self.training)
        gsp = F.dropout(gsp, 0.05, training=self.training)
        gmp = F.dropout(gmp, 0.05, training=self.training)
        return torch.cat((x, gap, gsp, gmp), dim=1)  # [B, C+3, L]


class SCNet_Base(nn.Module):
    """
    Base SCNet architecture without gaze integration
    Compatible with 22-channel EEG input and binary classification
    """
    def __init__(self, n_chan: int = 22, n_outputs: int = 2, original_time_length: int = 6000):
        super(SCNet_Base, self).__init__()
        
        self.n_chan = n_chan
        self.n_outputs = n_outputs
        self.original_time_length = original_time_length
        
        # SILM: Input channels = n_chan, Output channels = n_chan + 3
        self.silm = SILM()
        
        # After SILM and pooling: (n_chan + 3) * 2 = (n_chan + 3)*2
        silm_out_channels = n_chan + 3
        self.bn1 = nn.BatchNorm1d(silm_out_channels * 2)
        
        # First set of MFFM blocks
        self.mffm_block1 = MFFMBlock(silm_out_channels * 2)  # Input: (n_chan+3)*2
        self.mffm_block2 = MFFMBlock(silm_out_channels * 2)
        
        # After MFFM blocks: each outputs (input_channels + 24)
        # So after sum, channels = (silm_out_channels*2) + 24
        mffm_out_channels = (silm_out_channels * 2) + 24
        
        self.conv1 = nn.Conv1d(in_channels=mffm_out_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Second set of MFFM blocks
        self.mffm_block3 = MFFMBlock(64)  # Input: 64, Output: 64+24=88
        self.mffm_block4 = MFFMBlock(64)
        
        self.conv2 = nn.Conv1d(in_channels=88, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Third MFFM block
        self.mffm_block5 = MFFMBlock(64)  # Input: 64, Output: 64+24=88
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
    
    def forward(self, x):
        """
        Forward pass
        x: [B, n_chan, T] - EEG signals
        Returns: [B, n_outputs] - logits
        """
        # SILM - add statistical features
        x = self.silm(x)                                 # [B, n_chan+3, T]
        
        # Dual pooling
        x1 = F.avg_pool1d(x, 2, 2)                       # [B, n_chan+3, T/2]
        x2 = F.max_pool1d(x, 2, 2)                       # [B, n_chan+3, T/2]
        x = torch.cat((x1, x2), dim=1)                   # [B, (n_chan+3)*2, T/2]
        x = self.bn1(x)
        
        # First MFFM blocks with residual
        y1 = self.mffm_block1(x)                         # [B, (n_chan+3)*2+24, T/2]
        y2 = self.mffm_block2(x)                         # [B, (n_chan+3)*2+24, T/2]
        x = y1 + y2                                       # [B, (n_chan+3)*2+24, T/2]
        x = F.dropout(x, 0.5, training=self.training)
        
        # Pool and conv
        x = F.max_pool1d(x, 2, 2)                        # [B, (n_chan+3)*2+24, T/4]
        x = F.relu(self.bn2(self.conv1(x)))              # [B, 64, T/4]
        
        # Second MFFM blocks with residual
        y1 = self.mffm_block3(x)                         # [B, 88, T/4]
        y2 = self.mffm_block4(x)                         # [B, 88, T/4]
        x = y1 + y2                                       # [B, 88, T/4]
        x = F.relu(self.bn3(self.conv2(x)))              # [B, 64, T/4]
        
        # Third MFFM block
        x = self.mffm_block5(x)                           # [B, 88, T/4]
        x = F.max_pool1d(x, 2, 2)                        # [B, 88, T/8]
        x = self.conv3(x)                                 # [B, 64, T/8]
        
        # Global pooling and classification
        x = x.mean(dim=2)                                 # [B, 64]
        logits = self.fc(x)                               # [B, n_outputs]
        
        return logits
    
    def get_config(self):
        """Get model configuration"""
        return {
            'model': 'SCNet_Base',
            'gaze_integration': 'none',
            'n_chan': self.n_chan,
            'n_outputs': self.n_outputs,
            'original_time_length': self.original_time_length
    }