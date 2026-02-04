"""
Base components shared by all NeuroGATE models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== COMMON COMPONENTS ==========

class GateDilateLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, 
                             padding=self.padding, dilation=dilation)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.filter = nn.Conv1d(in_channels, in_channels, 1)
        self.gate = nn.Conv1d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)
        
        nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
        nn.init.xavier_uniform_(self.filter.weight, gain=1.0)
        nn.init.xavier_uniform_(self.gate.weight, gain=1.0)
        nn.init.xavier_uniform_(self.conv2.weight, gain=1.0)
    
    def forward(self, x):
        output = self.conv(x)
        filter = self.filter(output)
        gate = self.gate(output)
        tanh = self.tanh(filter)
        sig = self.sig(gate)
        z = tanh * sig
        z = z[:, :, :-self.padding] if self.padding > 0 else z
        z = self.conv2(z)
        z = F.dropout(z, p=0.3, training=self.training)
        x = x + z
        return x

class GateDilate(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
        super().__init__()
        self.layers = nn.ModuleList()
        dilations = [2**i for i in range(dilation_rates)]
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1)
        for dilation in dilations:
            self.layers.append(GateDilateLayer(out_channels, kernel_size, dilation))
        nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0)
    
    def forward(self, x):
        x = self.conv1d(x)
        for layer in self.layers:
            x = layer(x)
        return x

class ResConv(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.dropout1 = nn.Dropout1d(dropout_rate)
        self.conv2 = nn.Conv1d(in_channels + 8, 16, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout1d(dropout_rate)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.dropout1(x1)
        x1 = torch.cat([x1, x], dim=1)
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = self.dropout2(x2)
        return torch.cat([x2, x1], dim=1)

class EEG_Attention(nn.Module):
    """Learnable attention module for gaze alignment"""
    def __init__(self, n_channels=22, feature_channels=20, original_time_length=15000):
        super().__init__()
        
        self.n_channels = n_channels
        self.original_time_length = original_time_length
        
        # Temporal Attention
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(feature_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(feature_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_channels),
            nn.Sigmoid()
        )
        
        # Smooth upsampling
        self.upsample = nn.ConvTranspose1d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=5,
            stride=125,
            padding=2,
            output_padding=0,
            groups=n_channels
        )
        
        nn.init.kaiming_uniform_(self.upsample.weight, nonlinearity='linear')
        nn.init.zeros_(self.upsample.bias)
        
    def forward(self, features):
        """Generate attention map from features"""
        batch_size, feat_channels, reduced_time = features.shape
        
        # Temporal attention
        temporal_att = self.temporal_attention(features)
        
        # Spatial attention
        spatial_att = self.spatial_attention(features)
        
        # Combine
        temporal_exp = temporal_att.unsqueeze(1)
        spatial_exp = spatial_att.unsqueeze(-1).unsqueeze(-1)
        combined = temporal_exp * spatial_exp
        combined = combined.squeeze(2)
        
        # Upsample
        attention_map = self.upsample(combined)
        
        # Ensure exact original length
        if attention_map.shape[-1] != self.original_time_length:
            attention_map = F.interpolate(
                attention_map, size=self.original_time_length, mode='linear', align_corners=False
            )
        
        attention_map = torch.clamp(attention_map, 0, 1)
        
        return attention_map
    

# ========== EEG ATTENTION MODULE (for combined) ==========
class EEG_Attention_MultiRes(nn.Module):
    def __init__(self, n_channels=22, feature_channels=20, original_time_length=15000):
        super().__init__()
        self.n_channels = n_channels
        self.original_time_length = original_time_length
        
        # Temporal attention at 3 scales
        self.temporal_3000 = nn.Sequential(
            nn.Conv1d(44, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 3, padding=1), nn.Sigmoid()
        )
        self.temporal_600 = nn.Sequential(
            nn.Conv1d(20, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 3, padding=1), nn.Sigmoid()
        )
        self.temporal_120 = nn.Sequential(
            nn.Conv1d(feature_channels, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 3, padding=1), nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(feature_channels, 64), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_channels),
            nn.Sigmoid()
        )
        
        # Learnable weights for scale fusion
        self.scale_weights = nn.Parameter(torch.ones(3)/3)

    def forward(self, features_3000, features_600, features_120):
        # Temporal attention
        att_3000 = F.interpolate(self.temporal_3000(features_3000),
                                 size=self.original_time_length,
                                 mode='linear', align_corners=False)
        att_600 = F.interpolate(self.temporal_600(features_600),
                                size=self.original_time_length,
                                mode='linear', align_corners=False)
        att_120 = F.interpolate(self.temporal_120(features_120),
                                size=self.original_time_length,
                                mode='linear', align_corners=False)
        
        # Fuse temporal scales
        weights = F.softmax(self.scale_weights, dim=0)
        temporal_fused = weights[0]*att_3000 + weights[1]*att_600 + weights[2]*att_120  # [B,1,T]
        
        # Spatial attention
        spatial_att = self.spatial_attention(features_120)  # [B, n_channels]
        attention_map = temporal_fused * spatial_att.unsqueeze(-1)  # [B, n_channels, T]
        
        return torch.clamp(attention_map, 0, 1)  # now 3D, compatible with avg_pool1d
