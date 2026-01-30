"""
NeuroGATE with BOTH input and output gaze integration
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_components import GateDilate, ResConv, EEG_Attention

class NeuroGATE_Combined(nn.Module):
    """
    NeuroGATE with both input and output gaze integration
    1. Input integration: Gaze modulates EEG input
    2. Output integration: Generates attention maps aligned with gaze
    """
    
    def __init__(self, n_chan: int = 22, n_outputs: int = 2, original_time_length: int = 15000):
        super().__init__()
        
        self.n_chan = n_chan
        self.n_outputs = n_outputs
        self.original_time_length = original_time_length
        
        # 🔑 Learnable gaze strength for input integration
        self.gaze_alpha = nn.Parameter(torch.tensor(1.0))
        
        # NeuroGATE architecture
        fused_ch = 2 * n_chan
        res1_in = fused_ch
        res1_out = res1_in + 24
        
        self.res_conv1 = ResConv(res1_in)
        self.gate_dilate1 = GateDilate(res1_in, res1_out, 3, 8)
        
        self.conv1 = nn.Conv1d(in_channels=res1_out, out_channels=20, 
                              kernel_size=3, padding=1)
        
        self.res_conv2 = ResConv(20)
        self.gate_dilate2 = GateDilate(20, 20 + 24, 3, 8)
        
        self.res_conv3 = ResConv(20)
        
        self.bn1 = nn.BatchNorm1d(fused_ch)
        self.bn2 = nn.BatchNorm1d(20)
        self.conv2 = nn.Conv1d(in_channels=20 + 24, out_channels=20, 
                              kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(20)
        self.conv3 = nn.Conv1d(in_channels=20 + 24, out_channels=20, 
                              kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(20)
        
        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(20, 4, dropout=0.5, batch_first=True), 2
        )
        
        # ATTENTION LAYER for output integration
        self.attention_layer = EEG_Attention(
            n_channels=n_chan,
            feature_channels=20,
            original_time_length=original_time_length
        )
        
        # Final classification layer
        self.fc = nn.Linear(20, n_outputs)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, eeg, gaze=None, return_attention=False):
        """
        Forward pass with both input and output gaze integration
        
        Args:
            eeg: [B, C, T] - EEG signals
            gaze: [B, C, T] - Gaze attention maps (optional for input integration)
            return_attention: Whether to return attention map for output integration
            
        Returns:
            If return_attention=False: logits only [batch, n_outputs]
            If return_attention=True: dictionary with 'logits' and 'attention_map'
        """
        batch_size, channels, original_time = eeg.shape
        
        # ====================================================
        # INPUT INTEGRATION: GAZE-AS-GATE
        # ====================================================
        if gaze is not None:
            eeg = eeg * (1.0 + self.gaze_alpha * gaze)
        
        # ====================================================
        # NEUROGATE ARCHITECTURE
        # ====================================================
        # Pool Fusion
        x1 = F.avg_pool1d(eeg, kernel_size=5, stride=5)
        x2 = F.max_pool1d(eeg, kernel_size=5, stride=5)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn1(x)
        
        # ResGate Dilated Fusion 1
        x1 = self.res_conv1(x)
        x2 = self.gate_dilate1(x)
        x = x1 + x2
        x = F.dropout2d(x, 0.5, training=self.training)
        x = F.max_pool1d(x, kernel_size=5, stride=5)
        
        # Conv Block
        x = F.relu(self.bn2(self.conv1(x)))
        
        # ResGate Dilated Fusion 2
        x1 = self.res_conv2(x)
        x2 = self.gate_dilate2(x)
        x = x1 + x2
        x = self.bn3(self.conv2(x))
        
        # ResConv + Pool
        x = self.res_conv3(x)
        x = F.max_pool1d(x, kernel_size=5, stride=5)
        
        # Conv Block
        x = self.bn4(self.conv3(x))
        
        # ====================================================
        # TRANSFORMER ENCODER
        # ====================================================
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        
        features = x
        
        # ====================================================
        # OUTPUT INTEGRATION: GENERATE ATTENTION MAP
        # ====================================================
        attention_map = None
        if return_attention:
            attention_map = self.attention_layer(features)
        
        # ====================================================
        # CLASSIFICATION
        # ====================================================
        if attention_map is not None:
            attention_down = F.avg_pool1d(attention_map, kernel_size=125, stride=125)
            attention_weights = attention_down.mean(dim=1, keepdim=True)
            attended_features = features * attention_weights
            x_pooled = torch.mean(attended_features, dim=2)
        else:
            x_pooled = torch.mean(features, dim=2)
        
        logits = self.fc(x_pooled)
        
        # ====================================================
        # RETURN VALUES
        # ====================================================
        if return_attention:
            return {
                'logits': logits,
                'attention_map': attention_map,
                'features': features
            }
        else:
            return logits
    
    def get_config(self):
        """Get model configuration"""
        return {
            'gaze_integration': 'both',
            'gaze_alpha': self.gaze_alpha.item()
        }