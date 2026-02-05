"""
NeuroGATE with BOTH input and output gaze integration
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_components import GateDilate, ResConv, EEG_Attention_MultiRes

class NeuroGATE_Combined(nn.Module):
    def __init__(self, n_chan=22, n_outputs=2, original_time_length=15000, dropout_rate=0.5):
        super().__init__()
        self.n_chan = n_chan
        self.n_outputs = n_outputs
        self.original_time_length = original_time_length
        self.dropout_rate = dropout_rate
        self.gaze_alpha = nn.Parameter(torch.tensor(1.0))
        
        # Architecture
        fused_ch = 2 * n_chan
        res1_in = fused_ch
        res1_out = res1_in + 24
        
        self.res_conv1 = ResConv(res1_in, dropout_rate)
        self.gate_dilate1 = GateDilate(res1_in, res1_out, 3, 8)
        self.conv1 = nn.Conv1d(res1_out, 20, 3, padding=1)
        
        self.res_conv2 = ResConv(20, dropout_rate)
        self.gate_dilate2 = GateDilate(20, 20+24, 3, 8)
        self.res_conv3 = ResConv(20, dropout_rate)
        
        self.bn1 = nn.BatchNorm1d(fused_ch)
        self.bn2 = nn.BatchNorm1d(20)
        self.conv2 = nn.Conv1d(20+24, 20, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(20)
        self.conv3 = nn.Conv1d(20+24, 20, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(20)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(20, 4, dropout=dropout_rate, batch_first=True), 2
        )
        
        self.attention_layer = EEG_Attention_MultiRes(n_channels=n_chan, feature_channels=20, original_time_length=original_time_length)
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(20, n_outputs)
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, eeg, gaze=None, return_attention=False):
        if gaze is not None:
            eeg = eeg * (1 + self.gaze_alpha * gaze)
        
        # Multi-scale features
        x1 = torch.cat([F.avg_pool1d(eeg,5,5), F.max_pool1d(eeg,5,5)], dim=1)  # 44, 3000
        x1 = self.bn1(x1)
        features_3000 = x1
        
        x = self.res_conv1(x1) + self.gate_dilate1(x1)
        x = F.dropout1d(x, self.dropout_rate, training=self.training)
        x = F.max_pool1d(x,5,5)
        x = F.relu(self.bn2(self.conv1(x)))
        features_600 = x  # 600 points
        
        x = self.res_conv2(x) + self.gate_dilate2(x)
        x = self.bn3(self.conv2(x))
        x = self.res_conv3(x)
        x = F.max_pool1d(x,5,5)
        x = self.bn4(self.conv3(x))
        x = x.permute(0,2,1)
        x = self.encoder(x)
        x = x.permute(0,2,1)
        features_120 = x
        
        attention_map = None
        if return_attention:
            attention_map = self.attention_layer(features_3000, features_600, features_120)
        
        # Classification
        if attention_map is not None:
            attention_down = F.avg_pool1d(attention_map, 125, 125)  # [B, n_channels, 120]
            attention_down = attention_down.mean(dim=1, keepdim=True)  # [B,1,120]
            attended_features = features_120 * attention_down
            x_pooled = torch.mean(attended_features, dim=2)

        else:
            x_pooled = torch.mean(features_120, dim=2)
        
        x_pooled = self.fc_dropout(x_pooled)
        logits = self.fc(x_pooled)
        
        if return_attention:
            return {'logits': logits, 'attention_map': attention_map, 'features': features_120}
        return logits
    
    def get_config(self):
        """Get model configuration"""
        return {
            'gaze_integration': 'both',
            'gaze_alpha': self.gaze_alpha.item()
        }
    def get_attention(self, x, gaze=None):
        """Convenience method to get attention map only"""
        with torch.no_grad():
            outputs = self.forward(x, gaze, return_attention=True)
            return outputs['attention_map']
    
    def get_features(self, x, gaze=None):
        """Get intermediate features"""
        with torch.no_grad():
            outputs = self.forward(x, gaze, return_attention=True)
            return outputs['features']