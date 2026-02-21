"""
NeuroGATE with gaze integration in OUTPUT (attention maps)
This is your existing model - updated to use base components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_components import GateDilate, ResConv, EEG_Attention_MultiRes

class NeuroGATE_Gaze_Output(nn.Module):
    def __init__(self, n_chan: int = 22, n_outputs: int = 2, 
                 original_time_length: int = 6000, dropout_rate: float = 0.2):
        super(NeuroGATE_Gaze_Output, self).__init__()
        
        self.n_chan = n_chan
        self.n_outputs = n_outputs
        self.original_time_length = original_time_length
        self.dropout_rate = dropout_rate
        
        # ========== EVERYTHING BELOW IS IDENTICAL TO ORIGINAL ==========
        
        # NeuroGATE architecture
        fused_ch = 2 * n_chan
        res1_in = fused_ch
        res1_out = res1_in + 24
        
        self.res_conv1 = ResConv(res1_in, dropout_rate)  # PASS DROPOUT_RATE
        self.gate_dilate1 = GateDilate(res1_in, res1_out, 3, 8)
        
        self.conv1 = nn.Conv1d(in_channels=res1_out, out_channels=20, 
                              kernel_size=3, padding=1)
        
        self.res_conv2 = ResConv(20, dropout_rate)  # PASS DROPOUT_RATE
        self.gate_dilate2 = GateDilate(20, 20 + 24, 3, 8)
        
        self.res_conv3 = ResConv(20, dropout_rate)  # PASS DROPOUT_RATE
        
        self.bn1 = nn.BatchNorm1d(fused_ch)
        self.bn2 = nn.BatchNorm1d(20)
        self.conv2 = nn.Conv1d(in_channels=20 + 24, out_channels=20, 
                              kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(20)
        self.conv3 = nn.Conv1d(in_channels=20 + 24, out_channels=20, 
                              kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(20)
        
        # Encoder - UPDATED TO USE DROPOUT_RATE
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(20, 4, dropout=dropout_rate, batch_first=True), 2
        )
        
        # ========== ONLY CHANGE: Use MultiRes Attention ==========
        self.attention_layer = EEG_Attention_MultiRes(
            n_channels=n_chan,
            feature_channels=20,
            original_time_length=original_time_length
        )
        
        # Final classification layer - ADDED DROPOUT BEFORE FC
        self.fc_dropout = nn.Dropout(dropout_rate)  # ADDED DROPOUT LAYER
        self.fc = nn.Linear(20, n_outputs)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x, return_attention=False):
        """
        x: (batch_size, channels, time_steps) - Original EEG
        
        Returns:
        - If return_attention=False: logits only [batch, n_outputs]
        - If return_attention=True: dictionary with 'logits' and 'attention_map'
        """
        batch_size, channels, original_time = x.shape
        
        # ========== 1. FEATURE EXTRACTION ==========
        # Multi-scale feature extraction with pooling
        # Note: Actual dimensions depend on original_time_length parameter
        
        # First pooling: reduces by factor of 5
        x1 = F.avg_pool1d(x, kernel_size=5, stride=5)
        x2 = F.max_pool1d(x, kernel_size=5, stride=5)
        x = torch.cat((x1, x2), dim=1)  # [batch, 44, T/5]
        x = self.bn1(x)
        
        # SAVE early features (T/5 time points)
        features_3000 = x
        
        x1 = self.res_conv1(x)  # NOW HAS DROPOUT INSIDE
        x2 = self.gate_dilate1(x)
        x = x1 + x2
        x = F.dropout1d(x, self.dropout_rate, training=self.training)  # USE PARAMETER
        x = F.max_pool1d(x, kernel_size=5, stride=5)  # [batch, *, T/25]
        
        x = F.relu(self.bn2(self.conv1(x)))
        
        # SAVE middle features (T/25 time points)
        features_600 = x  # [batch, 20, T/25]
        
        x1 = self.res_conv2(x)  # NOW HAS DROPOUT INSIDE
        x2 = self.gate_dilate2(x)
        x = x1 + x2
        x = self.bn3(self.conv2(x))
        
        x = self.res_conv3(x)  # NOW HAS DROPOUT INSIDE
        x = F.max_pool1d(x, kernel_size=5, stride=5)  # [batch, *, T/125]
        
        x = self.bn4(self.conv3(x))  # [batch, 20, T/125]
        
        # ========== 2. TRANSFORMER ENCODER ==========
        x = x.permute(0, 2, 1)  # (batch, time, features)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)  # Back to (batch, features, time)
        
        # These are our final encoded features at reduced temporal resolution
        features_120 = x  # [batch, 20, T/125]
        
        # ========== 3. GENERATE ATTENTION MAP ==========
        attention_map = None
        if return_attention:
            # USE MULTI-RESOLUTION: Pass features from ALL 3 scales
            attention_map = self.attention_layer(
                features_3000, features_600, features_120
            )  # [batch, n_chan, original_time_length]
        
        # ========== 4. CLASSIFICATION ==========
        # SAME AS ORIGINAL
        if attention_map is not None:
            # Downsample attention to match feature resolution (ADAPTIVE)
            # Calculate pooling factor dynamically based on actual dimensions
            feature_time_dim = features_120.shape[-1]  # Actual temporal dimension of features
            attention_time_dim = attention_map.shape[-1]  # Temporal dimension of attention
            pool_factor = attention_time_dim // feature_time_dim
            
            if pool_factor > 1:
                attention_down = F.avg_pool1d(attention_map, kernel_size=pool_factor, stride=pool_factor)
            else:
                # If dimensions already match, no pooling needed
                attention_down = attention_map
            
            # Apply attention (average over channels for features)
            attention_weights = attention_down.mean(dim=1, keepdim=True)  # [batch, 1, feature_time_dim]
            attended_features = features_120 * attention_weights
            x_pooled = torch.mean(attended_features, dim=2)
        else:
            x_pooled = torch.mean(features_120, dim=2)
        
        # APPLY DROPOUT BEFORE FINAL LAYER
        x_pooled = self.fc_dropout(x_pooled)  # ADDED DROPOUT HERE
        logits = self.fc(x_pooled)
        
        # ========== 5. RETURN VALUES ==========
        if return_attention:
            return {
                'logits': logits,
                'attention_map': attention_map,
                'features': features_120
            }
        else:
            return logits
    
    def get_attention(self, x):
        """Convenience method to get attention map only"""
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True)
            return outputs['attention_map']
    
    def get_features(self, x):
        """Get intermediate features"""
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True)
            return outputs['features']
    
    def get_config(self):
        """Get model configuration"""
        return {
            'gaze_integration': 'output'
        }