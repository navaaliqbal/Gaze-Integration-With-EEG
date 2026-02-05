"""
SCNet with gaze integration in OUTPUT (attention maps)
Modified for 22-channel EEG input
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MFFMBlock(nn.Module):
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
    def __init__(self):
        super(SILM, self).__init__()

    def forward(self, x):
        # x shape: [B, 22, L] - 22 EEG channels
        gap = torch.mean(x, dim=1, keepdim=True)  # [B, 1, L]
        gsp = torch.std(x, dim=1, keepdim=True, unbiased=False)  # [B, 1, L]
        gmp, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, L]
        gap = F.dropout(gap, 0.05, training=self.training)
        gsp = F.dropout(gsp, 0.05, training=self.training)
        gmp = F.dropout(gmp, 0.05, training=self.training)
        return torch.cat((x, gap, gsp, gmp), dim=1)  # [B, 25, L] (22 + 3)

class SCNet_Gaze_Output(nn.Module):
    """
    SCNet with gaze integration in output (attention maps)
    Now accepts 22-channel EEG input
    """
    
    def __init__(self, n_chan: int = 22, n_outputs: int = 2, original_time_length: int = 15000):
        super().__init__()
        
        self.n_chan = n_chan  # Changed to 22
        self.n_outputs = n_outputs  # Changed to 2 for binary classification
        self.original_time_length = original_time_length
        
        # SCNet Base Architecture - adjusted for 22→25 channels after SILM
        self.silm = SILM()  # Output: [B, 25, L] (22 channels + 3 stats)
        self.bn1 = nn.BatchNorm1d(50)  # Changed from 8 to 50 (25*2 after pooling)
        
        # MFFM Blocks - now accept 50 channels input
        self.mffm_block1 = MFFMBlock(50)  # Input: 50, Output: 50 + 24 = 74
        self.mffm_block2 = MFFMBlock(50)
        
        # Middle layers adjusted for larger channel dimensions
        # MFFMBlock output: 50 + 24 = 74 channels
        # After sum: 74 channels
        self.conv1 = nn.Conv1d(in_channels=74, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.mffm_block3 = MFFMBlock(64)  # Output: 64 + 24 = 88
        self.mffm_block4 = MFFMBlock(64)
        
        # After sum: 88 channels
        self.conv2 = nn.Conv1d(in_channels=88, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.mffm_block5 = MFFMBlock(64)  # Output: 64 + 24 = 88
        self.conv3 = nn.Conv1d(in_channels=88, out_channels=64, kernel_size=3, padding=1)
        
        # ATTENTION LAYER for output integration
        self.attention_layer = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 22, kernel_size=3, padding=1),  # Output: 22 channels
            nn.Sigmoid()
        )
        
        # Upsample attention to original length
        self.upsample = nn.Upsample(
            size=original_time_length, 
            mode='linear', 
            align_corners=False
        )
        
        # Final classification layer
        self.fc = nn.Linear(64, n_outputs)  # Input: 64 channels from conv3
        
        # Initialize weights
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
        Forward pass for 22-channel EEG
        
        Args:
            x: [B, 22, T] - EEG signals (22 channels)
            return_attention: Whether to return attention map
            
        Returns:
            If return_attention=False: logits only [batch, n_outputs]
            If return_attention=True: dictionary with 'logits' and 'attention_map'
        """
        batch_size, channels, original_time = x.shape
        
        # ========== 1. SCNet FEATURE EXTRACTION ==========
        x = self.silm(x)                                 # [B, 25, L]
        x1 = F.avg_pool1d(x, 2, 2)                       # [B, 25, L/2]
        x2 = F.max_pool1d(x, 2, 2)                       # [B, 25, L/2]
        x = torch.cat((x1, x2), dim=1)                   # [B, 50, L/2]
        x = self.bn1(x)

        y1 = self.mffm_block1(x)                        # [B, 74, L/2]
        y2 = self.mffm_block2(x)                        # [B, 74, L/2]
        x = y1 + y2                                      # [B, 74, L/2]
        # FIX: Use dropout1d instead of dropout2d for 1D signals
        x = F.dropout(x, 0.5, training=self.training)  # Changed from dropout2d

        x = F.max_pool1d(x, 2, 2)                       # [B, 74, L/4]
        x = F.relu(self.bn2(self.conv1(x)))             # [B, 64, L/4]

        #  Extract features at this level for attention generation
        attention_features = x.clone()                  # [B, 64, L/4]
        
        y1 = self.mffm_block3(x)                        # [B, 88, L/4]
        y2 = self.mffm_block4(x)                        # [B, 88, L/4]
        x = y1 + y2                                      # [B, 88, L/4]
        x = F.relu(self.bn3(self.conv2(x)))             # [B, 64, L/4]

        x = self.mffm_block5(x)                         # [B, 88, L/4]
        x = F.max_pool1d(x, 2, 2)                       # [B, 88, L/8]

        x = self.conv3(x)                               # [B, 64, L/8]
        
        # ========== 2. GENERATE ATTENTION MAP ==========
        attention_map = None
        if return_attention:
            # Generate attention from features
            attention_reduced = self.attention_layer(attention_features)  # [B, 22, L/4]
            
            # Upsample to original time length
            attention_map = self.upsample(attention_reduced)  # [B, 22, T]
            
            # FIX: Apply attention to features CORRECTLY
            # We need to downsample attention to match feature size (L/8)
            attention_down = F.avg_pool1d(attention_map, kernel_size=8, stride=8)  # T → T/8
            attention_down = F.avg_pool1d(attention_down, kernel_size=2, stride=2)  # T/8 → T/16? Wait, let's compute...
            
            # Actually, features are at L/8, attention_map is at T
            # We need: T → L/8 = T/8 if L=T
            # But L = T/4 at attention_features stage...
            # Let me trace:
            # Input: T
            # After silm: T
            # After first pool: T/2
            # After second pool: T/4 (attention_features stage)
            # After third pool: T/8 (final features stage)
            
            # So to go from attention_map (T) to features (T/8):
            # We need to downsample by factor of 8
            attention_down = F.avg_pool1d(attention_map, kernel_size=8, stride=8)  # T → T/8
            
            # Now attention_down should match x in time dimension
            attention_weights = attention_down.mean(dim=1, keepdim=True)  # [B, 1, T/8]
            
            # Verify dimensions match
            if x.shape[-1] != attention_weights.shape[-1]:
                # If still mismatched, interpolate
                attention_weights = F.interpolate(
                    attention_weights, 
                    size=x.shape[-1], 
                    mode='linear', 
                    align_corners=False
                )
            
            x = x * attention_weights
        
        # ========== 3. CLASSIFICATION ==========
        x_pooled = torch.mean(x, dim=2)                  # [B, 64]
        logits = self.fc(x_pooled)                      # [B, n_outputs]
        
        # ========== RETURN VALUES ==========
        if return_attention:
            return {
                'logits': logits,
                'attention_map': attention_map,  # [B, 22, T] - matches EEG channels
                'features': x
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
            'model': 'SCNet_Gaze_Output',
            'gaze_integration': 'output',
            'n_chan': self.n_chan,
            'n_outputs': self.n_outputs,
            'original_time_length': self.original_time_length
        }