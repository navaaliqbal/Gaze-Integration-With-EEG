"""
EEGNet model with gaze integration variants for input, output, and combined.
Based on the EXACT original EEGNet architecture from the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet_Base(nn.Module):
    """
    Exact original EEGNet architecture without any gaze integration.
    Based on the Keras implementation from the paper.
    """
    def __init__(self, n_chan=22, n_time=6000, n_outputs=2, 
                 F1=8, D=2, F2=16, kernLength=64, dropoutRate=0.5):
        super(EEGNet_Base, self).__init__()
        
        self.n_chan = n_chan
        self.n_time = n_time
        self.n_outputs = n_outputs
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernLength = kernLength
        self.dropoutRate = dropoutRate
        
        # ==================== BLOCK 1 ====================
        # Temporal convolution
        # Input: (batch, 1, channels, time) - but we'll use (batch, 1, time, channels) for clarity
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise spatial convolution
        # This should convolve across channels dimension (which is dimension 3 after permute)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_chan, 1), 
                                         groups=F1, bias=False,
                                         padding='valid')
        self.bn_depthwise = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))  # Pool across time only
        self.dropout1 = nn.Dropout(dropoutRate)
        
        # ==================== BLOCK 2 ====================
        # Separable convolution (depthwise + pointwise)
        self.depthwise_conv2 = nn.Conv2d(F1 * D, F1 * D, (1, 16), 
                                          groups=F1 * D, 
                                          padding='same', bias=False)
        self.pointwise_conv = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))  # Pool across time only
        self.dropout2 = nn.Dropout(dropoutRate)
        
        # ==================== CLASSIFIER ====================
        # We'll initialize FC after we know the size
        self.fc = None
        
    def _init_fc_size(self):
        """Calculate flattened feature size after convolutions"""
        self.eval()
        with torch.no_grad():
            # Use batch size of 2 for initialization
            dummy = torch.zeros(2, 1, self.n_time, self.n_chan)
            x = self._forward_features(dummy)
            # x shape: (batch, F2, reduced_time, 1)
            self.fc_input_size = x.shape[1] * x.shape[2]  # F2 * reduced_time
            self.feature_shape = x.shape[1:]  # (F2, reduced_time, 1)
            
        # Final classifier
        self.fc = nn.Linear(self.fc_input_size, self.n_outputs)
        self._init_weights()
        self.train()
    
    def _init_weights(self):
        """Initialize weights with xavier uniform"""
        nn.init.xavier_uniform_(self.fc.weight)
    
    def _forward_features(self, x):
        """
        Forward pass through feature extraction layers.
        x: (batch, 1, time, channels) - after initial reshape
        """
        # Block 1
        x = self.conv1(x)                    # (B, F1, time, 1)
        x = self.bn1(x)
        x = self.elu1(x)
        
        # Depthwise conv expects input (B, F1, channels, time)
        # Current shape: (B, F1, time, 1) -> permute to (B, F1, 1, time)
        x = x.permute(0, 1, 3, 2)            # (B, F1, 1, time)
        x = self.depthwise_conv(x)            # (B, F1*D, 1, time)
        x = self.bn_depthwise(x)
        x = self.elu1(x)
        
        # Now x is (B, F1*D, 1, time) - permute to (B, F1*D, time, 1) for pooling
        x = x.permute(0, 1, 3, 2)            # (B, F1*D, time, 1)
        x = self.avg_pool1(x)                 # (B, F1*D, time/4, 1)
        x = self.dropout1(x)
        
        # Block 2
        # Current shape: (B, F1*D, time/4, 1) -> permute to (B, F1*D, 1, time/4)
        x = x.permute(0, 1, 3, 2)            # (B, F1*D, 1, time/4)
        x = self.depthwise_conv2(x)           # (B, F1*D, 1, time/4)
        x = self.pointwise_conv(x)            # (B, F2, 1, time/4)
        x = self.bn2(x)
        x = self.elu2(x)
        
        # Permute back for pooling: (B, F2, 1, time/4) -> (B, F2, time/4, 1)
        x = x.permute(0, 1, 3, 2)            # (B, F2, time/4, 1)
        x = self.avg_pool2(x)                 # (B, F2, time/32, 1)
        x = self.dropout2(x)
        
        return x
    
    def forward(self, eeg, **kwargs):
        """
        Forward pass for EEG-only model.
        eeg: (batch, channels, time) – as from dataloader
        """
        # Original EEGNet expects input shape (batch, channels, time, 1)
        # But we'll use (batch, 1, time, channels) for easier handling
        eeg = eeg.unsqueeze(1)                # (B, 1, channels, time)
        eeg = eeg.permute(0, 1, 3, 2)         # (B, 1, time, channels)
        
        features = self._forward_features(eeg)
        flat = features.reshape(features.size(0), -1)
        logits = self.fc(flat)
        return logits


class EEGNet_Gaze_Input(EEGNet_Base):
    """
    Gaze integration at input: gaze modulates EEG before feature extraction.
    """
    def __init__(self, n_chan=22, n_time=6000, n_outputs=2, **kwargs):
        super(EEGNet_Gaze_Input, self).__init__(n_chan, n_time, n_outputs, **kwargs)
        self.gaze_alpha = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, eeg, gaze=None, **kwargs):
        """
        eeg: (batch, channels, time)
        gaze: (batch, channels, time) or None
        """
        eeg = eeg.unsqueeze(1)                # (B, 1, channels, time)
        eeg = eeg.permute(0, 1, 3, 2)         # (B, 1, time, channels)
        
        if gaze is not None:
            gaze = gaze.unsqueeze(1)          # (B, 1, channels, time)
            gaze = gaze.permute(0, 1, 3, 2)   # (B, 1, time, channels)
            eeg = eeg * (1.0 + self.gaze_alpha * gaze)
        
        features = self._forward_features(eeg)
        flat = features.reshape(features.size(0), -1)
        logits = self.fc(flat)
        return logits


class EEGNet_Gaze_Output(EEGNet_Base):
    """
    Gaze integration at output: generate attention map from features.
    """
    def __init__(self, n_chan=22, n_time=6000, n_outputs=2, **kwargs):
        super(EEGNet_Gaze_Output, self).__init__(n_chan, n_time, n_outputs, **kwargs)
        self.attention_conv = None
    
    def _init_fc_size(self):
        """Override to initialize attention after feature shape is known"""
        super()._init_fc_size()
        # feature_shape is (F2, reduced_time, 1)
        self.attention_conv = nn.Conv2d(self.feature_shape[0], 1, 
                                         kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.attention_conv.weight)
    
    def forward(self, eeg, return_attention=False, **kwargs):
        """
        eeg: (batch, channels, time)
        Returns: logits if return_attention=False, else dict with 'logits' and 'attention_map'
        """
        eeg = eeg.unsqueeze(1)                # (B, 1, channels, time)
        eeg = eeg.permute(0, 1, 3, 2)         # (B, 1, time, channels)
        
        features = self._forward_features(eeg)      # (B, F2, reduced_time, 1)
        
        # Generate attention at feature resolution
        att_low = torch.sigmoid(self.attention_conv(features))  # (B, 1, reduced_time, 1)
        
        # Apply attention to features
        weighted_features = features * att_low       # (B, F2, reduced_time, 1)
        
        # Flatten and classify
        flat = weighted_features.reshape(weighted_features.size(0), -1)
        logits = self.fc(flat)
        
        if return_attention:
            # Upsample attention to original time resolution
            att_full = F.interpolate(
                att_low, 
                size=(self.n_time, 1), 
                mode='bilinear', 
                align_corners=False
            )  # (B, 1, time, 1)
            
            # Reshape to match gaze map shape (batch, channels, time)
            att_full = att_full.squeeze(1).squeeze(-1)  # (B, time)
            att_full = att_full.unsqueeze(1).repeat(1, self.n_chan, 1)  # (B, channels, time)
            
            return {'logits': logits, 'attention_map': att_full}
        else:
            return logits


class EEGNet_Gaze_Combined(EEGNet_Base):
    """
    Both input modulation and output attention.
    """
    def __init__(self, n_chan=22, n_time=6000, n_outputs=2, **kwargs):
        super(EEGNet_Gaze_Combined, self).__init__(n_chan, n_time, n_outputs, **kwargs)
        self.gaze_alpha = nn.Parameter(torch.tensor(1.0))
        self.attention_conv = None
    
    def _init_fc_size(self):
        """Override to initialize attention after feature shape is known"""
        super()._init_fc_size()
        self.attention_conv = nn.Conv2d(self.feature_shape[0], 1, 
                                         kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.attention_conv.weight)
    
    def forward(self, eeg, gaze=None, return_attention=False, **kwargs):
        """
        eeg: (batch, channels, time)
        gaze: (batch, channels, time) or None
        """
        eeg = eeg.unsqueeze(1)                # (B, 1, channels, time)
        eeg = eeg.permute(0, 1, 3, 2)         # (B, 1, time, channels)
        
        # Input modulation if gaze provided
        if gaze is not None:
            gaze = gaze.unsqueeze(1)          # (B, 1, channels, time)
            gaze = gaze.permute(0, 1, 3, 2)   # (B, 1, time, channels)
            eeg = eeg * (1.0 + self.gaze_alpha * gaze)
        
        features = self._forward_features(eeg)      # (B, F2, reduced_time, 1)
        
        # Generate attention
        att_low = torch.sigmoid(self.attention_conv(features))  # (B, 1, reduced_time, 1)
        
        # Apply attention
        weighted_features = features * att_low
        
        # Flatten and classify
        flat = weighted_features.reshape(weighted_features.size(0), -1)
        logits = self.fc(flat)
        
        if return_attention:
            # Upsample attention to original resolution
            att_full = F.interpolate(
                att_low, 
                size=(self.n_time, 1), 
                mode='bilinear', 
                align_corners=False
            )
            att_full = att_full.squeeze(1).squeeze(-1)  # (B, time)
            att_full = att_full.unsqueeze(1).repeat(1, self.n_chan, 1)  # (B, channels, time)
            
            return {'logits': logits, 'attention_map': att_full}
        else:
            return logits