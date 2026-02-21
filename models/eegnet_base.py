"""
Original EEGNet architecture without any gaze integration
Based on the specific implementation provided
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet_Base(nn.Module):
    """
    Base EEGNet architecture without gaze integration
    
    Parameters:
        num_input: number of input channels (always 1 for EEG)
        num_class: number of output classes
        channel: number of EEG channels
        signal_length: number of samples per trial
        fs: sampling frequency
        F1: number of temporal filters
        D: depth multiplier
        F2: number of pointwise filters (D*F1)
        dropout_rate: dropout probability
    """
    def __init__(self, num_input=1, num_class=2, channel=22, signal_length=6000, 
                 fs=200, F1=8, D=3, dropout_rate=0.2):
        super(EEGNet_Base, self).__init__()
        
        self.num_input = num_input
        self.num_class = num_class
        self.channel = channel
        self.signal_length = signal_length
        self.fs = fs
        self.F1 = F1
        self.D = D
        self.F2 = D * F1
        self.dropout_rate = dropout_rate
        
        # Calculate kernel sizes based on fs
        kernel_size_1 = (1, round(fs/2))
        kernel_size_2 = (channel, 1)
        kernel_size_3 = (1, round(fs/8))
        kernel_size_4 = (1, 1)
        
        kernel_avgpool_1 = (1, 4)
        kernel_avgpool_2 = (1, 8)
        
        # Calculate paddings
        ks0 = int(round((kernel_size_1[0] - 1) / 2))
        ks1 = int(round((kernel_size_1[1] - 1) / 2))
        kernel_padding_1 = (ks0, ks1 - 1)
        
        ks0 = int(round((kernel_size_3[0] - 1) / 2))
        ks1 = int(round((kernel_size_3[1] - 1) / 2))
        kernel_padding_3 = (ks0, ks1)
        
        # Layer 1 - Temporal convolution
        self.conv2d = nn.Conv2d(num_input, F1, kernel_size_1, padding=kernel_padding_1)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Layer 2 - Depthwise spatial convolution
        self.depthwise_conv = nn.Conv2d(F1, D * F1, kernel_size_2, groups=F1)
        self.bn2 = nn.BatchNorm2d(D * F1)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d(kernel_avgpool_1)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Layer 3 - Separable convolution
        self.sep_conv_depth = nn.Conv2d(D * F1, D * F1, kernel_size_3,
                                         padding=kernel_padding_3, groups=D * F1)
        self.sep_conv_point = nn.Conv2d(D * F1, self.F2, kernel_size_4)
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.avg_pool2 = nn.AvgPool2d(kernel_avgpool_2)
        
        # Layer 4 - Classification
        self.flatten = nn.Flatten()
        
        # Calculate flattened size dynamically
        self._init_fc_size()
        
        self.softmax = nn.Softmax(dim=1)
        
        self._initialize_weights()
    
    def _init_fc_size(self):
        """Calculate the size of the flattened features dynamically"""
        with torch.no_grad():
            # Create dummy input
            dummy = torch.zeros(1, 1, self.channel, self.signal_length)
            
            # Forward through conv layers
            x = self.bn1(self.conv2d(dummy))
            x = self.bn2(self.depthwise_conv(x))
            x = self.elu(x)
            x = self.avg_pool1(x)
            x = self.sep_conv_depth(x)
            x = self.bn3(self.sep_conv_point(x))
            x = self.elu(x)
            x = self.avg_pool2(x)
            
            # Flatten and get size
            x = self.flatten(x)
            self.flat_size = x.shape[1]
        
        # Create linear layer with correct size
        self.fc = nn.Linear(self.flat_size, self.num_class)
    
    def _initialize_weights(self):
        """Initialize weights with xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        """
        Forward pass
        x: (batch, channel, time) - EEG signals
        """
        # Add channel dimension for Conv2d (batch, 1, channel, time)
        x = x.unsqueeze(1)
        
        # Layer 1
        x = self.bn1(self.conv2d(x))
        
        # Layer 2
        x = self.bn2(self.depthwise_conv(x))
        x = self.elu(x)
        x = self.dropout(self.avg_pool1(x))
        
        # Layer 3
        x = self.sep_conv_depth(x)
        x = self.bn3(self.sep_conv_point(x))
        x = self.elu(x)
        x = self.dropout(self.avg_pool2(x))
        
        # Layer 4
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        
        return x
    
    def get_config(self):
        """Get model configuration"""
        return {
            'model': 'EEGNet_Base',
            'gaze_integration': 'none',
            'num_class': self.num_class,
            'channel': self.channel,
            'signal_length': self.signal_length,
            'F1': self.F1,
            'D': self.D,
            'F2': self.F2,
            'flat_size': self.flat_size
        }