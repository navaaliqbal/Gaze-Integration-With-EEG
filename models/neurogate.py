import torch
import torch.nn as nn
import torch.nn.functional as F

class GateDilateLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super(GateDilateLayer, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.filter = nn.Conv1d(in_channels, in_channels, 1)
        self.gate = nn.Conv1d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.filter.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.gate.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=1.0)

    def forward(self, x):
        output = self.conv(x)
        filter = self.filter(output)
        gate = self.gate(output)
        tanh = self.tanh(filter)
        sig = self.sig(gate)
        z = tanh * sig
        z = z[:, :, :-self.padding]
        z = self.conv2(z)
        x = x + z
        return x

class GateDilate(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
        super(GateDilate, self).__init__()
        self.layers = nn.ModuleList()
        dilations = [2**i for i in range(dilation_rates)]
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1)
        for dilation in dilations:
            self.layers.append(GateDilateLayer(out_channels, kernel_size, dilation))
        torch.nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0)

    def forward(self, x):
        x = self.conv1d(x)
        for layer in self.layers:
            x = layer(x)
        return x

class ResConv(nn.Module):
    def __init__(self, in_channels):
        super(ResConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(16)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, input):
        x1 = F.relu(self.bn1(self.conv1(input)))
        x1 = torch.cat((x1, input), dim=1)
        x2 = F.relu(self.bn2(self.conv2(x1)))
        return torch.cat((x2, x1), dim=1)

class NeuroGATE(nn.Module):
    def __init__(self, n_chan: int = 22, n_outputs: int = 2):
        """
        n_chan: number of input channels to the model (before the avg/max pooling concat).
                Original code assumed n_chan=22 leading to 2*n_chan=44 used internally.
        n_outputs: number of final outputs (originally 2).
        """
        super(NeuroGATE, self).__init__()

        # after pooling fusion the channels become 2 * n_chan
        fused_ch = 2 * n_chan

        # ResGate Dilated Fusion 1 expects in_channels = fused_ch
        # ResConv(fused_ch) will output fused_ch + 24, so set GateDilate out_channels to that
        res1_in = fused_ch
        res1_out = res1_in + 24  # matches original 44 -> 68

        self.res_conv1 = ResConv(res1_in)
        self.gate_dilate1 = GateDilate(res1_in, res1_out, 3, 8)

        # conv1 will take the GateDilate/res_conv1 output channels (res1_out) and map to 20
        self.conv1 = nn.Conv1d(in_channels=res1_out, out_channels=20, kernel_size=3, padding=1)

        # ResGate Dilated Fusion 2 uses channels produced by conv1 (20)
        self.res_conv2 = ResConv(20)
        # res_conv2 produces 20 + 24 = 44, so GateDilate2 out_channels should be 44
        self.gate_dilate2 = GateDilate(20, 20 + 24, 3, 8)

        # ResConv 3 operates on conv2 output which is kept at 20 channels
        self.res_conv3 = ResConv(20)

        # Conv / BatchNorm blocks
        self.bn1 = nn.BatchNorm1d(fused_ch)
        self.bn2 = nn.BatchNorm1d(20)
        self.conv2 = nn.Conv1d(in_channels=20 + 24, out_channels=20, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(20)
        self.conv3 = nn.Conv1d(in_channels=20 + 24, out_channels=20, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(20)

        # Encoder and final FC
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(20, 4, dropout=0.5, batch_first=True), 2
        )
        self.fc = nn.Linear(20, n_outputs)

        # inits
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, return_attention=False):
        """
        Forward pass for baseline NeuroGATE
        
        Args:
            x: Input EEG tensor
            return_attention: Ignored (kept for API compatibility with gaze models)
        """
        # Pool Fusion Block
        x1 = F.avg_pool1d(x, kernel_size=5, stride=5)
        x2 = F.max_pool1d(x, kernel_size=5, stride=5)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn1(x)

        # ResGate Dilated Fusion 1
        x1 = self.res_conv1(x)
        x2 = self.gate_dilate1(x)
        x = x1 + x2

        #Apply spatial dropout
        x = F.dropout2d(x, 0.5, training=self.training)

        x = F.max_pool1d(x, kernel_size=5, stride=5)

        # ConvNorm Block 1
        x = F.relu(self.bn2(self.conv1(x)))

        # ResGate Dilated Fusion 2
        x1 = self.res_conv2(x)
        x2 = self.gate_dilate2(x)
        x = x1 + x2

        # ConvNorm Block 2
        x = self.bn3(self.conv2(x))

        # ResConv Block
        x = self.res_conv3(x)
        x = F.max_pool1d(x, kernel_size=5, stride=5)

        # ConvNorm Block 3
        x = self.bn4(self.conv3(x))

        # Encoder
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)

        # Feature Aggregation Pool
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x

