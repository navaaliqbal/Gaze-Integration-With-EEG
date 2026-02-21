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
        # Compute global statistics across channels
        gap = torch.mean(x, dim=1, keepdim=True)
        gsp = torch.std(x, dim=1, keepdim=True, unbiased=False)
        gmp, _ = torch.max(x, dim=1, keepdim=True)
        gap = F.dropout(gap, 0.05, training=self.training)
        gsp = F.dropout(gsp, 0.05, training=self.training)
        gmp = F.dropout(gmp, 0.05, training=self.training)
        return torch.cat((x, gap, gsp, gmp), dim=1)  # [B, 25, T]

class SCNet_Gaze_Input(nn.Module):
    def __init__(self, n_chan=22, n_outputs=2,original_time_length: int = 15000):
        super(SCNet_Gaze_Input, self).__init__()

        # Learnable gaze alpha
        self.gaze_alpha = nn.Parameter(torch.tensor(1.0))
        self.n_chan = n_chan  # Changed to 22
        self.n_outputs = n_outputs  # Changed to 2 for binary classification
        self.original_time_length = original_time_length
        # Modules
        self.silm = SILM()
        self.bn1 = nn.BatchNorm1d((n_chan + 3) * 2)  # After SILM + pooling concat = (22+3)*2 = 50
        self.mffm_block1 = MFFMBlock((n_chan + 3) * 2)  # 50 -> 74
        self.mffm_block2 = MFFMBlock((n_chan + 3) * 2)  # 50 -> 74

        self.conv1 = nn.Conv1d(in_channels=74, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.mffm_block3 = MFFMBlock(32)
        self.mffm_block4 = MFFMBlock(32)
        self.conv2 = nn.Conv1d(in_channels=56, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.mffm_block5 = MFFMBlock(32)
        self.conv3 = nn.Conv1d(in_channels=56, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32, n_outputs)  # 2 for binary classification

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, gaze=None):
        """
        x: [B, 22, 15000] EEG input
        gaze: [B, 22, 15000] optional gaze map
        """
        # =========================
        # INPUT MODULATION WITH GAZE
        # =========================
        if gaze is not None:
            x = x * (1.0 + self.gaze_alpha * gaze)

        # =========================
        # SILM & initial pooling
        # =========================
        x = self.silm(x)                                 # [B, 25, T]
        x1 = F.avg_pool1d(x, 2, 2)
        x2 = F.max_pool1d(x, 2, 2)
        x = torch.cat((x1, x2), dim=1)                   # [B, 50, T/2]
        x = self.bn1(x)

        # =========================
        # MFFM Blocks
        # =========================
        y1 = self.mffm_block1(x)                        # [B, 32, T/2]
        y2 = self.mffm_block2(x)                        # [B, 32, T/2]
        x = y1 + y2                                      # [B,32,T/2]
        x = F.dropout2d(x, 0.5, training=self.training)
        x = F.max_pool1d(x, 2, 2)                       # [B,32,T/4]
        x = F.relu(self.bn2(self.conv1(x)))             # [B,32,T/4]

        y1 = self.mffm_block3(x)                        # [B,56,T/4]
        y2 = self.mffm_block4(x)                        # [B,56,T/4]
        x = y1 + y2                                      # [B,56,T/4]
        x = F.relu(self.bn3(self.conv2(x)))             # [B,32,T/4]

        x = self.mffm_block5(x)                         # [B,56,T/4]
        x = F.max_pool1d(x, 2, 2)                       # [B,56,T/8]
        x = self.conv3(x)                               # [B,32,T/8]
        x = x.mean(dim=2)                               # Global average over time
        x = self.fc(x)                                  # [B,2] logits for binary classification
        return x

    def get_config(self):
        """Get model configuration"""
        return {
            'model': 'SCNet_Gaze_Input',
            'gaze_integration': 'input',
            'n_chan': self.n_chan,
            'n_outputs': self.n_outputs,
            'original_time_length': self.original_time_length
        }
