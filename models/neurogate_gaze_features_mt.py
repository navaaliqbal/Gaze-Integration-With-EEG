"""
NeuroGATE with gaze features - MULTI-TASK LEARNING VERSION
Predicts both classification labels AND gaze features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_components import GateDilate, ResConv

class NeuroGATE_MultiTask(nn.Module):
    """
    Multi-task NeuroGATE:
    1. Main task: EEG classification (binary)
    2. Auxiliary task: Gaze feature reconstruction/regression
    """
    
    def __init__(self, 
                 n_chan: int = 22, 
                 n_outputs: int = 2, 
                 original_time_length: int = 15000,
                 gaze_feature_dim: int = 5,
                 fusion_method: str = 'early',
                 task_weight: float = 0.5):  # Weight for gaze reconstruction loss
    
        super().__init__()
        
        self.n_chan = n_chan
        self.n_outputs = n_outputs
        self.original_time_length = original_time_length
        self.gaze_feature_dim = gaze_feature_dim
        self.fusion_method = fusion_method
        self.task_weight = task_weight  # α in loss: L = L_class + α * L_gaze
        
        # ====================================================
        # GAZE FEATURE PROCESSING NETWORK
        # ====================================================
        self.gaze_encoder = nn.Sequential(
            nn.Linear(gaze_feature_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Learnable gaze strength
        self.gaze_alpha = nn.Parameter(torch.tensor(1.0))
        
        # ====================================================
        # GAZE FEATURE DECODER (for reconstruction)
        # ====================================================
        self.gaze_decoder = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, gaze_feature_dim)  # Reconstruct original gaze features
        )
        
        # ====================================================
        # NEUROGATE ARCHITECTURE
        # ====================================================
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
        
        # ====================================================
        # GAZE-EEG FUSION LAYERS
        # ====================================================
        if fusion_method == 'early':
            self.gaze_to_eeg = nn.Sequential(
                nn.Linear(32, n_chan * 4),
                nn.ReLU(),
                nn.Linear(n_chan * 4, n_chan * original_time_length // 50)
            )
            
        elif fusion_method == 'late':
            self.eeg_feature_dim = 20
            
        elif fusion_method == 'adaptive':
            self.gaze_attention = nn.Sequential(
                nn.Linear(32, 20),
                nn.Sigmoid()
            )
        
        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(20, 4, dropout=0.5, batch_first=True), 2
        )
        
        # ====================================================
        # MULTI-TASK HEADS
        # ====================================================
        # 1. Classification head
        if fusion_method == 'late':
            self.classifier = nn.Linear(20 + 32, n_outputs)
        else:
            self.classifier = nn.Linear(20, n_outputs)
        
        # 2. Gaze feature reconstruction head
        self.gaze_reconstructor = nn.Sequential(
            nn.Linear(20 + 32, 32),  # Combine EEG + gaze features
            nn.ReLU(),
            nn.Linear(32, gaze_feature_dim)  # Predict original gaze features
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, eeg, gaze_features=None):
        """
        Forward pass with multi-task learning
        
        Returns:
            classification_logits: [B, n_outputs] - Class predictions
            gaze_predictions: [B, gaze_feature_dim] - Reconstructed gaze features
            gaze_encoded: [B, 32] - Encoded gaze features (optional)
        """
        batch_size = eeg.size(0)
        device = eeg.device
        
        # ====================================================
        # PROCESS GAZE FEATURES
        # ====================================================
        if gaze_features is not None:
            # Store original gaze features for reconstruction loss
            original_gaze = gaze_features.clone()
            
            # Encode gaze features
            if gaze_features.dim() == 1:
                gaze_features = gaze_features.unsqueeze(0)
            gaze_encoded = self.gaze_encoder(gaze_features)  # [B, 32]
            
            if self.fusion_method == 'early':
                gaze_modulation = self.gaze_to_eeg(gaze_encoded)
                gaze_modulation = gaze_modulation.view(batch_size, self.n_chan, -1)
                
                if gaze_modulation.size(2) != eeg.size(2):
                    gaze_modulation = F.interpolate(
                        gaze_modulation, 
                        size=eeg.size(2), 
                        mode='linear', 
                        align_corners=False
                    )
                
                eeg = eeg * (1.0 + self.gaze_alpha * gaze_modulation)
        else:
            # Create zero gaze encoding if no gaze features
            gaze_encoded = torch.zeros(batch_size, 32, device=device)
            original_gaze = None
        
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
        
        # ====================================================
        # FEATURE EXTRACTION FOR MULTI-TASK LEARNING
        # ====================================================
        eeg_features = torch.mean(x, dim=2)  # [B, 20]
        
        # Combine EEG and gaze features for reconstruction
        if self.fusion_method == 'late':
            combined_features = torch.cat([eeg_features, gaze_encoded], dim=1)
            classification_logits = self.classifier(combined_features)
        else:
            classification_logits = self.classifier(eeg_features)
            combined_features = torch.cat([eeg_features, gaze_encoded], dim=1)
        
        # ====================================================
        # MULTI-TASK PREDICTIONS
        # ====================================================
        # 1. Gaze feature reconstruction
        gaze_predictions = self.gaze_reconstructor(combined_features)
        
        # 2. Optional: Decode gaze features through decoder
        gaze_reconstructed = self.gaze_decoder(gaze_encoded)
        
        return {
            'classification': classification_logits,
            'gaze_predictions': gaze_predictions,
            'gaze_reconstructed': gaze_reconstructed,
            'gaze_encoded': gaze_encoded,
            'original_gaze': original_gaze  # For loss calculation
        }
    
    def get_config(self):
        return {
            'gaze_integration': 'multi_task',
            'gaze_feature_dim': self.gaze_feature_dim,
            'fusion_method': self.fusion_method,
            'task_weight': self.task_weight,
            'gaze_alpha': self.gaze_alpha.item()
        }