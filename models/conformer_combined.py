"""
Conformer with BOTH input and output gaze integration (Combined approach)

This model integrates gaze at TWO levels:
1. INPUT: Gaze-as-gate modulation of EEG signals (like conformer_gaze_input)
2. OUTPUT: Attention map generation and alignment with gaze (like conformer_gaze_output)

Similar to NeuroGATE_Combined but using Conformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from EEGConformer.conformer import Conformer, PatchEmbedding, TransformerEncoder
from einops.layers.torch import Reduce


class ConformerAttentionModule(nn.Module):
    """
    Generate attention map from Conformer transformer features.
    Adapted from EEG_Attention for Conformer's patch-based features.
    """
    def __init__(self, emb_size=30, n_channels=22, original_time_length=6000):
        super().__init__()
        self.n_channels = n_channels
        self.original_time_length = original_time_length
        self.emb_size = emb_size
        
        # Temporal attention from patch embeddings
        # Input: [B, num_patches, emb_size]
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(emb_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention (channel-wise)
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(emb_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_channels),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        Generate attention map from transformer features.
        
        Args:
            features: [B, num_patches, emb_size] from transformer encoder
            
        Returns:
            attention_map: [B, n_channels, original_time_length]
        """
        batch_size = features.shape[0]
        
        # Transpose for Conv1d: [B, num_patches, emb_size] -> [B, emb_size, num_patches]
        features_t = features.transpose(1, 2)
        
        # Temporal attention: [B, emb_size, num_patches] -> [B, 1, num_patches]
        temporal_att = self.temporal_attention(features_t)  # [B, 1, num_patches]
        
        # Spatial attention: [B, emb_size, num_patches] -> [B, n_channels]
        spatial_att = self.spatial_attention(features_t)  # [B, n_channels]
        
        # Combine spatial and temporal: [B, n_channels, num_patches]
        # temporal_att: [B, 1, num_patches] -> broadcast to [B, n_channels, num_patches]
        # spatial_att: [B, n_channels] -> [B, n_channels, 1] -> broadcast
        combined = temporal_att * spatial_att.unsqueeze(-1)  # [B, n_channels, num_patches]
        
        # Upsample to original time length
        attention_map = F.interpolate(
            combined,
            size=self.original_time_length,
            mode='linear',
            align_corners=False
        )  # [B, n_channels, original_time_length]
        
        # Clamp to [0, 1]
        attention_map = torch.clamp(attention_map, 0, 1)
        
        return attention_map


class Conformer_Combined(nn.Module):
    """
    Conformer with BOTH input and output gaze integration.
    
    Integration levels:
    1. INPUT: Gaze modulates EEG signals (gaze-as-gate)
       eeg_modulated = eeg * (1 + gaze_alpha * gaze)
       
    2. OUTPUT: Generate attention map from features and use for classification
       attention_map = AttentionModule(transformer_features)
       attended_features = features * attention_map
    """
    
    def __init__(self, emb_size=30, depth=4, n_classes=2, n_channels=22, 
                 original_time_length=6000, dropout_rate=0.2):
        """
        Args:
            emb_size: Transformer embedding dimension (default: 30)
            depth: Number of transformer layers (default: 4)
            n_classes: Number of output classes (default: 2)
            n_channels: Number of EEG channels (default: 22)
            original_time_length: Expected time length (default: 6000)
            dropout_rate: Dropout rate (default: 0.2)
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.original_time_length = original_time_length
        self.emb_size = emb_size
        self.depth = depth
        self.dropout_rate = dropout_rate
        
        # Learnable gaze strength for INPUT modulation
        # Initialized to 0.1 to start with subtle modulation
        self.gaze_alpha = nn.Parameter(torch.tensor(0.1))
        
        # Conformer components
        self.patch_embedding = PatchEmbedding(emb_size, n_channels=n_channels)
        self.transformer_encoder = TransformerEncoder(depth, emb_size)
        
        # Attention generator for OUTPUT integration
        self.attention_module = ConformerAttentionModule(
            emb_size=emb_size,
            n_channels=n_channels,
            original_time_length=original_time_length
        )
        
        # Classification head with attention weighting
        self.classifier = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_size, n_classes)
        )
        
    def forward(self, eeg, gaze=None, return_attention=False):
        """
        Forward pass with combined gaze integration.
        
        Args:
            eeg: [B, C, T] or [B, 1, C, T] - EEG signals
            gaze: [B, C, T] - Gaze attention maps (optional)
            return_attention: If True, return dict with attention_map
            
        Returns:
            If return_attention=False: (features, logits) tuple
            If return_attention=True: dict with 'logits', 'attention_map', 'features'
        """
        # ====================================================
        # STEP 1: INPUT INTEGRATION (GAZE-AS-GATE)
        # ====================================================
        # Modulate EEG with gaze before any processing
        # Formula: eeg_modulated = eeg * (1 + alpha * gaze)
        if gaze is not None:
            # Handle 4D input: [B, 1, C, T] -> [B, C, T]
            if eeg.ndim == 4:
                eeg_squeezed = eeg.squeeze(1)
            else:
                eeg_squeezed = eeg
                
            # Apply input modulation
            eeg_squeezed = eeg_squeezed * (1.0 + self.gaze_alpha * gaze)
            
            # Restore 4D if needed
            if eeg.ndim == 4:
                eeg = eeg_squeezed.unsqueeze(1)
            else:
                eeg = eeg_squeezed
        
        # ====================================================
        # STEP 2: CONFORMER ARCHITECTURE
        # ====================================================
        # Handle both 3D [B, C, T] and 4D [B, 1, C, T] inputs
        if eeg.ndim == 3:
            eeg = eeg.unsqueeze(1)  # [B, C, T] -> [B, 1, C, T]
        
        batch_size = eeg.shape[0]
        
        # Patch embedding: [B, 1, n_channels, time] -> [B, num_patches, emb_size]
        patches = self.patch_embedding(eeg)
        
        # Transformer encoding: [B, num_patches, emb_size] -> [B, num_patches, emb_size]
        encoded_features = self.transformer_encoder(patches)
        
        # ====================================================
        # STEP 3: OUTPUT INTEGRATION (ATTENTION-BASED)
        # ====================================================
        # Generate attention map: [B, num_patches, emb_size] -> [B, n_channels, time]
        attention_map = self.attention_module(encoded_features)
        
        # Apply attention to original (modulated) input
        # eeg: [B, 1, n_channels, time], attention_map: [B, n_channels, time]
        eeg_squeezed = eeg.squeeze(1)  # [B, n_channels, time]
        attended_input = eeg_squeezed * attention_map  # Element-wise multiplication
        
        # Re-encode attended input through patch embedding
        attended_input = attended_input.unsqueeze(1)  # [B, 1, n_channels, time]
        attended_patches = self.patch_embedding(attended_input)
        
        # Re-encode through transformer
        attended_features = self.transformer_encoder(attended_patches)
        
        # ====================================================
        # STEP 4: CLASSIFICATION
        # ====================================================
        # Classify using attended features
        logits = self.classifier(attended_features)
        
        # Global average pooled features for compatibility
        features = attended_features.mean(dim=1)  # [B, emb_size]
        
        if return_attention:
            return {
                'logits': logits,
                'attention_map': attention_map,
                'features': features
            }
        else:
            return features, logits
    
    def get_config(self):
        """Get model configuration"""
        return {
            'model': 'Conformer_Combined',
            'gaze_integration': 'both',
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'original_time_length': self.original_time_length,
            'emb_size': self.emb_size,
            'depth': self.depth,
            'gaze_alpha': self.gaze_alpha.item()
        }
    
    def get_attention(self, eeg, gaze=None):
        """Convenience method to get attention map only"""
        with torch.no_grad():
            outputs = self.forward(eeg, gaze, return_attention=True)
            return outputs['attention_map']
    
    def get_features(self, eeg, gaze=None):
        """Get intermediate features"""
        with torch.no_grad():
            outputs = self.forward(eeg, gaze, return_attention=True)
            return outputs['features']
    
    def get_attention_map(self, eeg, gaze=None):
        """Helper to get only the attention map (alias for get_attention)"""
        return self.get_attention(eeg, gaze)
