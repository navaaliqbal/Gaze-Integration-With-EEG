"""
Conformer with OUTPUT gaze integration using ACTUAL transformer attention weights.
This directly supervises what the model uses for classification.

IMPROVEMENTS:
1. Layer Aggregation Strategies:
   - 'learnable': Learnable weighted combination (adds depth parameters)
     Pro: Adaptive, can learn importance of each layer
     Con: More parameters, risk of overfitting
   
   - 'last_layer': Use only the deepest transformer layer
     Pro: No extra parameters, deeper layers are more task-specific
     Con: Ignores potentially useful information from earlier layers
   
   - 'uniform': Simple average across all layers
     Pro: No extra parameters, stable, benefits from all layers
     Con: Treats all layers equally, may dilute task-specific signals

2. Receptive Field Mapping:
   - Precise: Computes exact receptive field of each patch based on:
     * Conv kernel size (25)
     * Pooling kernel (75) and stride (15)
     * Maps patches to their exact temporal coverage in original signal
   
   - Interpolation: Simple linear upsampling (original method)
     * Faster but less precise
     * May blur boundaries between patches

RECOMMENDATION:
- Start with 'uniform' + precise_receptive_field for baseline (no extra params)
- Try 'last_layer' if you want deeper task-specific attention
- Use 'learnable' only if you have enough data and want to tune per-layer importance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from EEGConformer.conformer import PatchEmbedding, ClassificationHead
from einops import rearrange


class TransformerEncoderWithAttention(nn.Module):
    """
    Transformer encoder that returns attention weights for supervision.
    """
    def __init__(self, depth, emb_size, num_heads=10, drop_p=0.45):
        super().__init__()
        self.depth = depth
        self.emb_size = emb_size
        self.num_heads = num_heads
        
        # Create transformer blocks that return attention
        self.blocks = nn.ModuleList([
            TransformerBlockWithAttention(emb_size, num_heads, drop_p)
            for _ in range(depth)
        ])
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: [B, num_patches, emb_size]
            return_attention: If True, return attention weights
            
        Returns:
            features: [B, num_patches, emb_size]
            attentions: [B, depth, num_heads, num_patches, num_patches] if return_attention
        """
        all_attentions = []
        
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                all_attentions.append(attn)
            else:
                x = block(x, return_attention=False)
        
        if return_attention:
            # Stack attention weights: [depth, B, num_heads, patches, patches]
            attentions = torch.stack(all_attentions, dim=1)  # [B, depth, num_heads, patches, patches]
            return x, attentions
        else:
            return x


class TransformerBlockWithAttention(nn.Module):
    """Single transformer block that returns attention weights."""
    def __init__(self, emb_size, num_heads=10, drop_p=0.45, forward_expansion=4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=drop_p,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(forward_expansion * emb_size, emb_size),
        )
        self.dropout = nn.Dropout(drop_p)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: [B, num_patches, emb_size]
            
        Returns:
            x: [B, num_patches, emb_size]
            attn_weights: [B, num_heads, num_patches, num_patches] if return_attention
        """
        # Self-attention with residual
        normed = self.norm1(x)
        
        if return_attention:
            attn_output, attn_weights = self.attention(
                normed, normed, normed, 
                need_weights=True,
                average_attn_weights=False  # Keep per-head weights
            )
            x = x + self.dropout(attn_output)
        else:
            attn_output, _ = self.attention(normed, normed, normed, need_weights=False)
            x = x + self.dropout(attn_output)
            attn_weights = None
        
        # Feed-forward with residual
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        
        if return_attention:
            return x, attn_weights
        else:
            return x


class ConformerGazeOutputV2(nn.Module):
    """
    Conformer with direct transformer attention supervision.
    Extracts actual attention weights and aligns with gaze.
    """
    def __init__(self, emb_size=30, depth=4, n_classes=2, n_channels=22, 
                 original_time_length=6000, num_heads=5, 
                 layer_aggregation='learnable', use_precise_receptive_field=True):
        """
        Args:
            layer_aggregation: How to combine attention across layers:
                - 'learnable': learnable weighted combination (adds parameters)
                - 'last_layer': use only the last transformer layer (deepest, most task-specific)
                - 'uniform': simple average across all layers (no parameters)
            use_precise_receptive_field: If True, compute exact receptive field mapping
                                        from patches to original temporal positions
        """
        super().__init__()
        self.n_channels = n_channels
        self.original_time_length = original_time_length
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.num_heads = num_heads
        self.depth = depth
        self.layer_aggregation = layer_aggregation
        self.use_precise_receptive_field = use_precise_receptive_field
        
        # Conformer components
        self.patch_embedding = PatchEmbedding(emb_size, n_channels=n_channels)
        self.transformer_encoder = TransformerEncoderWithAttention(depth, emb_size, num_heads)
        self.classifier = ClassificationHead(emb_size, n_classes)
        
        # Learnable aggregation weights for layers (only if using learnable mode)
        if layer_aggregation == 'learnable':
            self.layer_weights = nn.Parameter(torch.ones(depth) / depth)
        else:
            self.register_buffer('layer_weights', None)
        
        # Channel attention module for generating channel-specific attention
        # This learns to modulate temporal attention per channel
        self.channel_attention = nn.Sequential(
            nn.Linear(emb_size, n_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_channels * 2, n_channels),
            nn.Sigmoid()  # Output [0, 1] importance per channel
        )
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: [B, 1, n_channels, time_length] or [B, n_channels, time_length]
            return_attention: If True, return attention-based spatial map
            
        Returns:
            If return_attention=False: (features, logits) tuple
            If return_attention=True: dict with 'logits', 'attention_map', 'features'
        """
        # Handle input shape
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        batch_size = x.shape[0]
        
        # Patch embedding: [B, 1, C, T] → [B, num_patches, emb_size]
        patches = self.patch_embedding(x)
        num_patches = patches.shape[1]
        
        # Transformer encoding
        if return_attention:
            features, attentions = self.transformer_encoder(patches, return_attention=True)
            # attentions: [B, depth, num_heads, num_patches, num_patches]
            
            # Convert attention weights to spatial map with channel-specific weighting
            attention_map = self._attention_to_spatial_map(
                attentions, x.shape, pooled_features=patches.mean(dim=1)  # Use mean patch features
            )
        else:
            features = self.transformer_encoder(patches, return_attention=False)
            attention_map = None
        
        # Classification
        pooled_features, logits = self.classifier(features)
        
        if return_attention:
            return {
                'logits': logits,
                'attention_map': attention_map,
                'features': pooled_features,
                'raw_attentions': attentions  # For debugging
            }
        else:
            return pooled_features, logits
    
    def _attention_to_spatial_map(self, attentions, input_shape, pooled_features=None):
        """
        Convert patch-level attention weights to channel-temporal attention map.
        Now with channel-specific attention weighting.
        
        Args:
            attentions: [B, depth, num_heads, num_patches, num_patches]
            input_shape: [B, 1, C, T]
            pooled_features: [B, emb_size] optional features for channel attention
            
        Returns:
            spatial_map: [B, C, T] aligned with input EEG with channel-specific weights
        """
        B, depth, num_heads, num_patches, _ = attentions.shape
        _, _, C, T = input_shape
        
        # Average attention TO each patch (what each patch attends to)
        # Sum over the query dimension to get "importance" of each key patch
        patch_importance = attentions.mean(dim=2).sum(dim=3)  # [B, depth, num_patches]
        
        # Aggregate across layers based on strategy
        if self.layer_aggregation == 'learnable':
            # Learnable weighted combination
            layer_weights = F.softmax(self.layer_weights, dim=0)
            patch_importance = (patch_importance * layer_weights.view(1, depth, 1)).sum(dim=1)
        elif self.layer_aggregation == 'last_layer':
            # Use only the last (deepest) layer - most task-specific
            patch_importance = patch_importance[:, -1, :]  # [B, num_patches]
        elif self.layer_aggregation == 'uniform':
            # Simple average - no learnable parameters
            patch_importance = patch_importance.mean(dim=1)  # [B, num_patches]
        else:
            raise ValueError(f"Unknown layer_aggregation: {self.layer_aggregation}")
        
        # Normalize to [0, 1]
        patch_importance = patch_importance / (patch_importance.sum(dim=1, keepdim=True) + 1e-8)
        
        # Map patch importance to spatial dimensions [B, C, T]
        if self.use_precise_receptive_field:
            # Compute exact receptive field mapping
            temporal_attention = self._map_patches_to_time_precise(
                patch_importance, num_patches, T
            )
        else:
            # Simple interpolation (original approach)
            temporal_attention = self._map_patches_to_time_interpolate(
                patch_importance, T
            )
        
        # Generate channel-specific attention weights using learned module
        if pooled_features is not None:
            channel_weights = self.channel_attention(pooled_features)  # [B, C]
        else:
            # Fallback: uniform weights
            channel_weights = torch.ones(B, C, device=temporal_attention.device) / C
        
        # Apply channel-specific weighting instead of uniform broadcasting
        # temporal_attention: [B, 1, T], channel_weights: [B, C]
        spatial_map = temporal_attention * channel_weights.unsqueeze(2)  # [B, C, T]
        
        # Normalize using sum (matching dataset normalization)
        spatial_map_sum = spatial_map.sum(dim=(1, 2), keepdim=True)
        spatial_map = spatial_map / (spatial_map_sum + 1e-8)
        
        return spatial_map
    
    def _map_patches_to_time_interpolate(self, patch_importance, T):
        """
        Simple interpolation-based upsampling (original method).
        
        Args:
            patch_importance: [B, num_patches]
            T: target time length
            
        Returns:
            temporal_attention: [B, 1, T]
        """
        patch_importance_2d = patch_importance.unsqueeze(1)  # [B, 1, num_patches]
        
        temporal_attention = F.interpolate(
            patch_importance_2d,
            size=T,
            mode='linear',
            align_corners=False
        )  # [B, 1, T]
        
        return temporal_attention
    
    def _map_patches_to_time_precise(self, patch_importance, num_patches, T):
        """
        Precise receptive field-based mapping from patches to time.
        
        The patch embedding architecture:
        1. Conv2d(1, 40, (1, 25), (1, 1)) - temporal conv, kernel=25, stride=1
        2. Conv2d(40, 40, (C, 1), (1, 1)) - spatial conv (channel collapse)
        3. AvgPool2d((1, 75), (1, 15)) - temporal pooling, kernel=75, stride=15
        
        Each patch at position i corresponds to temporal range:
        - After first conv: positions cover [i*15, i*15 + 25)
        - After pooling: aggregate over [i*15, i*15 + 75)
        - Effective receptive field: [i*15, i*15 + 75 + 24] = [i*15, i*15 + 99)
        
        Args:
            patch_importance: [B, num_patches]
            num_patches: number of patches
            T: original time length
            
        Returns:
            temporal_attention: [B, 1, T]
        """
        B = patch_importance.shape[0]
        device = patch_importance.device
        
        # PatchEmbedding architecture parameters
        conv_kernel = 25  # first temporal conv kernel
        pool_kernel = 75  # pooling kernel
        pool_stride = 15  # pooling stride
        
        # Effective receptive field for each patch
        receptive_field = conv_kernel + pool_kernel - 1  # = 99
        
        # Initialize temporal attention map
        temporal_attention = torch.zeros(B, 1, T, device=device)
        
        # Map each patch to its receptive field in the original time dimension
        for i in range(num_patches):
            # Compute receptive field boundaries
            start_idx = i * pool_stride
            end_idx = min(start_idx + receptive_field, T)
            
            # Distribute patch importance uniformly over its receptive field
            # (or use a weighted distribution like Gaussian)
            if end_idx > start_idx:
                temporal_attention[:, 0, start_idx:end_idx] += \
                    patch_importance[:, i].unsqueeze(-1) / (end_idx - start_idx)
        
        return temporal_attention


def compute_attention_gaze_loss(attention_map, gaze_map, labels=None, loss_type='cosine'):
    """
    Compute loss between model attention and human gaze.
    
    Args:
        attention_map: [B, C, T] model attention (from transformer)
        gaze_map: [B, C, T] human gaze fixations
        labels: [B] class labels (optional, for class-specific loss)
        loss_type: 'cosine', 'mse', 'kl'
        
    Returns:
        loss: scalar tensor
    """
    B, C, T = attention_map.shape
    
    # Flatten spatial dimensions
    attn_flat = attention_map.view(B, -1)  # [B, C*T]
    gaze_flat = gaze_map.view(B, -1)
    
    if loss_type == 'cosine':
        # Cosine similarity loss (1 - similarity)
        # Normalize vectors
        attn_norm = F.normalize(attn_flat, p=2, dim=1)
        gaze_norm = F.normalize(gaze_flat, p=2, dim=1)
        
        # Compute cosine similarity
        cosine_sim = (attn_norm * gaze_norm).sum(dim=1)  # [B]
        
        # Loss: 1 - similarity (want to maximize similarity)
        loss = (1 - cosine_sim).mean()
        
    elif loss_type == 'mse':
        # Mean squared error
        loss = F.mse_loss(attn_flat, gaze_flat)
        
    elif loss_type == 'kl':
        # KL divergence (treat as distributions)
        # Normalize to probability distributions
        attn_prob = attn_flat / (attn_flat.sum(dim=1, keepdim=True) + 1e-8)
        gaze_prob = gaze_flat / (gaze_flat.sum(dim=1, keepdim=True) + 1e-8)
        
        # KL(gaze || attention) - want attention to match gaze
        loss = F.kl_div(
            attn_prob.log(),
            gaze_prob,
            reduction='batchmean'
        )
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss


# Example usage:
if __name__ == "__main__":
    print("Testing different layer aggregation strategies...")
    
    for aggregation in ['learnable', 'last_layer', 'uniform']:
        print(f"\n=== Testing {aggregation} aggregation ===")
        
        # Test model
        model = ConformerGazeOutputV2(
            emb_size=30,
            depth=4,
            n_classes=2,
            n_channels=22,
            original_time_length=6000,
            num_heads=5,
            layer_aggregation=aggregation,
            use_precise_receptive_field=True
        )
        
        # Test input
        x = torch.randn(2, 22, 6000)
        gaze = torch.randn(2, 22, 6000)
        
        # Forward with attention
        output = model(x, return_attention=True)
        print(f"Logits: {output['logits'].shape}")
        print(f"Attention map: {output['attention_map'].shape}")
        print(f"Features: {output['features'].shape}")
        
        # Compute gaze loss
        loss = compute_attention_gaze_loss(
            output['attention_map'],
            gaze,
            loss_type='cosine'
        )
        print(f"Gaze loss: {loss.item():.4f}")
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")
    
    print("\n=== Comparing precise vs interpolation receptive field ===")
    for precise in [True, False]:
        model = ConformerGazeOutputV2(
            emb_size=30, depth=4, n_classes=2, n_channels=22,
            original_time_length=6000, num_heads=5,
            layer_aggregation='uniform',
            use_precise_receptive_field=precise
        )
        
        x = torch.randn(1, 22, 6000)
        output = model(x, return_attention=True)
        attn = output['attention_map']
        
        method = "Precise RF" if precise else "Interpolation"
        print(f"{method}: min={attn.min():.4f}, max={attn.max():.4f}, "
              f"mean={attn.mean():.4f}, std={attn.std():.4f}")
