"""
Loss functions for gaze-guided attention training
"""
import torch
import torch.nn.functional as F

def compute_gaze_attention_loss(attention_map, gaze, labels, loss_type='mse'):
    """
    Compute loss between attention maps and gaze maps.
    
    Args:
        attention_map: (batch, channels, time) - predicted attention
        gaze: (batch, channels, time) - ground truth gaze
        labels: (batch,) - class labels (unused in basic losses)
        loss_type: type of loss to compute
    """
    if attention_map is None or gaze is None:
        return torch.tensor(0.0, device=attention_map.device if attention_map is not None else gaze.device)
    
    # Ensure same shape
    if attention_map.shape != gaze.shape:
        # Try to align shapes
        if attention_map.dim() == 3 and gaze.dim() == 3:
            if attention_map.shape[1] != gaze.shape[1]:
                # Average over channels if needed
                if attention_map.shape[1] > 1:
                    attention_map = attention_map.mean(dim=1, keepdim=True)
                if gaze.shape[1] > 1:
                    gaze = gaze.mean(dim=1, keepdim=True)
    
    if loss_type == 'mse':
        return F.mse_loss(attention_map, gaze)
    
    elif loss_type == 'weighted_mse':
        # Weight based on gaze intensity
        weights = gaze * 2 + 0.1
        return (weights * (attention_map - gaze) ** 2).mean()
    
    elif loss_type == 'cosine':
        # Cosine similarity loss
        att_flat = attention_map.reshape(attention_map.shape[0], -1)
        gaze_flat = gaze.reshape(gaze.shape[0], -1)
        return 1 - F.cosine_similarity(att_flat, gaze_flat).mean()
    
    elif loss_type == 'kl':
        # KL divergence (treat as distributions)
        att_prob = F.softmax(attention_map.reshape(attention_map.shape[0], -1), dim=1)
        gaze_prob = F.softmax(gaze.reshape(gaze.shape[0], -1), dim=1)
        return F.kl_div(att_prob.log(), gaze_prob, reduction='batchmean')
    
    elif loss_type == 'combined':
        # Combine MSE and cosine
        mse = F.mse_loss(attention_map, gaze)
        att_flat = attention_map.reshape(attention_map.shape[0], -1)
        gaze_flat = gaze.reshape(gaze.shape[0], -1)
        cosine = 1 - F.cosine_similarity(att_flat, gaze_flat).mean()
        return mse + 0.5 * cosine
    elif loss_type == 'combined_2':
        # Combine multiple losses - best for sum-normalized distributions
        # 1. Cosine similarity for direction
        att_flat = attention_map.view(attention_map.shape[0], -1)
        gaze_flat = gaze.view(gaze.shape[0], -1)
        cosine = 1 - F.cosine_similarity(att_flat, gaze_flat).mean()
        
        # 2. KL divergence for distribution matching
        att_flat_norm = (att_flat + 1e-8) / (att_flat.sum(dim=1, keepdim=True) + 1e-8)
        gaze_flat_norm = (gaze_flat + 1e-8) / (gaze_flat.sum(dim=1, keepdim=True) + 1e-8)
        kl = F.kl_div(att_flat_norm.log(), gaze_flat_norm, reduction='batchmean')
        
        return 0.5 * cosine + 0.5 * kl
    else:
        raise ValueError(f"Unknown gaze loss type: {loss_type}")