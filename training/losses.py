"""
Loss functions for gaze-guided attention training
"""
import torch
import torch.nn.functional as F

def compute_gaze_attention_loss(attention_map, gaze, labels, loss_type='mse'):
    """Compute loss between attention maps and gaze maps."""
    if attention_map is None or gaze is None:
        return torch.tensor(0.0).to(attention_map.device if attention_map is not None else gaze.device)
    
    if loss_type == 'mse':
        return F.mse_loss(attention_map, gaze)
    elif loss_type == 'weighted_mse':
        weights = gaze * 2 + 0.1
        return (weights * (attention_map - gaze) ** 2).mean()
    elif loss_type == 'cosine':
        att = attention_map.view(attention_map.shape[0], -1)
        gz = gaze.view(gaze.shape[0], -1)
        return 1 - F.cosine_similarity(att, gz).mean()
    elif loss_type == 'kl':
        att_prob = F.softmax(attention_map.view(attention_map.shape[0], -1), dim=1)
        gaze_prob = F.softmax(gaze.view(gaze.shape[0], -1), dim=1)
        return F.kl_div(att_prob.log(), gaze_prob, reduction='batchmean')
    elif loss_type == 'combined':
        # Combine multiple losses
        mse = F.mse_loss(attention_map, gaze)
        att_flat = attention_map.view(attention_map.shape[0], -1)
        gaze_flat = gaze.view(gaze.shape[0], -1)
        cosine = 1 - F.cosine_similarity(att_flat, gaze_flat).mean()
        return mse + 0.5 * cosine
    else:
        raise ValueError(f"Unknown gaze loss type: {loss_type}")