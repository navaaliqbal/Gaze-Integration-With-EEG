"""
baseline_neurogate_fixed.py – Train NeuroGATE_Gaze (with EEG_CAM) without gaze.
Forward method now accepts `return_attention` (aliased to `return_cam`).
Run with: python baseline_neurogate_fixed.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.paths import PathConfig
from config.hyperparameters import get_hyp_for_integration
from data.dataloader_builder import get_dataloaders_fixed
from training.trainer import train_epoch
from training.metrics import evaluate_model_comprehensive
from training.early_stopping import EarlyStopping
from utils.debugger import DataDebugger
from utils.statistics_tracker import TrainingStatistics
from sklearn.metrics import classification_report

# ========== MODEL DEFINITION (with forward accepting return_attention) ==========

class EEG_CAM(nn.Module):
    """Class Activation Mapping for EEG signals (2D: channels × time)"""
    def __init__(self, in_channels, num_classes, eeg_channels=22):
        super(EEG_CAM, self).__init__()
        self.num_classes = num_classes
        self.eeg_channels = eeg_channels
        
        # Global Average Pooling over time
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Attention weights for each class
        self.class_fc = nn.Linear(in_channels, num_classes)
        
        # Channel-specific weights (for spatial attention)
        self.channel_fc = nn.Linear(in_channels, eeg_channels)
        
        # Initialize weights
        nn.init.normal_(self.class_fc.weight, 0, 0.01)
        nn.init.constant_(self.class_fc.bias, 0)
        nn.init.normal_(self.channel_fc.weight, 0, 0.01)
        nn.init.constant_(self.channel_fc.bias, 0)
        
    def forward(self, features, original_time_length=None):
        """
        features: (batch, feature_channels, reduced_time) from model
        Returns CAM aligned with original EEG dimensions
        """
        batch_size, feat_channels, feat_time = features.shape
        
        # Global average pooling
        gap_features = self.gap(features).squeeze(-1)  # (batch, feat_channels)
        
        # Get class-specific weights
        class_weights = self.class_fc(gap_features)  # (batch, num_classes)
        
        # Get channel-specific weights
        channel_weights = self.channel_fc(gap_features)  # (batch, eeg_channels)
        
        # Weighted sum of feature maps for each class
        cam_maps = []
        for c in range(self.num_classes):
            # Get weights for this class across batch
            class_weight = class_weights[:, c].unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
            
            # Weight features by class importance
            weighted_features = features * class_weight  # (batch, feat_channels, feat_time)
            
            # Sum across feature channels
            temporal_cam = torch.sum(weighted_features, dim=1)  # (batch, feat_time)
            
            # Add channel dimension using channel weights
            temporal_cam = temporal_cam.unsqueeze(1)  # (batch, 1, feat_time)
            channel_weight = channel_weights.unsqueeze(-1)  # (batch, eeg_channels, 1)
            
            # Combine temporal and spatial attention
            combined_cam = temporal_cam * channel_weight  # (batch, eeg_channels, feat_time)
            
            # Normalize
            combined_cam = (combined_cam - combined_cam.min(dim=2, keepdim=True)[0]) / \
                          (combined_cam.max(dim=2, keepdim=True)[0] - combined_cam.min(dim=2, keepdim=True)[0] + 1e-8)
            
            cam_maps.append(combined_cam)
        
        # Stack: (batch, num_classes, channels, time)
        cam_maps = torch.stack(cam_maps, dim=1)
        
        # If we need to resize to original EEG time length
        if original_time_length is not None and feat_time != original_time_length:
            cam_maps = F.interpolate(
                cam_maps.view(batch_size * self.num_classes * self.eeg_channels, 1, -1), 
                size=original_time_length, 
                mode='linear', 
                align_corners=False
            ).view(batch_size, self.num_classes, self.eeg_channels, original_time_length)
        
        return cam_maps, class_weights

class GateDilateLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super(GateDilateLayer, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, 
                             padding=self.padding, dilation=dilation)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.filter = nn.Conv1d(in_channels, in_channels, 1)
        self.gate = nn.Conv1d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)

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
        z = z[:, :, :-self.padding] if self.padding > 0 else z
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
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, 
                              kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, 
                              kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(16)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, input):
        x1 = F.relu(self.bn1(self.conv1(input)))
        x1 = torch.cat((x1, input), dim=1)
        x2 = F.relu(self.bn2(self.conv2(x1)))
        return torch.cat((x2, x1), dim=1)

class NeuroGATE_Gaze(nn.Module):
    def __init__(self, n_chan: int = 22, n_outputs: int = 2):
        super(NeuroGATE_Gaze, self).__init__()
        
        self.n_chan = n_chan
        self.n_outputs = n_outputs
        
        # Keep your original architecture up to encoder
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
        
        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(20, 4, dropout=0.5, batch_first=True), 2
        )
        
        # Add CAM layer
        self.cam_layer = EEG_CAM(in_channels=20, num_classes=n_outputs, eeg_channels=n_chan)
        
        # Final classification layer
        self.fc = nn.Linear(20, n_outputs)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x, return_attention=None, return_cam=None):
        """
        x: (batch_size, channels, time_steps) - Original EEG
        Accepts either return_attention (for trainer compatibility) or return_cam.
        If return_attention is provided, it overrides return_cam.
        """
        # Determine whether to return CAM maps
        if return_attention is not None:
            return_cam = return_attention
        # Default to False if neither is given
        if return_cam is None:
            return_cam = False

        batch_size, channels, original_time = x.shape
        
        # Your original forward pass until encoder
        x1 = F.avg_pool1d(x, kernel_size=5, stride=5)
        x2 = F.max_pool1d(x, kernel_size=5, stride=5)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn1(x)
        
        x1 = self.res_conv1(x)
        x2 = self.gate_dilate1(x)
        x = x1 + x2
        x = F.dropout2d(x, 0.5, training=self.training)
        x = F.max_pool1d(x, kernel_size=5, stride=5)
        
        x = F.relu(self.bn2(self.conv1(x)))
        
        x1 = self.res_conv2(x)
        x2 = self.gate_dilate2(x)
        x = x1 + x2
        x = self.bn3(self.conv2(x))
        
        x = self.res_conv3(x)
        x = F.max_pool1d(x, kernel_size=5, stride=5)
        
        x = self.bn4(self.conv3(x))
        
        # Pass through encoder
        x = x.permute(0, 2, 1)  # (batch, time, features)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)  # Back to (batch, features, time)
        
        # Store features for CAM
        features_for_cam = x
        
        # Final classification
        x_pooled = torch.mean(x, dim=2)  # Global average pooling
        logits = self.fc(x_pooled)
        
        if return_cam:
            # Generate CAM aligned with original EEG time
            cam_maps, _ = self.cam_layer(features_for_cam, original_time_length=original_time)
            
            # Ensure cam_maps has correct shape: (batch, classes, channels, time)
            if cam_maps.shape[2] != channels:
                # Resize channel dimension if needed
                cam_maps = F.interpolate(
                    cam_maps, 
                    size=(channels, original_time), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Return as tuple (logits, cam_maps) as expected by trainer_output
            return logits, cam_maps
        
        return logits

# ========== TRAINING FUNCTION ==========

def train_baseline(integration_type='output', output_suffix='baseline_fixed'):
    DataDebugger.print_header(f"BASELINE NEUROGATE (NO GAZE) – FIXED MODEL", width=80)

    # Setup directories
    PathConfig.setup_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get hyperparameters (we'll override gaze_weight)
    hyps = get_hyp_for_integration(integration_type)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_results_baseline_fixed_{timestamp}"
    stats_tracker = TrainingStatistics(output_dir=output_dir)
    print(f"Statistics will be saved to: {stats_tracker.run_dir}")

    # Build dataloaders (gaze maps are still loaded, but we won't use them)
    try:
        train_loader, eval_loader, gaze_stats = get_dataloaders_fixed(
            data_dir=PathConfig.DATA_DIR,
            batch_size=hyps.batch_size,
            seed=hyps.seed,
            target_length=hyps.target_length,
            gaze_json_dir=PathConfig.GAZE_JSON_DIR,
            only_matched=True,
            suffixes_to_strip=hyps.suffixes_to_strip,
            eeg_sampling_rate=hyps.eeg_sampling_rate,
            train_subdir=PathConfig.TRAIN_SUBDIR,
            eval_subdir=PathConfig.EVAL_SUBDIR
        )
    except Exception as e:
        print("Error building dataloaders:", e)
        traceback.print_exc()
        return None

    # Class distributions
    train_dist = stats_tracker.record_class_distribution(train_loader, "train")
    eval_dist = stats_tracker.record_class_distribution(eval_loader, "eval")
    print(f"  Train distribution: {dict(train_dist)}")
    print(f"  Eval distribution: {dict(eval_dist)}")

    # Determine number of channels
    try:
        sample_batch = next(iter(train_loader))
        n_chan = sample_batch['eeg'].shape[1]
        print(f"\nDetected {n_chan} channels from data")
    except:
        n_chan = hyps.n_channels
        print(f"\nUsing default {n_chan} channels")

    # Instantiate model
    model = NeuroGATE_Gaze(n_chan=n_chan, n_outputs=hyps.n_outputs).to(device)

    print(f"\nModel Configuration:")
    print(f"  Type: NeuroGATE_Gaze (with EEG_CAM, fixed forward)")
    print(f"  Channels: {n_chan}")
    print(f"  Outputs: {hyps.n_outputs}")

    optimizer = torch.optim.Adam(model.parameters(), lr=hyps.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, min_lr=1e-6
    )
    es_path = f"best_model_baseline_fixed.pth"
    es = EarlyStopping(patience=hyps.patience, path=es_path, verbose=True)

    # Test forward pass
    try:
        sample_batch = next(iter(train_loader))
        test_eeg = sample_batch['eeg'].to(device)[:2]
        # Test with return_attention=True (as trainer will do)
        outputs = model(test_eeg, return_attention=False)  # just logits
        print(f"\nModel forward OK (logits only), shape: {outputs.shape}")
        # Test with return_attention=True
        logits, cam = model(test_eeg, return_attention=True)
        print(f"Model forward with CAM OK, logits shape: {logits.shape}, CAM shape: {cam.shape}")
    except Exception as e:
        print("Model forward error:", e)
        traceback.print_exc()
        return None

    # ---- GAZE DISABLED ----
    gaze_weight = 0.0
    gaze_loss_scale = 1.0
    print(f"\n{'='*80}")
    print(f"GAZE WEIGHT SET TO 0.0 – GAZE LOSS DISABLED")
    print(f"{'='*80}")

    # Class weights (correct ordering by class index)
    num_classes = 2
    total = len(train_loader.dataset)
    class_weights = torch.tensor(
        [total / train_dist.get(i, 1) for i in range(num_classes)],
        dtype=torch.float32
    )
    print(f"Class weights (class0, class1): {class_weights.tolist()}")

    # Training loop
    best_acc = 0.0
    print(f"\nStarting training for {hyps.epochs} epochs...")
    print("=" * 80)

    for epoch in range(hyps.epochs):
        DataDebugger.print_header(f"EPOCH {epoch+1}/{hyps.epochs} [BASELINE FIXED]", width=60, char='-')

        train_stats = train_epoch(
            model, train_loader, optimizer, device,
            gaze_weight=gaze_weight,
            gaze_loss_type=hyps.gaze_loss_type,
            class_weights=class_weights,
            stats_tracker=stats_tracker,
            epoch=epoch,
            gaze_loss_scale=gaze_loss_scale
        )

        # Evaluate (no attention maps needed, but we can still pass return_attention=False)
        eval_stats, ev_labels, ev_preds, ev_files, _ = evaluate_model_comprehensive(
            model, eval_loader, device, stats_tracker, "eval",
            return_attention=False
        )

        scheduler.step(eval_stats['balanced_acc'])

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train: Loss={train_stats['loss']:.4f} | Acc={train_stats['acc']:.2f}%")
        print(f"  Cls Loss={train_stats['cls_loss']:.4f}")
        if 'gaze_loss' in train_stats and train_stats['gaze_loss'] > 0:
            print(f"         Gaze Loss={train_stats['gaze_loss']:.4f} (should be zero)")
        print(f"  Eval:  Acc={eval_stats['acc']:.2f}% | "
              f"Balanced Acc={eval_stats['balanced_acc']:.4f} | "
              f"Macro F1={eval_stats['macro_f1']:.4f}")
        print(f"  LR:    {train_stats['lr']:.2e}")

        report = classification_report(ev_labels, ev_preds, digits=4)
        print(f"\nClassification Report for Epoch {epoch+1}:\n{report}")

        # Save best model
        if eval_stats['acc'] > best_acc:
            best_acc = eval_stats['acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': eval_stats['acc'],
                'balanced_acc': eval_stats['balanced_acc'],
                'macro_f1': eval_stats['macro_f1'],
                'n_chan': n_chan
            }, es_path)
            print(f"  Saved best model at epoch {epoch+1} (acc {eval_stats['acc']:.2f}%)")

        # Early stopping
        metric_value = eval_stats['balanced_acc']
        es(metric_value, model, save_best_acc=True)
        if es.early_stop:
            print(f"  Early stopping triggered")
            break

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    final_stats, final_labels, final_preds, final_files, _ = evaluate_model_comprehensive(
        model, eval_loader, device, stats_tracker, "eval_final",
        return_attention=False
    )

    print(f"\nFinal Results:")
    print(f"  Accuracy: {final_stats['acc']:.2f}%")
    print(f"  Balanced Accuracy: {final_stats['balanced_acc']:.4f}")
    print(f"  Macro F1: {final_stats['macro_f1']:.4f}")

    stats_tracker.save_final_results(model=model)
    print(f"\nResults saved to: {stats_tracker.run_dir}")
    print(f"Baseline training complete! Best accuracy: {best_acc:.2f}%")

    return {
        'best_accuracy': best_acc,
        'final_stats': final_stats,
        'output_dir': str(stats_tracker.run_dir)
    }

if __name__ == "__main__":
    train_baseline(integration_type='output', output_suffix='nogaze_fixed')