"""
Main entry point for training NeuroGATE / EEGNet with (optional) gaze integration

Supports:
- NeuroGATE: input | output | both
- EEGNet:    none  | input  | output | both

Usage examples at bottom.
"""
import os
import sys
import torch
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.paths import PathConfig
from config.hyperparameters import get_hyp_for_integration
from data.dataloader_builder import get_dataloaders_fixed

# ✅ UPDATED: use unified factory (must exist in models/model_factory.py)
from models.model_factory import create_model

from training.trainer import train_epoch
from training.metrics import evaluate_model_comprehensive, compute_gaze_loss_scale
from training.early_stopping import EarlyStopping
from utils.debugger import DataDebugger
from utils.statistics_tracker import TrainingStatistics
from sklearn.metrics import classification_report


def _validate_args(arch: str, integration_type: str):
    arch = arch.lower()
    integration_type = integration_type.lower()

    if arch not in ["neurogate", "eegnet"]:
        raise ValueError("arch must be one of: neurogate, eegnet")

    if arch == "neurogate":
        if integration_type not in ["input", "output", "both"]:
            raise ValueError("For neurogate, --type must be: input | output | both")
    else:
        # EEGNet supports EEG-only baseline
        if integration_type not in ["none", "input", "output", "both"]:
            raise ValueError("For eegnet, --type must be: none | input | output | both")

    return arch, integration_type


def train_integration_approach(arch="neurogate", integration_type="output", output_suffix=None):
    """
    Train a specific architecture + gaze integration approach.

    Args:
        arch: 'neurogate' or 'eegnet'
        integration_type: for neurogate: 'input','output','both'
                          for eegnet:    'none','input','output','both'
        output_suffix: optional suffix for output directory
    """
    arch, integration_type = _validate_args(arch, integration_type)

    title = f"{arch.upper()} | GAZE INTEGRATION: {integration_type.upper()}"
    DataDebugger.print_header(title, width=90)

    # Setup directories
    PathConfig.setup_directories()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters
    # NOTE: Your get_hyp_for_integration currently expects 'input/output/both'.
    # For EEGNet 'none', we reuse output hyps safely.
    hyp_key = integration_type if integration_type != "none" else "output"
    hyps = get_hyp_for_integration(hyp_key)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_results_{arch}_{integration_type}_{timestamp}"
    if output_suffix:
        output_dir = f"{output_dir}_{output_suffix}"

    # Stats tracker
    stats_tracker = TrainingStatistics(output_dir=output_dir)
    print(f"Statistics will be saved to: {stats_tracker.run_dir}")

    # Build dataloaders
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

    # Record class distributions
    print("\nRecording initial class distributions...")
    train_dist = stats_tracker.record_class_distribution(train_loader, "train")
    eval_dist = stats_tracker.record_class_distribution(eval_loader, "eval")
    print(f"  Train distribution: {dict(train_dist)}")
    print(f"  Eval distribution: {dict(eval_dist)}")

    # Detect n_chan
    try:
        sample_batch = next(iter(train_loader))
        n_chan = sample_batch["eeg"].shape[1]
        print(f"\nDetected {n_chan} channels from data")
    except Exception:
        n_chan = hyps.n_channels
        print(f"\nUsing default {n_chan} channels")

    # ✅ Create model
    model = create_model(
        arch=arch,
        integration_type=integration_type,
        n_chan=n_chan,
        n_outputs=hyps.n_outputs,
        original_time_length=hyps.original_time_length,
    ).to(device)

    print(f"\nModel Configuration:")
    print(f"  Arch: {arch}")
    print(f"  Type: {integration_type}")
    print(f"  Channels: {n_chan}")
    print(f"  Outputs: {hyps.n_outputs}")
    if hasattr(model, "gaze_alpha"):
        print(f"  Initial gaze alpha: {model.gaze_alpha.item():.3f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=hyps.learning_rate)

    # Scheduler: 'max' because for eval_loss we pass -loss (higher is better)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.1
    )

    # Early stopping
    es_path = f"best_model_{arch}_{integration_type}.pth"
    es = EarlyStopping(patience=hyps.patience, path=es_path, verbose=True)

    print(f"\nEarly stopping and checkpointing will monitor: {hyps.early_stop_metric}")
    if hyps.early_stop_metric == "eval_loss":
        print("  -> Using validation loss (recommended for detecting overfitting)")

    # -------------------------------------------------------
    # Test forward pass
    # -------------------------------------------------------
    try:
        sample_batch = next(iter(train_loader))
        test_eeg = sample_batch["eeg"].to(device)[:2]
        has_gaze = ("gaze" in sample_batch) and (sample_batch["gaze"] is not None)
        test_gaze = sample_batch["gaze"].to(device)[:2] if has_gaze else None

        if arch == "eegnet":
            if integration_type == "none":
                logits = model(test_eeg)
            elif integration_type == "input":
                logits = model(test_eeg, test_gaze)
            elif integration_type == "output":
                out = model(test_eeg, return_attention=True)
                logits = out["logits"] if isinstance(out, dict) else out
            elif integration_type == "both":
                out = model(test_eeg, test_gaze, return_attention=True)
                logits = out["logits"] if isinstance(out, dict) else out
        else:
            # NeuroGATE
            if integration_type == "input":
                logits = model(test_eeg, test_gaze)
            elif integration_type == "output":
                out = model(test_eeg, return_attention=True)
                logits = out["logits"] if isinstance(out, dict) else out
            elif integration_type == "both":
                out = model(test_eeg, test_gaze, return_attention=True)
                logits = out["logits"] if isinstance(out, dict) else out

        print(f"\nModel forward OK, logits shape: {logits.shape}")
    except Exception as e:
        print("Model forward error:", e)
        traceback.print_exc()
        return None

    # -------------------------------------------------------
    # Gaze-loss scaling (only meaningful when training with gaze loss)
    # -------------------------------------------------------
    uses_gaze_loss = integration_type in ["output", "both"]  # for both archs
    if uses_gaze_loss and hyps.use_gaze_loss_scaling:
        gaze_loss_scale, scale_metrics = compute_gaze_loss_scale(
            model, train_loader, device, hyps.gaze_loss_type
        )
        print("\n" + "=" * 80)
        print(f"FIXED SCALING FACTOR FOR ENTIRE TRAINING: {gaze_loss_scale:.2f}")
        print(f"Effective gaze loss = gaze_weight × gaze_loss_scale × gaze_loss_raw")
        print(f"With gaze_weight={hyps.gaze_weight:.2f}: Effective scale = {hyps.gaze_weight * gaze_loss_scale:.2f}")
        print("=" * 80)
    else:
        gaze_loss_scale = 1.0
        print("\n" + "=" * 80)
        if not uses_gaze_loss:
            print("No gaze-loss used for this mode (type=none or type=input).")
        print(f"Using gaze_loss_scale: {gaze_loss_scale:.2f}")
        print("=" * 80)

    # -------------------------------------------------------
    # Best metric tracking
    # -------------------------------------------------------
    if hyps.early_stop_metric == "eval_loss":
        best_metric = float("inf")  # lower is better
        metric_name = "Validation Loss"
        print(f"\n{'='*80}")
        print(f"MONITORING: {metric_name} (minimize)")
        print(f"{'='*80}")
    elif hyps.early_stop_metric == "balanced_acc":
        best_metric = 0.0
        metric_name = "Balanced Accuracy"
        print(f"\nMonitoring: {metric_name}")
    else:
        best_metric = 0.0
        metric_name = "Macro F1"
        print(f"\nMonitoring: {metric_name}")

    # Class weights
    class_counts = list(train_dist.values())
    total = sum(class_counts) if len(class_counts) else 1
    class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float32).to(device)
    print(f"\nClass weights for loss: {class_weights.tolist()}")
    print(f"Best model checkpoint saved when {metric_name} improves")

    # -------------------------------------------------------
    # Training loop
    # -------------------------------------------------------
    print(f"\nStarting training for {hyps.epochs} epochs...")
    print("=" * 80)

    for epoch in range(hyps.epochs):
        DataDebugger.print_header(
            f"EPOCH {epoch+1}/{hyps.epochs} [{arch.upper()} | {integration_type.upper()}]",
            width=70,
            char="-"
        )

        train_stats = train_epoch(
            model, train_loader, optimizer, device,
            gaze_weight=hyps.gaze_weight,
            gaze_loss_type=hyps.gaze_loss_type,
            class_weights=class_weights,
            stats_tracker=stats_tracker,
            epoch=epoch,
            gaze_loss_scale=gaze_loss_scale
        )

        # Return attention during eval only if model supports it
        return_attention = integration_type in ["output", "both"]

        if return_attention:
            eval_stats, ev_labels, ev_preds, ev_files, ev_attention_maps = evaluate_model_comprehensive(
                model, eval_loader, device, stats_tracker, "eval",
                return_attention=True,
                gaze_weight=hyps.gaze_weight,
                gaze_loss_type=hyps.gaze_loss_type,
                class_weights=class_weights,
                gaze_loss_scale=gaze_loss_scale
            )
        else:
            eval_stats, ev_labels, ev_preds, ev_files = evaluate_model_comprehensive(
                model, eval_loader, device, stats_tracker, "eval",
                return_attention=False,
                gaze_weight=hyps.gaze_weight,
                gaze_loss_type=hyps.gaze_loss_type,
                class_weights=class_weights,
                gaze_loss_scale=gaze_loss_scale
            )

        # Scheduler metric
        if hyps.early_stop_metric == "eval_loss":
            metric_for_sched = -eval_stats["loss"]
        else:
            metric_for_sched = eval_stats["balanced_acc"] if hyps.early_stop_metric == "balanced_acc" else eval_stats["macro_f1"]
        scheduler.step(metric_for_sched)

        # Record epoch
        stats_tracker.record_epoch(epoch, train_stats, eval_stats, model)

        # Print summary
        print(f"\nEpoch {epoch+1} Summary [{arch.upper()} | {integration_type.upper()}]:")
        print(f"  Train: Loss={train_stats['loss']:.4f} | Acc={train_stats['acc']:.2f}%")
        if "cls_loss" in train_stats:
            print(f"        Cls Loss={train_stats['cls_loss']:.4f}")
        if "gaze_loss" in train_stats and train_stats["gaze_loss"] > 0:
            print(f"        Gaze Loss={train_stats['gaze_loss']:.4f}")
        if "gaze_alpha" in train_stats:
            print(f"        Gaze Alpha={train_stats['gaze_alpha']:.3f}")

        print(
            f"  Eval:  Loss={eval_stats['loss']:.4f} | Acc={eval_stats['acc']:.2f}% | "
            f"Balanced Acc={eval_stats['balanced_acc']:.4f} | Macro F1={eval_stats['macro_f1']:.4f}"
        )
        print(f"  Gaze:  {train_stats.get('gaze_samples', 0)}/{train_stats.get('total_samples', 0)} samples")
        print(f"  LR:    {train_stats['lr']:.2e}")

        report = classification_report(ev_labels, ev_preds, digits=4, zero_division=0)
        print(f"\nClassification Report for Epoch {epoch+1}:\n{report}")

        # Early-stopping metric
        if hyps.early_stop_metric == "eval_loss":
            metric_value = -eval_stats["loss"]  # higher is better
            current_metric = eval_stats["loss"]
            is_better = current_metric < best_metric
        elif hyps.early_stop_metric == "balanced_acc":
            metric_value = eval_stats["balanced_acc"]
            current_metric = eval_stats["balanced_acc"]
            is_better = current_metric > best_metric
        else:
            metric_value = eval_stats["macro_f1"]
            current_metric = eval_stats["macro_f1"]
            is_better = current_metric > best_metric

        if is_better:
            best_metric = current_metric

        es(metric_value, model, save_best_acc=True)
        # if es.early_stop:
        #     print("Early stopping triggered.")
        #     break

    # -------------------------------------------------------
    # Load best model
    # -------------------------------------------------------
    try:
        if os.path.exists(es_path):
            model.load_state_dict(torch.load(es_path, map_location=device))
            print(f"\nLoaded best model from {es_path}")
            print(f"  Best {metric_name}: {best_metric:.4f}")
    except Exception as e:
        print("Could not load best model:", e)

    # -------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION [{arch.upper()} | {integration_type.upper()}]")
    print("=" * 80)

    if return_attention:
        final_stats, final_labels, final_preds, final_files, final_attention_maps = evaluate_model_comprehensive(
            model, eval_loader, device, stats_tracker, "eval_final",
            return_attention=True,
            gaze_weight=hyps.gaze_weight,
            gaze_loss_type=hyps.gaze_loss_type,
            class_weights=class_weights,
            gaze_loss_scale=gaze_loss_scale
        )
    else:
        final_stats, final_labels, final_preds, final_files = evaluate_model_comprehensive(
            model, eval_loader, device, stats_tracker, "eval_final",
            return_attention=False,
            gaze_weight=hyps.gaze_weight,
            gaze_loss_type=hyps.gaze_loss_type,
            class_weights=class_weights,
            gaze_loss_scale=gaze_loss_scale
        )

    print(f"\nFinal Results [{arch.upper()} | {integration_type.upper()}]:")
    print(f"  Accuracy: {final_stats['acc']:.2f}%")
    print(f"  Balanced Accuracy: {final_stats['balanced_acc']:.4f}")
    print(f"  Macro F1: {final_stats['macro_f1']:.4f}")
    print(f"  Weighted F1: {final_stats['weighted_f1']:.4f}")
    print(f"  Precision: {final_stats['precision']:.4f}")
    print(f"  Recall: {final_stats['recall']:.4f}")

    # Save results
    print("\n" + "=" * 80)
    print(f"SAVING RESULTS [{arch.upper()} | {integration_type.upper()}]")
    print("=" * 80)

    stats_tracker.save_final_results(model=model)

    # Summary report
    with open(stats_tracker.run_dir / "training_summary.txt", "w") as f:
        f.write(f"TRAINING SUMMARY REPORT - {arch.upper()} | {integration_type.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Architecture: {arch}\n")
        f.write(f"Integration type: {integration_type}\n")
        f.write(f"Total epochs: {len(stats_tracker.epoch_stats)}\n")
        f.write(f"Best {metric_name}: {best_metric:.4f}\n")
        f.write(f"\nHyperparameters:\n")
        f.write(f"  Learning rate: {hyps.learning_rate}\n")
        f.write(f"  Batch size: {hyps.batch_size}\n")
        f.write(f"  Gaze weight: {hyps.gaze_weight}\n")
        f.write(f"  Gaze loss type: {hyps.gaze_loss_type}\n")
        f.write(f"  Early stop metric: {hyps.early_stop_metric}\n")
        f.write(f"\nDataset Statistics:\n")
        f.write(f"  Train samples: {len(train_loader.dataset)}\n")
        f.write(f"  Eval samples: {len(eval_loader.dataset)}\n")
        f.write(f"  Train class distribution: {dict(train_dist)}\n")
        f.write(f"  Eval class distribution: {dict(eval_dist)}\n")
        f.write(f"\nModel Statistics:\n")
        f.write(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"  Accuracy: {final_stats['acc']:.2f}%\n")
        f.write(f"  Balanced Accuracy: {final_stats['balanced_acc']:.4f}\n")
        f.write(f"  Macro F1: {final_stats['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1: {final_stats['weighted_f1']:.4f}\n")
        f.write(f"\nResults saved to: {stats_tracker.run_dir}\n")

    print(f"\nTraining complete! [{arch.upper()} | {integration_type.upper()}]")
    print(f"Best {metric_name}: {best_metric:.4f}")
    print(f"Results saved to: {stats_tracker.run_dir}")

    return {
        "arch": arch,
        "integration_type": integration_type,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "final_stats": final_stats,
        "output_dir": str(stats_tracker.run_dir),
    }


def main(arch="neurogate", integration_type="output", output_suffix=None):
    arch, integration_type = _validate_args(arch, integration_type)
    return train_integration_approach(arch=arch, integration_type=integration_type, output_suffix=output_suffix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train NeuroGATE / EEGNet with optional gaze integration")
    parser.add_argument("--arch", type=str, default="neurogate",
                        choices=["neurogate", "eegnet"],
                        help="Model architecture")
    parser.add_argument("--type", type=str, default="output",
                        choices=["none", "input", "output", "both"],
                        help="Gaze integration type (EEGNet supports 'none')")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Suffix for output directory")

    args = parser.parse_args()
    main(arch=args.arch, integration_type=args.type, output_suffix=args.suffix)