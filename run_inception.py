import os
import sys
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.paths import PathConfig
from config.hyperparameters import get_hyp_for_integration
from data.dataloader_builder import get_dataloaders_fixed

# If you keep the model in models/inception_time.py, import from there
from models.inception_time import (
    InceptionTimeBase,
    InceptionTimeGazeInput,
    InceptionTimeGazeOutput,
    InceptionTimeGazeBoth
)


# -----------------------------
# Helpers
# -----------------------------

def ensure_gaze_shape(gaze: torch.Tensor, target_channels: int, target_time: int) -> torch.Tensor:
    """
    Dataset returns gaze as (C, T) per-sample; DataLoader batches -> (B, C, T) already.
    But we still guard:
      (B, T), (B, 1, T), (B, C, T)
    Returns (B, target_channels, target_time)
    """
    if gaze is None:
        return None

    if gaze.dim() == 2:
        # (B, T) -> (B, 1, T)
        gaze = gaze.unsqueeze(1)
    elif gaze.dim() != 3:
        raise ValueError(f"Unsupported gaze shape: {tuple(gaze.shape)}")

    # Fix time dim
    if gaze.shape[-1] != target_time:
        gaze = F.interpolate(gaze, size=target_time, mode="linear", align_corners=False)

    # Fix channels dim
    if gaze.shape[1] == 1 and target_channels > 1:
        gaze = gaze.repeat(1, target_channels, 1)
    elif gaze.shape[1] != target_channels:
        # In your pipeline gaze_map is built to match EEG channels, so this shouldn't happen.
        raise ValueError(
            f"Gaze channels ({gaze.shape[1]}) != EEG channels ({target_channels}). "
            "Expected gaze to be (B, 22, T) or (B, 1, T)."
        )

    return gaze


def attention_loss(att_map, gaze_map, kind="bce"):
    """
    att_map, gaze_map: (B, C, T)
    Gaze map from dataset is already 0..1 normalized (mostly), but normalize defensively.
    """
    gaze_min = gaze_map.amin(dim=-1, keepdim=True)
    gaze_max = gaze_map.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    gaze_norm = (gaze_map - gaze_min) / gaze_max

    if kind == "bce":
        return F.binary_cross_entropy(att_map.clamp(1e-6, 1 - 1e-6), gaze_norm)
    elif kind == "mse":
        return F.mse_loss(att_map, gaze_norm)
    else:
        raise ValueError("kind must be 'bce' or 'mse'")


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    gaze_weight: float
    max_grad_norm: float = 1.0


def build_model(mode: str, input_shape, nb_classes: int, nb_filters=32, depth=6, kernel_size=41,
                use_residual=True, use_bottleneck=True):
    if mode == "none":
        return InceptionTimeBase(
            input_shape=input_shape, nb_classes=nb_classes,
            nb_filters=nb_filters, depth=depth, kernel_size=kernel_size,
            use_residual=use_residual, use_bottleneck=use_bottleneck
        )
    if mode == "input":
        return InceptionTimeGazeInput(
            input_shape=input_shape, nb_classes=nb_classes,
            nb_filters=nb_filters, depth=depth, kernel_size=kernel_size,
            use_residual=use_residual, use_bottleneck=use_bottleneck
        )
    if mode == "output":
        return InceptionTimeGazeOutput(
            input_shape=input_shape, nb_classes=nb_classes,
            nb_filters=nb_filters, depth=depth, kernel_size=kernel_size,
            use_residual=use_residual, use_bottleneck=use_bottleneck
        )
    if mode == "both":
        return InceptionTimeGazeBoth(
            input_shape=input_shape, nb_classes=nb_classes,
            nb_filters=nb_filters, depth=depth, kernel_size=kernel_size,
            use_residual=use_residual, use_bottleneck=use_bottleneck
        )
    raise ValueError(f"Unknown type: {mode}")


# -----------------------------
# Train / Eval
# -----------------------------

def train_one_epoch(model, loader, optimizer, device, mode: str, cfg: TrainConfig, att_kind="bce"):
    model.train()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_ce = 0.0
    total_att = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Training[{mode}]", leave=False)
    for batch in pbar:
        eeg = batch["eeg"].to(device)               # (B, C, T)
        y = batch["label"]
        # Your dataset returns 'label' as int; DataLoader makes it a list/tensor
        y = torch.as_tensor(y, device=device, dtype=torch.long)

        gaze = batch.get("gaze", None)
        gaze = gaze.to(device) if gaze is not None else None

        optimizer.zero_grad(set_to_none=True)

        if mode == "none":
            logits = model(eeg)
            loss = ce(logits, y)

        elif mode == "input":
            logits = model(eeg, gaze=gaze)
            loss = ce(logits, y)

        elif mode == "output":
            if gaze is not None:
                out = model(eeg, return_attention=True)
                logits, att = out["logits"], out["attention_map"]
                gaze_map = ensure_gaze_shape(gaze, model.n_chans, model.n_times)
                att_l = attention_loss(att, gaze_map, kind=att_kind)
                loss = ce(logits, y) + cfg.gaze_weight * att_l
                total_att += att_l.item()
            else:
                logits = model(eeg, return_attention=False)
                loss = ce(logits, y)

        elif mode == "both":
            if gaze is not None:
                out = model(eeg, gaze=gaze, return_attention=True)
                logits, att = out["logits"], out["attention_map"]
                gaze_map = ensure_gaze_shape(gaze, model.n_chans, model.n_times)
                att_l = attention_loss(att, gaze_map, kind=att_kind)
                loss = ce(logits, y) + cfg.gaze_weight * att_l
                total_att += att_l.item()
            else:
                out = model(eeg, gaze=None, return_attention=False)
                logits = out if not isinstance(out, dict) else out["logits"]
                loss = ce(logits, y)

        else:
            raise ValueError(mode)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce(logits, y).item()

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / max(1, total):.2f}%")

    n = max(1, len(loader))
    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "att": (total_att / n) if mode in ("output", "both") else 0.0,
        "acc": 100.0 * correct / max(1, total),
    }


@torch.no_grad()
def evaluate(model, loader, device, mode: str):
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        eeg = batch["eeg"].to(device)
        y = batch["label"]
        y = torch.as_tensor(y, device=device, dtype=torch.long)

        if mode == "none":
            logits = model(eeg)
        elif mode == "input":
            logits = model(eeg, gaze=None)
        elif mode == "output":
            out = model(eeg, return_attention=False)
            logits = out if not isinstance(out, dict) else out["logits"]
        elif mode == "both":
            out = model(eeg, gaze=None, return_attention=False)
            logits = out if not isinstance(out, dict) else out["logits"]
        else:
            raise ValueError(mode)

        loss = ce(logits, y)
        total_loss += loss.item()

        pred = logits.argmax(dim=1)
        all_preds.extend(pred.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds) * 100.0
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "preds": all_preds,
        "labels": all_labels
    }


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Run InceptionTime with EEG-only or gaze integration")
    parser.add_argument("--type", default="none", choices=["none", "input", "output", "both"])
    parser.add_argument("--att_kind", default="bce", choices=["bce", "mse"])
    parser.add_argument("--save", default=None, help="Path to save best model (.pth)")
    args = parser.parse_args()

    mode = args.type

    print("=" * 80)
    print(f"INCEPTION TIME TRAINING | MODE: {mode.upper()}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Your hyperparams object (you already use this)
    hyps = get_hyp_for_integration("output")  # use as base config

    # Build dataloaders using YOUR pipeline
    train_loader, val_loader, meta = get_dataloaders_fixed(
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

    # Peek a batch for shape
    sample = next(iter(train_loader))
    eeg = sample["eeg"]
    gaze = sample.get("gaze", None)

    if eeg.dim() != 3:
        raise RuntimeError(f"Expected eeg batch shape (B,C,T), got {tuple(eeg.shape)}")

    n_chans, n_times = eeg.shape[1], eeg.shape[2]
    print("\nDATA SHAPES")
    print(f"  EEG batch:  {tuple(eeg.shape)}  (B,C,T)")
    if gaze is not None:
        print(f"  Gaze batch: {tuple(gaze.shape)} (should be B,C,T)")
    else:
        print("  Gaze batch: None")

    nb_classes = 2  # your pipeline labels are 0/1; change if needed

    print("\nCREATING MODEL")
    model = build_model(
        mode=mode,
        input_shape=(n_chans, n_times),
        nb_classes=nb_classes,
        nb_filters=32,
        depth=6,
        kernel_size=41,
        use_residual=True,
        use_bottleneck=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    cfg = TrainConfig(
        epochs=hyps.epochs,
        lr=hyps.learning_rate,
        gaze_weight=getattr(hyps, "gaze_weight", 0.1) if mode in ("output", "both") else 0.0
    )

    if args.save is None:
        best_path = f"best_inception_{mode}.pth"
    else:
        best_path = args.save

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    best_val_acc = -1.0

    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        print("-" * 60)

        train_stats = train_one_epoch(model, train_loader, optimizer, device, mode, cfg, att_kind=args.att_kind)
        val_stats = evaluate(model, val_loader, device, mode)

        scheduler.step(val_stats["loss"])
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Train: loss={train_stats['loss']:.4f} (ce={train_stats['ce']:.4f}, att={train_stats['att']:.4f}) "
            f"acc={train_stats['acc']:.2f}%"
        )
        print(
            f" Val:  loss={val_stats['loss']:.4f} acc={val_stats['accuracy']:.2f}% "
            f"bal_acc={val_stats['balanced_accuracy']:.4f} f1={val_stats['f1']:.4f}"
        )
        print(f" LR: {lr_now:.2e}")

        if val_stats["accuracy"] > best_val_acc:
            best_val_acc = val_stats["accuracy"]
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved best model to {best_path} (val acc {best_val_acc:.2f}%)")

    print("\n" + "=" * 80)
    print("FINAL EVALUATION (BEST MODEL)")
    print("=" * 80)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best model: {best_path}")

    final_stats = evaluate(model, val_loader, device, mode)
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print("Classification report:")
    print(classification_report(final_stats["labels"], final_stats["preds"], digits=4))

    print("\nDONE.")


if __name__ == "__main__":
    main()