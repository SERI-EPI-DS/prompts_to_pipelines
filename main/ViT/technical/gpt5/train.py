#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune Swin-V2-B on fundus photographs.

Env: Python 3.11, PyTorch 2.3.1, torchvision 0.18.1, CUDA 12.1
Single GPU (RTX 3090, 24GB)

Features:
- ImageFolder train/val with strong but sane augmentations
- Label smoothing CE
- AdamW + cosine decay with linear warm-up
- AMP mixed precision
- Optional backbone freeze warmup
- Early stopping
- Saves: best.pt (best val acc), last.pt, history.csv, class_mapping.json, run_config.json
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from torchvision.transforms.functional import InterpolationMode
import pandas as pd


# --------------------------- Utils ---------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism can slow things; we prefer speed + repeatability
    cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imagenet_stats_from_weights(weights) -> Tuple[List[float], List[float]]:
    """Robustly extract (mean, std) from weights; fallback to standard ImageNet stats."""
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    if weights is None:
        return default_mean, default_std
    try:
        # torchvision >= 0.13: Normalize in the transform pipeline
        tfms = weights.transforms()
        for t in reversed(tfms.transforms):
            if isinstance(t, T.Normalize):
                return list(t.mean), list(t.std)
    except Exception:
        pass
    try:
        m = weights.meta
        if "mean" in m and "std" in m:
            return list(m["mean"]), list(m["std"])
    except Exception:
        pass
    return default_mean, default_std


def build_transforms(image_size: int, weights, use_aug: bool = True):
    mean, std = imagenet_stats_from_weights(weights)
    if use_aug:
        train_tfms = T.Compose(
            [
                T.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.RandAugment(num_ops=2, magnitude=7),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                T.RandomErasing(p=0.25, scale=(0.02, 0.2), value=0),
            ]
        )
    else:
        train_tfms = T.Compose(
            [
                T.Resize(
                    int(image_size * 1.15),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    eval_tfms = T.Compose(
        [
            T.Resize(
                int(image_size * 1.15),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    return train_tfms, eval_tfms, (mean, std)


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        return correct / max(1, target.size(0))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_acc: float,
    class_to_idx: Dict[str, int],
    meta: Dict,
):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_acc": best_acc,
        "class_to_idx": class_to_idx,
        "meta": meta,
    }
    torch.save(ckpt, path)


# --------------------------- Training / Eval ---------------------------


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, max_norm: float = 1.0
):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        if max_norm is not None and max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(outputs, labels) * bs
        n += bs

    return running_loss / max(1, n), running_acc / max(1, n)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, labels)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(outputs, labels) * bs
        n += bs

    return running_loss / max(1, n), running_acc / max(1, n)


# --------------------------- Main ---------------------------


def main():
    ap = argparse.ArgumentParser(description="Fine-tune Swin-V2-B on fundus images")
    ap.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root folder containing train/ val/ test/",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output folder for checkpoints and results",
    )
    ap.add_argument(
        "--epochs", type=int, default=50, help="Max epochs (will be capped at 50)"
    )
    ap.add_argument("--batch-size", type=int, default=32, help="Global batch size")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    ap.add_argument(
        "--image-size", type=int, default=256, help="Input image size for Swin-V2-B"
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Base learning rate (will scale with batch size/32)",
    )
    ap.add_argument(
        "--weight-decay", type=float, default=0.05, help="AdamW weight decay"
    )
    ap.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="CrossEntropy label smoothing",
    )
    ap.add_argument(
        "--warmup-epochs", type=int, default=5, help="Linear warm-up epochs"
    )
    ap.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help="Freeze backbone for N initial epochs",
    )
    ap.add_argument(
        "--early-stop",
        type=int,
        default=10,
        help="Early stop patience (epochs without val acc improvement)",
    )
    ap.add_argument(
        "--no-aug", action="store_true", help="Disable heavy training augmentation"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pretrained",
        action="store_true",
        help="Start from ImageNet pretrained weights",
    )
    ap.add_argument(
        "--compile", action="store_true", help="Use torch.compile for model"
    )
    args = ap.parse_args()

    # Hard cap epochs at 50
    args.epochs = min(args.epochs, 50)

    set_seed(args.seed)
    device = get_device()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_root / "train"
    val_dir = data_root / "val"

    # Weights / transforms
    weights = Swin_V2_B_Weights.IMAGENET1K_V1 if args.pretrained else None
    train_tfms, eval_tfms, (mean, std) = build_transforms(
        args.image_size, weights, use_aug=not args.no_aug
    )

    # Datasets / Loaders
    train_ds = ImageFolder(root=str(train_dir), transform=train_tfms)
    val_ds = ImageFolder(root=str(val_dir), transform=eval_tfms)

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    if num_classes < 2:
        raise ValueError("Need at least 2 classes in train/ to fine-tune a classifier.")

    bs = args.batch_size
    lr = args.lr * (bs / 32.0)  # simple linear scaling rule

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )

    # Model
    model = swin_v2_b(weights=weights)
    in_feats = model.head.in_features
    model.head = nn.Linear(in_feats, num_classes)

    if args.compile:
        # torch.compile can speed up training on 2.3+, but toggle if issues
        model = torch.compile(model)

    model = model.to(device)

    # Loss / Optimizer / Schedulers
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    warmup_epochs = max(0, min(args.warmup_epochs, args.epochs))
    cosine_epochs = max(1, args.epochs - warmup_epochs)

    warmup = (
        LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / max(1, warmup_epochs))
        if warmup_epochs > 0
        else None
    )
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=lr * 1e-2)
    scheduler = (
        SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        if warmup is not None
        else cosine
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Optionally freeze backbone for a few warm-up epochs (train head only)
    frozen = False
    if args.freeze_epochs > 0:
        for name, p in model.named_parameters():
            if not name.startswith("head."):
                p.requires_grad = False
        frozen = True

    # Logs
    history_rows = []
    best_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    t0 = time.time()

    # Write mapping & config
    with open(out_dir / "class_mapping.json", "w") as f:
        json.dump(
            {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2
        )

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "mean": mean,
                "std": std,
                "num_classes": num_classes,
                "classes": [idx_to_class[i] for i in range(num_classes)],
            },
            f,
            indent=2,
        )

    for epoch in range(args.epochs):
        # Unfreeze after freeze-epochs
        if frozen and epoch >= args.freeze_epochs:
            for p in model.parameters():
                p.requires_grad = True
            frozen = False

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, max_norm=1.0
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        history_rows.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"lr {optimizer.param_groups[0]['lr']:.3e} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # Save last checkpoint every epoch
        meta = {
            "mean": mean,
            "std": std,
            "image_size": args.image_size,
            "classes": [idx_to_class[i] for i in range(num_classes)],
            "model_name": "swin_v2_b",
        }
        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            scaler,
            epoch,
            best_acc,
            class_to_idx,
            meta,
        )

        # Best checkpoint on val acc
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_acc,
                class_to_idx,
                meta,
            )
        else:
            epochs_no_improve += 1

        # Early stopping
        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(
                f"Early stopping triggered (no val acc improvement for {args.early_stop} epochs)."
            )
            break

    # Save training history
    pd.DataFrame(history_rows).to_csv(out_dir / "history.csv", index=False)
    elapsed = time.time() - t0
    print(
        f"Training done in {elapsed/60:.1f} min. Best val acc {best_acc:.4f} at epoch {best_epoch+1}."
    )


if __name__ == "__main__":
    main()
