#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvNeXt-L fine-tuning script for colour fundus classification.

- PyTorch 2.3+ / TorchVision 0.18+
- Single-GPU (RTX 3090, 24GB) friendly
- Cross-Entropy with label smoothing
- AMP mixed precision + GradScaler
- Cosine LR with warmup (<= 50 epochs)
- Saves best checkpoint and class mapping
- CLI args for data root and results dir

Folder shape:
data_root/
  train/<class>/*.png|jpg
  val/<class>/*.png|jpg

Outputs:
results_dir/
  checkpoints/best_model.pt
  classes.json
  train_log.csv
  last_epoch.pt (optional latest snapshot)

Author: ChatGPT (GPT-5 Thinking)
"""
import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

# --------- Defaults / constants ----------
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    data_root: str
    results_dir: str
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    lr_min: float = 1e-6
    img_size: int = 224
    num_workers: int = 8
    seed: int = 42
    stochastic_depth_prob: float = 0.1
    clip_grad: float = 1.0
    freeze_backbone: bool = False
    compile: bool = False


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reproducible-ish while keeping speed high
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_transforms(img_size: int):
    # Robust ImageNet-style augments; retinal images are large so RandomResizedCrop is fine.
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
            ),
        ]
    )
    # Standard eval preprocessing
    short_side = int((256 / 224) * img_size)
    eval_tf = transforms.Compose(
        [
            transforms.Resize(short_side, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    return train_tf, eval_tf


def create_dataloaders(
    data_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
):
    train_tf, eval_tf = build_transforms(img_size)
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return train_ds, val_ds, train_loader, val_loader


def build_model(
    num_classes: int, stochastic_depth_prob: float = 0.1, freeze_backbone: bool = False
):
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
    model = convnext_large(weights=weights, stochastic_depth_prob=stochastic_depth_prob)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False

    return model


@torch.inference_mode(False)
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")  # to compute mean manually
    total_loss = 0.0
    correct = 0
    n = 0
    scaler_dtype = torch.float16  # use FP16 for speed
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=scaler_dtype):
                logits = model(images)
                loss = ce(logits, targets)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            n += targets.numel()
    return total_loss / max(1, n), correct / max(1, n)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    label_smoothing: float = 0.1,
    clip_grad: float = 1.0,
) -> Tuple[float, float]:
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="mean")
    total_loss = 0.0
    correct = 0
    n = 0
    scaler_dtype = torch.float16

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=scaler_dtype):
            logits = model(images)
            loss = ce(logits, targets)
        scaler.scale(loss).backward()
        if clip_grad is not None and clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        n += targets.size(0)

    return total_loss / max(1, n), correct / max(1, n)


def build_schedulers(
    optimizer: optim.Optimizer, epochs: int, warmup_epochs: int, lr_min: float
):
    if warmup_epochs > 0:
        warmup = LambdaLR(
            optimizer, lr_lambda=lambda e: (e + 1) / max(1, warmup_epochs)
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=lr_min
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    return scheduler


def save_checkpoint(state: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Fine-tune ConvNeXt-L on fundus images")
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder containing train/ val/ test/",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stochastic_depth_prob", type=float, default=0.1)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument(
        "--compile", action="store_true", help="Use torch.compile to speed up training"
    )
    args = p.parse_args()
    return TrainConfig(**vars(args))


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, training on CPU (will be slow).")

    # Data
    train_ds, val_ds, train_loader, val_loader = create_dataloaders(
        cfg.data_root,
        cfg.img_size,
        cfg.batch_size,
        cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    classes = train_ds.classes
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")
    print(f"Train images: {len(train_ds)}, Val images: {len(val_ds)}")

    # Model
    model = build_model(num_classes, cfg.stochastic_depth_prob, cfg.freeze_backbone)
    if cfg.compile:
        try:
            model = torch.compile(model)  # PyTorch 2.3+
            print("Compiled model with torch.compile()")
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}); continuing without compile.")
    model.to(device)

    # Optimizer & schedulers
    # Only optimize trainable params (e.g., if backbone is frozen)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999)
    )
    scheduler = build_schedulers(optimizer, cfg.epochs, cfg.warmup_epochs, cfg.lr_min)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # IO setup
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = results_dir / "checkpoints"
    log_csv_path = results_dir / "train_log.csv"
    classes_json = results_dir / "classes.json"

    # Persist class mapping (sorted by folder name as in ImageFolder)
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    with open(classes_json, "w") as f:
        json.dump(
            {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2
        )

    # Write CSV header
    with open(log_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = 0.0
    best_epoch = -1
    t0 = time.time()

    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        # Train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            label_smoothing=cfg.label_smoothing,
            clip_grad=cfg.clip_grad,
        )
        # Val
        val_loss, val_acc = evaluate(model, val_loader, device)

        # Log
        lr_now = optimizer.param_groups[0]["lr"]
        with open(log_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch + 1, lr_now, train_loss, train_acc, val_loss, val_acc])

        print(
            f"  lr={lr_now:.6f} | train_loss={train_loss:.4f} acc={train_acc*100:.2f}% "
            f"| val_loss={val_loss:.4f} acc={val_acc*100:.2f}%"
        )

        # Save latest snapshot
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "scaler_state": scaler.state_dict(),
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "img_size": cfg.img_size,
                "mean": IMAGENET_DEFAULT_MEAN,
                "std": IMAGENET_DEFAULT_STD,
                "config": asdict(cfg),
            },
            ckpt_dir / "last_epoch.pt",
        )

        # Track & save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "img_size": cfg.img_size,
                    "mean": IMAGENET_DEFAULT_MEAN,
                    "std": IMAGENET_DEFAULT_STD,
                    "config": asdict(cfg),
                },
                ckpt_dir / "best_model.pt",
            )
            print(f"  ✔ Saved new best checkpoint (val_acc={best_val_acc*100:.2f}%)")

        # Step scheduler after epoch
        if scheduler is not None:
            scheduler.step()

    dt = time.time() - t0
    print(
        f"\nTraining finished in {dt/60:.1f} min. Best val acc {best_val_acc*100:.2f}% at epoch {best_epoch}."
    )
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Class mapping saved to: {classes_json}")
    print(f"Log CSV saved to: {log_csv_path}")


if __name__ == "__main__":
    main()
