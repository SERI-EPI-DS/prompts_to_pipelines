#############################################
# train.py – Fine‑tune Swin‑V2‑B with AMP & memory‑savvy options
#############################################
"""Train a Swin‑V2‑B classifier on a fundus‑photo ImageFolder dataset.
Adds automatic mixed‑precision (AMP), optional gradient accumulation and
configurable input resolution to mitigate GPU OOM on 24‑GB cards.

Example
-------
python train.py \
    --data_dir /main/data \
    --output_dir /main/project/results \
    --epochs 50 \
    --batch_size 16 \
    --img_size 256  # smaller crop saves memory
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import Swin_V2_B_Weights

# -----------------------------------------------------------------------------
# Constants – fall back to hard‑coded ImageNet stats if weight metadata missing
# -----------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _get_norm_stats() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    try:
        meta = Swin_V2_B_Weights.IMAGENET1K_V1.meta  # type: ignore[attr-defined]
        return meta.get("mean", IMAGENET_MEAN), meta.get("std", IMAGENET_STD)
    except AttributeError:
        return IMAGENET_MEAN, IMAGENET_STD


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------


def build_dataloaders(
    data_dir: Path, batch_size: int, img_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    """Return train & validation DataLoaders with chosen *img_size*."""
    train_dir, val_dir = data_dir / "train", data_dir / "val"
    mean, std = _get_norm_stats()

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation(
                20, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.05)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dl, val_dl


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, amp: bool
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )
            with autocast(enabled=amp):
                logits = model(imgs)
                loss = criterion(logits, targets)
            loss_sum += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return correct / total, loss_sum / total


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser("Fine‑tune Swin‑V2‑B with AMP")
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per‑GPU batch size (will fit with AMP)",
    )
    p.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="Input crop size; reduce to 256 or 224 to save VRAM",
    )
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--accum_steps", type=int, default=1, help="Gradient accumulation steps"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--no_amp", action="store_true", help="Disable automatic mixed precision"
    )
    args = p.parse_args()

    use_amp = not args.no_amp
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Data ----------------
    train_loader, val_loader = build_dataloaders(
        args.data_dir, args.batch_size, args.img_size, args.num_workers
    )
    num_classes = len(train_loader.dataset.classes)

    # ---------------- Model ----------------
    model = models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, num_classes)
    model.to(device)

    # Gradient checkpointing (saves memory ~25% at extra compute cost)
    try:
        model.gradient_checkpointing_enable()
    except AttributeError:
        pass  # older torchvision

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    best_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (imgs, targets) in enumerate(train_loader, start=1):
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )
            with autocast(enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, targets) / args.accum_steps
            scaler.scale(loss).backward()

            if step % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset) if running_loss else 0.0
        val_acc, val_loss = evaluate(model, val_loader, device, use_amp)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "classes": train_loader.dataset.classes,
                "val_acc": val_acc,
            }
            torch.save(ckpt, args.output_dir / "best_swin_v2_b.pth")
        print(f"[Epoch {epoch:02d}] val_acc={val_acc:.4f} val_loss={val_loss:.4f}")

    with open(args.output_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training done – best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
