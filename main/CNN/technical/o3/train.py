# train.py
#!/usr/bin/env python3
"""
Fine‑tunes a ConvNeXt‑Large model on colour fundus photographs.

Example
-------
python train.py --data_dir /main/data --output_dir /main/project/results
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ConvNeXt_Large_Weights, convnext_large

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Fine‑tune ConvNeXt‑Large on fundus photographs")

    # Paths
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root containing train/val/test folders",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where checkpoints & logs are written",
    )

    # Optimisation
    p.add_argument(
        "--epochs", type=int, default=50, help="Max training epochs (default=50)"
    )
    p.add_argument(
        "--batch_size", type=int, default=8, help="Mini‑batch size (default=8)"
    )
    p.add_argument(
        "--lr", type=float, default=1e-4, help="Initial learning rate (default=1e‑4)"
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="AdamW weight decay (default=0.05)",
    )

    # Misc
    p.add_argument(
        "--image_size", type=int, default=512, help="Input resolution (default=512)"
    )
    p.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader workers (default=4)"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    return p.parse_args()


# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Top‑1 accuracy for one mini‑batch."""
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()


# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------


def build_dataloaders(root: Path, image_size: int, batch_size: int, num_workers: int):
    """Creates training & validation DataLoaders."""

    # NOTE: In torchvision 0.18 the ConvNeXt weights' meta does **not** expose
    # mean/std, so we fallback to the canonical ImageNet values defined above.
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_ds = datasets.ImageFolder(root / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(root / "val", transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, train_ds.classes, train_ds.class_to_idx


# -----------------------------------------------------------------------------
# TRAINING & EVAL
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
):
    model.train()
    epoch_loss = epoch_acc = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * images.size(0)
        epoch_acc += accuracy(logits.detach(), targets) * images.size(0)

    n = len(loader.dataset)
    return epoch_loss / n, epoch_acc / n


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
):
    model.eval()
    epoch_loss = epoch_acc = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        epoch_loss += loss.item() * images.size(0)
        epoch_acc += accuracy(logits, targets) * images.size(0)

    n = len(loader.dataset)
    return epoch_loss / n, epoch_acc / n


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, classes, class_to_idx = build_dataloaders(
        data_root, args.image_size, args.batch_size, args.num_workers
    )
    num_classes = len(classes)

    # Model
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
    model = convnext_large(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model.to(device)

    # Optimiser / Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_acc: float = 0.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        duration_min = (time.time() - epoch_start) / 60.0
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{duration_min:.1f} min"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time_min": duration_min,
            }
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "classes": classes,
                "class_to_idx": class_to_idx,
                "image_size": args.image_size,
            }
            torch.save(checkpoint, out_root / "best_model.pth")
            print(f"  >> New best model saved (val_acc={best_acc:.4f})")

    # Persist training log
    with open(out_root / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
