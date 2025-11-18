#!/usr/bin/env python3
"""
Fine-tune ConvNeXt-Large on a folder-structured dataset.

Fixes:
* Fallback to ImageNet mean/std if `weights.meta["mean"]` / `["std"]` are absent.
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_large

# Try to import the weight enum (newer torchvision); otherwise stay compatible
try:
    from torchvision.models import ConvNeXt_Large_Weights

    _HAS_ENUM = True
except ImportError:  # very old torchvision
    ConvNeXt_Large_Weights = None
    _HAS_ENUM = False


# ─────────────────────── Data helpers ───────────────────────
def _imagenet_stats():
    """Default ImageNet mean/std (RGB)."""
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_dataloaders(data_dir: Path, batch_size: int, workers: int = 4):
    """
    Returns:
        image_datasets (dict[str, ImageFolder])
        dataloaders    (dict[str, DataLoader])
    """
    # Choose pre-trained weights if available
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1 if _HAS_ENUM else None

    # Robustly fetch mean/std
    if _HAS_ENUM and "mean" in weights.meta and "std" in weights.meta:
        mean, std = weights.meta["mean"], weights.meta["std"]
    else:
        mean, std = _imagenet_stats()

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    image_datasets = {
        split: datasets.ImageFolder(
            data_dir / split, transform=(train_tfms if split == "train" else val_tfms)
        )
        for split in ["train", "val"]
    }
    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=True,
        )
        for split in ["train", "val"]
    }
    return image_datasets, dataloaders


# ───────────────────── Train / Evaluate ─────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    run_loss, run_correct = 0.0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        run_loss += loss.item() * x.size(0)
        run_correct += (logits.argmax(1) == y).sum().item()

    n = len(loader.dataset)
    return run_loss / n, run_correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    run_loss, run_correct = 0.0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        run_loss += loss.item() * x.size(0)
        run_correct += (logits.argmax(1) == y).sum().item()

    n = len(loader.dataset)
    return run_loss / n, run_correct / n


# ─────────────────────────── Main ───────────────────────────
def main():
    pa = argparse.ArgumentParser("Fine-tune ConvNeXt-Large (robust)")
    pa.add_argument("--data_dir", type=Path, required=True)
    pa.add_argument("--output_dir", type=Path, required=True)
    pa.add_argument("--epochs", type=int, default=30)
    pa.add_argument("--batch_size", type=int, default=16)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--weight_decay", type=float, default=0.05)
    pa.add_argument("--workers", type=int, default=4)
    pa.add_argument(
        "--fp16/--no-fp16",
        dest="fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable AMP (CUDA)",
    )
    args = pa.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Using device: {device}")

    # Datasets / loaders
    datasets_dict, loaders = build_dataloaders(
        args.data_dir, args.batch_size, args.workers
    )
    num_classes = len(datasets_dict["train"].classes)
    print(f"📊 {num_classes} classes: {datasets_dict['train'].classes}")

    # Model
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1 if _HAS_ENUM else None
    model = convnext_large(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model.to(device)

    # Optimiser / scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    best_acc, best_path = 0.0, args.output_dir / "best_model.pth"

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} │ "
            f"train {tr_loss:.4f}/{tr_acc:.4f} │ "
            f"val {val_loss:.4f}/{val_acc:.4f} │ "
            f"{(time.time() - t0)/60:.1f} min"
        )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            },
            args.output_dir / f"epoch_{epoch+1:02d}.pth",
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"✔ Done. Best val acc {best_acc:.4f} (saved to {best_path})")


if __name__ == "__main__":
    main()
