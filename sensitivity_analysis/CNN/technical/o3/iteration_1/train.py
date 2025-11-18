"""
train.py  (patched)
Fine‑tune ConvNeXt‑Large for multi‑class classification on colour fundus photographs.

**Patch notes (2025‑07‑16)**
• Removed the `hue` parameter from `transforms.ColorJitter` because
  an upstream PIL / torchvision issue can trigger an `OverflowError`
  on some images when random hue shifts push uint8 values outside the
  valid 0‑255 range. All other augmentations remain unchanged.

Usage:
    python train.py --data-dir /path/to/data --output-dir /path/to/project/results
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# --------------------------------------------------
# Utility
# --------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------
# Data
# --------------------------------------------------
def get_dataloaders(data_dir: Path, batch_size: int = 8, num_workers: int = 4):
    """Return train & val dataloaders and class mapping."""

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(384, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15, fill=(0,)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_test_tfm = transforms.Compose(
        [
            transforms.Resize(426, antialias=True),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfm)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_test_tfm)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dl, val_dl, train_ds.classes, train_ds.class_to_idx


# --------------------------------------------------
# Model
# --------------------------------------------------
def build_model(num_classes: int):
    model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


# --------------------------------------------------
# Training helpers
# --------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.inference_mode()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine‑tune ConvNeXt‑L")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root folder containing train/val/test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save models and logs",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--patience", type=int, default=10, help="Early‑stopping patience"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, val_dl, classes, class_to_idx = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    model = build_model(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_dl, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_dl, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # checkpoint each epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "class_to_idx": class_to_idx,
            },
            args.output_dir / "checkpoints" / f"epoch_{epoch:03d}.pth",
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output_dir / "model_best.pth")
            with open(args.output_dir / "meta.json", "w") as f:
                json.dump(
                    {
                        "classes": classes,
                        "class_to_idx": class_to_idx,
                        "best_epoch": epoch,
                        "best_val_acc": best_acc,
                    },
                    f,
                    indent=2,
                )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print("Training complete. Best val acc: {:.4f}".format(best_acc))


if __name__ == "__main__":
    main()
