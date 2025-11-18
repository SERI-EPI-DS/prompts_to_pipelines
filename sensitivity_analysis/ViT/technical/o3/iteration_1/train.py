#!/usr/bin/env python
"""train.py
Fine‑tunes a Swin‑V2‑B classifier on colour fundus photographs.
Command‑line usage e.g.:
    python train.py --data_dir ../data --output_dir ../project/results --epochs 50 --batch_size 32
"""
import argparse
import copy
import os
import random
import time
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def set_seed(seed: int = 42):
    """For reproducibility (where possible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # Faster on fixed‑size images


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------


def get_transforms(
    img_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns training & validation/test transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, val_tf


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int = 8):
    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes


# -----------------------------------------------------------------------------
# Model / optimisation utilities
# -----------------------------------------------------------------------------


def build_model(num_classes: int):
    """Returns a Swin‑V2‑B model with ImageNet‑1K weights and custom head."""
    weights = models.Swin_V2_B_Weights.IMAGENET1K_V1
    model = models.swin_v2_b(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double().item() / len(loader.dataset)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double().item() / len(loader.dataset)
    return epoch_loss, epoch_acc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine‑tune Swin‑V2‑B on colour fundus photographs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", required=True, help="Path to dataset root (containing train/val)"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save weights and logs"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Max epochs (≤50 per spec)"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = build_dataloaders(
        args.data_dir, args.batch_size
    )
    model = build_model(len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        since = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - since

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} — "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{epoch_time:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(
                {"model_state_dict": best_weights, "classes": class_names},
                os.path.join(args.output_dir, "best_model.pth"),
            )

    print(f"Training complete. Best val acc: {best_acc:.4f}")
    torch.save(
        {"model_state_dict": best_weights, "classes": class_names},
        os.path.join(args.output_dir, "best_model_final.pth"),
    )


if __name__ == "__main__":
    main()
