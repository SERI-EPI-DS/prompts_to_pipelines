# train.py
#!/usr/bin/env python3
"""
train.py - Fine-tune Swin‑V2‑B on a fundus image classification dataset.

Example:
    python train.py \
        --data_dir /path/to/data \
        --output_dir /path/to/project/results \
        --epochs 50 \
        --batch_size 32

Folder structure expected:
data_dir/
    train/
        class_a/
            img1.png
            ...
        class_b/
            ...
    val/
        ...
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import swin_v2_b, Swin_V2_B_Weights


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 8,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                256,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize(288, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = ImageFolder(data_dir / "val", transform=val_tfms)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_dl, val_dl, train_ds.classes


def save_classes(classes, output_dir: Path):
    with open(output_dir / "classes.json", "w") as f:
        json.dump(classes, f)


def create_model(num_classes: int, pretrained: bool = True):
    weights = Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None
    model = swin_v2_b(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model, criterion, optimizer, data_loader, device, scaler, accumulation_steps=1
):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / len(data_loader), correct / total


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return val_loss / len(data_loader), correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Fine‑tune Swin‑V2‑B for fundus classification"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to dataset root containing train/ val/.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to write checkpoints and logs.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, val_dl, classes = build_dataloaders(
        args.data_dir, args.batch_size, num_workers=os.cpu_count() // 2
    )
    save_classes(classes, args.output_dir)

    model = create_model(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_dl, device, scaler, args.accum_steps
        )
        val_loss, val_acc = evaluate(model, criterion, val_dl, device)
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{args.epochs}] - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "classes": classes,
                },
                args.output_dir / "best_model.pth",
            )
            print(f"Saved new best model with val_acc={best_acc:.4f}")

    print("Training complete. Best val_acc: {:.4f}".format(best_acc))


if __name__ == "__main__":
    main()
