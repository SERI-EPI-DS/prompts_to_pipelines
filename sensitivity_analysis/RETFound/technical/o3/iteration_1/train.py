# train.py – Fine‑tuning RETFound_mae ViT‑L
"""
Usage example
-------------
python train.py \
    --data_dir /path/to/data \
    --output_dir ./results \
    --pretrained_path ../RETFound/RETFound_CFP_weights.pth \
    --epochs 50 --batch_size 8 --lr 1e-4

Folder structure (required)
---------------------------
<data_dir>/train/<class_folders>/images...
<data_dir>/val/<class_folders>/images...

Outputs
-------
* <output_dir>/best_model.pth – best val accuracy
* <output_dir>/training_log.csv – per‑epoch metrics
"""

import argparse
import csv
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torchvision.datasets import ImageFolder

# Prefer torchvision v2 transforms, fall back gracefully
try:
    import torchvision.transforms.v2 as transforms
except ImportError:  # torchvision<0.16
    from torchvision import transforms  # type: ignore

try:
    import timm  # provides ViT‑L backbone
except ImportError as e:
    raise ImportError("Please install timm: pip install timm") from e


def parse_args():
    parser = argparse.ArgumentParser(description="Fine‑tune RETFound_mae ViT‑L model")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root directory containing train/val folders",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./results"))
    parser.add_argument(
        "--pretrained_path",
        type=Path,
        default=Path("../RETFound/RETFound_CFP_weights.pth"),
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int = 42):
    import random, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_transforms(img_size: int):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),  # 256 for 224
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tfms, val_tfms


def get_dataloaders(data_root: Path, img_size: int, batch_size: int, num_workers: int):
    train_tfms, val_tfms = get_transforms(img_size)
    train_ds = ImageFolder(root=data_root / "train", transform=train_tfms)
    val_ds = ImageFolder(root=data_root / "val", transform=val_tfms)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes


def load_model(num_classes: int, pretrained_path: Path, image_size: int):
    # Create ViT‑L model with 16×16 patches
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=image_size,
        num_classes=num_classes,
        pretrained=False,
    )

    # Load RETFound MAE weights, ignoring classification head mismatch
    if pretrained_path.exists():
        ckpt = torch.load(pretrained_path, map_location="cpu")
        ckpt_state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
        print(
            f"Loaded RETFound weights with {len(missing)} missing & {len(unexpected)} unexpected keys"
        )
    else:
        print("Warning: Pretrained weights not found; training from scratch")

    return model


def evaluate(model: nn.Module, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad(), autocast():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, class_names = get_dataloaders(
        args.data_dir, args.image_size, args.batch_size, args.num_workers
    )
    num_classes = len(class_names)

    # Model
    model = load_model(num_classes, args.pretrained_path, args.image_size)
    model.to(device)

    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler()

    best_acc = 0.0
    log_path = args.output_dir / "training_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scaler, criterion, device
            )
            val_loss, val_acc = evaluate(model, val_loader, device)
            scheduler.step()

            # Logging
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"Train: {train_loss:.4f}/{train_acc:.3%} | "
                f"Val: {val_loss:.4f}/{val_acc:.3%} | "
                f"Time: {(time.time() - start):.1f}s"
            )

            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), args.output_dir / "best_model.pth")
                print(f"  ↳ Saved new best model with acc={best_acc:.3%}")

    print("Training complete. Best val acc:", best_acc)


if __name__ == "__main__":
    main()
