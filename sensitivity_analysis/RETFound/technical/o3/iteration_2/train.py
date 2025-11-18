#!/usr/bin/env python
"""Fine‑tune RETFound ViT‑L model on colour fundus photographs.

Example:
-------- 
python train.py \
    --data_root ../../data \
    --output_dir ../../project/results \
    --weights ../../RETFound/RETFound_CFP_weights.pth \
    --batch_size 32 --epochs 50 --lr 1e-4
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import timm
from timm.loss import LabelSmoothingCrossEntropy


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Fine‑tune RETFound ViT‑L")
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path containing train/val/test folders",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where checkpoints & logs are written",
    )
    p.add_argument(
        "--weights",
        type=str,
        default="../RETFound/RETFound_CFP_weights.pth",
        help="Path to RETFound pre‑trained MAE weights (.pth)",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument(
        "--amp", action="store_true", help="Use mixed‑precision training (recommended)"
    )
    return p.parse_args()


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------


def build_transforms(img_size):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(img_size + 32, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    return train_tf, val_tf


def save_checkpoint(state, is_best, output_dir, epoch):
    ckpt_path = Path(output_dir) / f"epoch_{epoch:03d}.pth"
    torch.save(state, ckpt_path)
    if is_best:
        best_path = Path(output_dir) / "best.pth"
        torch.save(state, best_path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def main():
    args = get_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # data
    train_tf, val_tf = build_transforms(args.img_size)
    train_set = datasets.ImageFolder(Path(args.data_root) / "train", transform=train_tf)
    val_set = datasets.ImageFolder(Path(args.data_root) / "val", transform=val_tf)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_classes = len(train_set.classes)
    class_names = train_set.classes

    # model
    model = timm.create_model(
        "vit_large_patch16_224", pretrained=False, num_classes=num_classes
    )
    print(f"Loading RETFound weights from {args.weights}")
    chkpt = torch.load(args.weights, map_location="cpu")
    # Some checkpoints use 'model' key
    state_dict = chkpt.get("model", chkpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Ignored layers ➜ missing: {len(missing)}, unexpected: {len(unexpected)}")

    model.to(device)

    # loss and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler(enabled=args.amp)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        count = 0
        start = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            count += labels.size(0)

        train_loss = running_loss / count
        train_acc = correct / count
        scheduler.step()

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(enabled=args.amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_count += labels.size(0)

        val_loss /= val_count
        val_acc = val_correct / val_count

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"time={elapsed/60:.1f}m"
        )

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        save_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
            "classes": class_names,
        }
        save_checkpoint(save_state, is_best, args.output_dir, epoch)

    # write meta
    meta = {"train_args": vars(args), "best_val_acc": best_acc, "classes": class_names}
    with open(Path(args.output_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
