#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import convnext_large, ConvNeXt_Large_Weights


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # We keep cudnn.benchmark True for speed with augmentation
    torch.backends.cudnn.benchmark = True


def build_model(num_classes: int, pretrained: bool = True):
    if pretrained:
        weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
        model = convnext_large(weights=weights)
    else:
        model = convnext_large(weights=None)

    # Replace classification head
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def make_transforms(img_size: int = 224):
    # Ensure RGB, then apply ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.RandomResizedCrop(
                img_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), value="random"),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(
                int(img_size * 1.14), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    ce = nn.CrossEntropyLoss()  # for reporting; training uses label smoothing
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = ce(logits, targets)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def train_one_epoch(
    model, loader, optimizer, scaler, device, label_smoothing=0.1, clip_grad_norm=1.0
):
    model.train()
    total = 0
    running_loss = 0.0
    running_correct = 0

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if clip_grad_norm is not None and clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, dim=1)
        running_correct += (preds == targets).sum().item()
        total += images.size(0)

    avg_loss = running_loss / max(1, total)
    acc = running_correct / max(1, total)
    return avg_loss, acc


def save_checkpoint(state, path):
    torch.save(state, path)


def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune ConvNeXt-L on fundus photos")
    ap.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root folder containing train/val/test subfolders",
    )
    ap.add_argument(
        "--train-dir",
        type=str,
        default="train",
        help="Train subfolder name (under data-root)",
    )
    ap.add_argument(
        "--val-dir",
        type=str,
        default="val",
        help="Val subfolder name (under data-root)",
    )
    ap.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Folder to write checkpoints, logs, outputs",
    )
    ap.add_argument("--epochs", type=int, default=50, help="Max epochs (<=50 enforced)")
    ap.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size")
    ap.add_argument(
        "--lr", type=float, default=2e-4, help="Initial learning rate (AdamW)"
    )
    ap.add_argument(
        "--weight-decay", type=float, default=0.05, help="AdamW weight decay"
    )
    ap.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    ap.add_argument("--img-size", type=int, default=224, help="Input image size")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do NOT use ImageNet-1K pretrained weights",
    )
    ap.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a checkpoint to resume (optional)",
    )
    ap.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing for cross-entropy",
    )
    ap.add_argument(
        "--no-amp", action="store_true", help="Disable mixed-precision (fp16)"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Enforce max 50 epochs
    epochs = min(args.epochs, 50)

    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is required per the task constraints."

    # Data
    train_tf, eval_tf = make_transforms(args.img_size)
    train_path = os.path.join(args.data_root, args.train_dir)
    val_path = os.path.join(args.data_root, args.val_dir)

    train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    val_ds = datasets.ImageFolder(val_path, transform=eval_tf)

    num_classes = len(train_ds.classes)
    if num_classes < 2:
        raise ValueError(f"Found {num_classes} classes. Expect at least 2.")

    # Persist class mapping for test-time
    classes_path = os.path.join(args.results_dir, "classes.json")
    with open(classes_path, "w") as f:
        json.dump(
            {"classes": train_ds.classes, "class_to_idx": train_ds.class_to_idx},
            f,
            indent=2,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    # Model
    model = build_model(num_classes, pretrained=not args.no_pretrained).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr * 0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

    start_epoch = 0
    best_acc = -1.0
    ckpt_last = os.path.join(args.results_dir, "last.pt")
    ckpt_best = os.path.join(args.results_dir, "best.pt")

    # Optionally resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt.get("scaler", scaler.state_dict()))
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", best_acc)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # CSV training log
    log_path = os.path.join(args.results_dir, "training_log.csv")
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,time\n")

    for epoch in range(start_epoch, epochs):
        lr_now = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch+1}/{epochs} - lr={lr_now:.6f}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            label_smoothing=args.label_smoothing,
            clip_grad_norm=1.0,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        scheduler.step()

        # Save last
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_acc": best_acc,
                "num_classes": num_classes,
                "classes": train_ds.classes,
                "timestamp": datetime.now().isoformat(),
            },
            ckpt_last,
        )

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc": best_acc,
                    "num_classes": num_classes,
                    "classes": train_ds.classes,
                    "timestamp": datetime.now().isoformat(),
                },
                ckpt_best,
            )

        # Append log row
        with open(log_path, "a") as f:
            f.write(
                f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{lr_now:.8f},{datetime.now().isoformat()}\n"
            )

        print(
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | Val: loss={val_loss:.4f} acc={val_acc:.4f} | best_acc={best_acc:.4f}"
        )

    print("\nTraining complete.")
    print(f"Best val acc: {best_acc:.4f}")
    print(f"Saved best to: {ckpt_best}")
    print(f"Saved last to: {ckpt_last}")
    print(f"Classes file: {classes_path}")


if __name__ == "__main__":
    main()
