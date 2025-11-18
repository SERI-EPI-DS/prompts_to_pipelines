#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune Swin-V2-B (torchvision) for multi-class classification.
- Uses ImageFolder datasets laid out as data_root/{train,val}/<class>/*.png|jpg|jpeg
- Mixed precision, AdamW, cosine LR, label smoothing CE
- Saves best model (by val accuracy), last checkpoint, class mapping, and a train log CSV

Tested with:
  Python 3.11
  PyTorch 2.3.1
  TorchVision 0.18.1
  CUDA 12.1
  Single NVIDIA RTX 3090 (24GB)
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import InterpolationMode
from torchvision.models.swin_transformer import swin_v2_b, Swin_V2_B_Weights


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism often reduces throughput; we prefer speed + repeatable init
    cudnn.deterministic = False
    cudnn.benchmark = True


def get_mean_std_from_weights(
    weights: Swin_V2_B_Weights,
) -> Tuple[List[float], List[float], InterpolationMode, int]:
    """
    Robustly extract normalization stats & defaults from torchvision weights.
    Falls back to ImageNet stats if keys are missing (some envs reported KeyError: 'mean').
    """
    # Sensible fallbacks
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    interpolation = InterpolationMode.BICUBIC
    size = 256

    try:
        meta = weights.meta  # 0.18 keeps stats in .meta
        mean = meta.get("mean", mean)
        std = meta.get("std", std)
        # torchvision includes a default size & interpolation in transforms(),
        # but we keep a robust fallback here:
        size = (
            int(meta.get("min_size", size))
            if isinstance(meta.get("min_size"), (int, float))
            else size
        )
    except Exception:
        pass

    # Try to borrow interpolation/size from the pre-configured transforms
    try:
        _t = weights.transforms()
        # no public accessor for mean/std/interp; we keep our extracted/fallback ones
        # but we can snag the interpolation used for resizing:
        if hasattr(_t, "transforms"):
            for t in _t.transforms:
                if hasattr(t, "interpolation"):
                    interpolation = t.interpolation
                if hasattr(t, "size") and isinstance(t.size, (tuple, list)):
                    # CenterCrop/Resize sometimes report size as tuple; use max side
                    size = max(size, max(t.size))
    except Exception:
        pass

    return mean, std, interpolation, size


def create_transforms(img_size: int, weights: Swin_V2_B_Weights):
    mean, std, interpolation, _ = get_mean_std_from_weights(weights)

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size, scale=(0.8, 1.0), interpolation=interpolation
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=10, interpolation=interpolation)],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
            ),
        ]
    )

    # Standard eval pipeline: Resize -> CenterCrop -> ToTensor -> Normalize
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14), interpolation=interpolation),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_tf, eval_tf


def prepare_dataloaders(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    weights: Swin_V2_B_Weights,
):
    train_tf, eval_tf = create_transforms(img_size, weights)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds = ImageFolder(val_dir, transform=eval_tf)

    classes = train_ds.classes
    assert (
        classes == val_ds.classes
    ), "Train/Val class sets differ. Ensure aligned folder names."

    # DataLoaders
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    return train_dl, val_dl, classes


def build_model(num_classes: int, pretrained: bool = True):
    weights = Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None
    model = swin_v2_b(weights=weights)
    # Replace classifier head
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total, correct = 0, 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")  # sum for easy averaging

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return acc, avg_loss


def save_checkpoint(state: Dict, is_best: bool, out_dir: Path):
    last_path = out_dir / "checkpoint_last.pth"
    torch.save(state, last_path)
    if is_best:
        best_path = out_dir / "best_model.pth"
        torch.save(state, best_path)


def train(args):
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Using device: {device}")

    # Build model with pretrained weights for stronger start
    _weights = Swin_V2_B_Weights.IMAGENET1K_V1
    train_dl, val_dl, classes = prepare_dataloaders(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weights=_weights,
    )
    num_classes = len(classes)

    model = build_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # Optimizer, scheduler, loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    # Cosine schedule with warmup
    total_steps = args.epochs * math.ceil(len(train_dl.dataset) / args.batch_size)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine from 1 -> 0
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scaler = GradScaler(enabled=(device.type == "cuda"))
    best_val_acc = 0.0
    global_step = 0

    # Save classes mapping for test time
    with open(out_dir / "classes.json", "w") as f:
        json.dump({"classes": classes}, f, indent=2)

    # CSV training log header
    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, targets in train_dl:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            running_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_acc, val_loss = evaluate(model, val_dl, device)

        # Save checkpoints
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "classes": classes,
                "args": vars(args),
            },
            is_best=is_best,
            out_dir=out_dir,
        )

        # Append log
        with open(log_path, "a") as f:
            f.write(
                f"{epoch:.0f},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{scheduler.get_last_lr()[0]:.8f}\n"
            )

        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"best_val_acc={best_val_acc:.4f}"
        )

    elapsed = time.time() - start_time
    print(
        f"Training completed in {elapsed/60.0:.1f} min. Best val acc: {best_val_acc:.4f}"
    )
    print(f"Best model saved to: {out_dir/'best_model.pth'}")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Swin-V2-B on fundus photographs")

    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data root that contains train/ val/ test/ subfolders",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Where to write results (checkpoints, logs, classes.json)",
    )

    p.add_argument(
        "--epochs", type=int, default=50, help="Max epochs (<=50 per requirement)"
    )
    p.add_argument("--batch_size", type=int, default=32, help="Per-step batch size")
    p.add_argument("--num_workers", type=int, default=6, help="DataLoader workers")
    p.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Input resolution (Swin-V2-B default is 256)",
    )
    p.add_argument(
        "--lr", type=float, default=5e-4, help="Initial learning rate for AdamW"
    )
    p.add_argument(
        "--weight_decay", type=float, default=0.05, help="AdamW weight decay"
    )
    p.add_argument(
        "--label_smoothing", type=float, default=0.1, help="CE label smoothing in [0,1)"
    )
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping; 0 or None to disable",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--cpu", action="store_true", help="Force CPU (debug)")

    args = p.parse_args()

    if args.epochs > 50:
        print("Capping epochs to 50 per requirement.")
        args.epochs = 50

    return args


if __name__ == "__main__":
    train(parse_args())
