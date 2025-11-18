#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune SwinV2-Base (timm) on an ImageFolder dataset.
Addresses device mismatch by moving the model back to device after reset_classifier.

Requires: torch, torchvision, timm, numpy, scikit-learn, matplotlib, tqdm
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from tqdm import tqdm

import timm
from timm.data import resolve_data_config, create_transform, Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2, accuracy


# ---------------------------
# Utilities
# ---------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = np.bincount(
        [y for _, y in dataset.samples], minlength=len(dataset.classes)
    )
    inv_freq = 1.0 / (counts + 1e-12)
    weights = inv_freq / inv_freq.sum() * len(dataset.classes)
    return torch.tensor(weights, dtype=torch.float32)


def make_dataloaders(
    data_dir: Path,
    train_sub: str,
    val_sub: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool,
    model_for_cfg: nn.Module,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    train_dir = data_dir / train_sub
    val_dir = data_dir / val_sub

    data_cfg = resolve_data_config({}, model=model_for_cfg)
    if img_size is not None:
        data_cfg["input_size"] = (3, img_size, img_size)

    train_tf = create_transform(
        input_size=data_cfg["input_size"],
        is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1",
    )
    val_tf = create_transform(
        input_size=data_cfg["input_size"],
        is_training=False,
    )

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    class_to_idx = train_ds.class_to_idx

    if use_weighted_sampler:
        cls_w = compute_class_weights(train_ds)
        sample_weights = [cls_w[y].item() for _, y in train_ds.samples]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_to_idx


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, use_amp: bool):
    model.eval()
    losses, top1s = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                outputs = model(images)
                loss = loss_fn(outputs, targets)
        else:
            outputs = model(images)
            loss = loss_fn(outputs, targets)

        losses.append(loss.item())
        top1s.append(accuracy(outputs, targets, topk=(1,))[0].item())
    return float(np.mean(losses)), float(np.mean(top1s))


def save_checkpoint(state: dict, is_best: bool, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_dir / "last.pth")
    if is_best:
        torch.save(state, out_dir / "best.pth")


# ---------------------------
# CLI
# ---------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune SwinV2-B on ImageFolder data")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root folder containing train/val/test subfolders",
    )
    p.add_argument(
        "--train_dir", type=str, default="train", help="Train subfolder name"
    )
    p.add_argument("--val_dir", type=str, default="val", help="Val subfolder name")
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output folder for checkpoints & logs",
    )
    p.add_argument(
        "--model",
        type=str,
        default="swinv2_base_window12to24_192to384_22kft1k",
        help="timm model name (default: SwinV2-B)",
    )
    p.add_argument("--img_size", type=int, default=384, help="Input resolution")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.2, help="Set 0 to disable")
    p.add_argument("--cutmix", type=float, default=1.0, help="Set 0 to disable")
    p.add_argument("--drop_path", type=float, default=0.2)
    p.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze all except classifier head (linear probe)",
    )
    p.add_argument("--ema", action="store_true", help="Enable EMA (ModelEmaV2)")
    p.add_argument(
        "--early_stop", type=int, default=10, help="Early stopping patience (epochs)"
    )
    p.add_argument(
        "--weighted_sampler",
        action="store_true",
        help="Use class-balanced sampler for imbalance",
    )
    p.add_argument(
        "--resume", type=str, default="", help="Path to checkpoint to resume from"
    )
    return p.parse_args()


# ---------------------------
# Main
# ---------------------------


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Device: {device} | AMP: {use_amp}")

    # 1) Create model WITHOUT final classifier (num_classes=0)
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=0,
        drop_path_rate=args.drop_path,
    )
    model = model.to(device)

    # 2) Build dataloaders (use model only to get transforms)
    train_loader, val_loader, class_to_idx = make_dataloaders(
        data_dir=data_dir,
        train_sub=args.train_dir,
        val_sub=args.val_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.weighted_sampler,
        model_for_cfg=model,
    )

    num_classes = len(class_to_idx)
    print(f"Detected {num_classes} classes.")
    with open(out_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    # 3) Attach the correct classifier head
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes)
    else:
        model.classifier = nn.Linear(getattr(model, "num_features", 1024), num_classes)

    # === DEVICE FIX ===
    # reset_classifier creates a NEW head on CPU; move model back to device.
    model = model.to(device)

    # 4) Optionally freeze backbone
    if args.freeze_backbone:
        for n, p in model.named_parameters():
            if n.startswith("head") or "classifier" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    # 5) Losses
    mixup_active = args.mixup > 0.0 or args.cutmix > 0.0
    mixup_fn = (
        Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            label_smoothing=args.label_smoothing,
            num_classes=num_classes,
        )
        if mixup_active
        else None
    )

    train_criterion = (
        SoftTargetCrossEntropy()
        if mixup_active
        else LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    )
    val_criterion = nn.CrossEntropyLoss()

    # 6) Optimizer & Scheduler
    optimizer = create_optimizer_v2(
        model, opt="adamw", lr=args.lr, weight_decay=args.weight_decay
    )

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=max(1, total_steps - warmup_steps),
        lr_min=1e-6,
        warmup_lr_init=args.lr * 0.1,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,  # step per-iteration
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 7) EMA
    model_ema = ModelEmaV2(model, decay=0.9999, device=device) if args.ema else None

    # 8) Resume (if any)
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "scaler" in ckpt and isinstance(ckpt["scaler"], dict):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", 0.0)
        if model_ema and "model_ema" in ckpt:
            model_ema.module.load_state_dict(ckpt["model_ema"], strict=False)

        # === DEVICE FIX (resume path) ===
        model = model.to(device)
        if model_ema:
            model_ema.module = model_ema.module.to(device)

        print(
            f"Resumed from {args.resume} at epoch {start_epoch} (best_acc={best_acc:.3f})"
        )

    # Save args
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---------------------------
    # Training loop
    # ---------------------------
    patience = args.early_stop
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, running_acc = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                images, targets = mixup_fn(images, targets)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=True
                ):
                    outputs = model(images)
                    loss = train_criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = train_criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            if model_ema is not None:
                model_ema.update(model)

            global_step = step + epoch * steps_per_epoch
            lr_scheduler.step(global_step)

            with torch.no_grad():
                if mixup_active:
                    pred = outputs.argmax(dim=1)
                    hard_t = targets.argmax(dim=1)
                    acc1 = (pred == hard_t).float().mean().item() * 100.0
                else:
                    acc1 = accuracy(outputs, targets, topk=(1,))[0].item()

            running_loss.append(loss.item())
            running_acc.append(acc1)
            pbar.set_postfix(
                {
                    "loss": f"{np.mean(running_loss):.4f}",
                    "acc": f"{np.mean(running_acc):.2f}%",
                }
            )

        # Validation
        eval_model = model_ema.module if (model_ema is not None) else model
        # (safe, but redundant; eval_model is already on device)
        eval_model = eval_model.to(device)

        val_loss, val_top1 = evaluate(
            eval_model, val_loader, device, nn.CrossEntropyLoss(), use_amp=False
        )
        print(f"\nVal loss: {val_loss:.4f} | Val acc@1: {val_top1:.2f}%")

        # Checkpoint
        state = {
            "epoch": epoch + 1,
            "model": (
                eval_model.state_dict() if model_ema is not None else model.state_dict()
            ),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_amp else {},
            "best_acc": max(best_acc, val_top1),
            "args": vars(args),
            "class_to_idx": class_to_idx,
            "model_name": args.model,
        }
        improved = val_top1 > best_acc
        best_acc = max(best_acc, val_top1)
        save_checkpoint(state, improved, out_dir)

        # Early stopping
        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping after {epoch+1} epochs (no improvement for {patience})."
                )
                break

    print(f"Training complete. Best Val Acc: {best_acc:.2f}%")
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
