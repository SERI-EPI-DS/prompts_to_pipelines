#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

from tqdm import tqdm


# -----------------------
# Utilities
# -----------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


@dataclass
class AverageMeter:
    name: str
    fmt: str = ":.4f"

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} (avg{avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(targets.numel(), 1)


class ModelEMA:
    """Exponential Moving Average of model parameters (simple, safe)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = self._clone_model(model)
        self.ema.eval()
        self.decay = decay

    def _clone_model(self, model: nn.Module) -> nn.Module:
        ema = self._copy_model_structure(model)
        ema.load_state_dict(model.state_dict(), strict=True)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    def _copy_model_structure(self, model: nn.Module) -> nn.Module:
        new = swin_v2_b(weights=None)
        new.head = nn.Linear(new.head.in_features, model.head.out_features, bias=True)
        return new

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k] * (1.0 - d))


# -------- FIX: robust mean/std resolution (no reliance on weights.meta) --------
def resolve_imagenet_stats(weights_enum) -> Tuple[List[float], List[float]]:
    """Try weights.transforms() for Normalize(mean,std); else fallback to ImageNet defaults."""
    # 1) Try weights.meta if present and has keys
    try:
        meta = getattr(weights_enum, "meta", None)
        if isinstance(meta, dict) and "mean" in meta and "std" in meta:
            return list(meta["mean"]), list(meta["std"])
    except Exception:
        pass

    # 2) Try to extract from the composed transforms (v1/v2 API)
    try:
        pipeline = weights_enum.transforms(antialias=True)
        # Compose (v1) exposes .transforms; v2 may expose .transforms or ._transforms
        tlist = getattr(pipeline, "transforms", None) or getattr(
            pipeline, "_transforms", None
        )
        if tlist:
            # Try both v1 and v2 Normalize classes
            from torchvision.transforms import Normalize as NormalizeV1

            try:
                from torchvision.transforms.v2 import Normalize as NormalizeV2  # type: ignore
            except Exception:
                NormalizeV2 = ()  # not available
            for t in tlist:
                if isinstance(t, (NormalizeV1, NormalizeV2)):
                    # t.mean/t.std may be lists/tuples/tensors
                    def _to_list(x):
                        if hasattr(x, "tolist"):
                            return [float(y) for y in x.tolist()]
                        return [float(y) for y in x]

                    return _to_list(t.mean), _to_list(t.std)
    except Exception:
        pass

    # 3) Final fallback: classic ImageNet stats
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------


def build_transforms(img_size: int, mean: List[float], std: List[float]):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5
            ),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), value="random"),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14), antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, eval_tfms


def make_datasets(data_dir: str, img_size: int):
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    # -------- FIX APPLIED HERE --------
    mean, std = resolve_imagenet_stats(weights)
    # ----------------------------------
    train_tfms, eval_tfms = build_transforms(img_size, mean, std)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_ds = ImageFolder(train_dir, transform=train_tfms)
    val_ds = ImageFolder(val_dir, transform=eval_tfms)
    return train_ds, val_ds


def make_loaders(
    train_ds, val_ds, batch_size: int, num_workers: int, balanced_sampler: bool = False
) -> Tuple[DataLoader, DataLoader]:
    if balanced_sampler:
        class_counts = [0] * len(train_ds.classes)
        for _, target in train_ds.samples:
            class_counts[target] += 1
        class_counts = [max(c, 1) for c in class_counts]
        weights_per_class = [1.0 / c for c in class_counts]
        sample_weights = [weights_per_class[target] for _, target in train_ds.samples]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def build_model(num_classes: int, finetune_from_imagenet: bool = True) -> nn.Module:
    if finetune_from_imagenet:
        weights = Swin_V2_B_Weights.IMAGENET1K_V1
        model = swin_v2_b(weights=weights)
    else:
        model = swin_v2_b(weights=None)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def save_checkpoint(
    state: Dict, is_best: bool, ckpt_dir: str, filename: str = "last.pt"
):
    ensure_dir(ckpt_dir)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(ckpt_dir, "best.pt")
        shutil.copyfile(path, best_path)


def train_one_epoch(
    model,
    ema,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    max_norm: float = 1.0,
    accum_steps: int = 1,
):
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, ncols=100, desc=f"Train {epoch}")
    for step, (images, targets) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            if max_norm is not None and max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

        with torch.no_grad():
            acc = accuracy(logits, targets)
            loss_meter.update(loss.item() * accum_steps, n=images.size(0))
            acc_meter.update(acc, n=images.size(0))
            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg*100:.2f}%"
            )

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(
    model, loader, criterion, device, use_ema=False, ema_model: ModelEMA = None
):
    model_to_eval = ema_model.ema if (use_ema and ema_model is not None) else model
    model_to_eval.eval()
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")

    for images, targets in tqdm(loader, ncols=100, desc="Validate", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model_to_eval(images)
            loss = criterion(logits, targets)
        acc = accuracy(logits, targets)
        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(acc, n=images.size(0))

    return loss_meter.avg, acc_meter.avg


def build_warmup_cosine(
    optimizer, epochs, steps_per_epoch, warmup_epochs=3, min_lr=1e-6, base_lr=5e-4
):
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(int(warmup_epochs * steps_per_epoch), 1)
    cosine_steps = max(total_steps - warmup_steps, 1)
    warmup = LinearLR(
        optimizer,
        start_factor=min_lr / max(base_lr, 1e-8),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_lr)
    sched = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
    return sched, warmup_steps, cosine_steps


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Swin-V2-B on fundus images")
    # Paths
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing train/val/test subfolders.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write results (checkpoints, logs, etc).",
    )

    # Core hyperparams
    p.add_argument("--epochs", type=int, default=50, help="Max epochs (<=50).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Input image size (Swin-V2-B works well at 256).",
    )
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])

    # Training tricks
    p.add_argument("--warmup_epochs", type=float, default=3.0)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument(
        "--accum_steps", type=int, default=1, help="Gradient accumulation steps."
    )
    p.add_argument(
        "--balanced_sampler",
        action="store_true",
        help="Use WeightedRandomSampler for class imbalance.",
    )
    p.add_argument(
        "--ema",
        action="store_true",
        help="Track an EMA of model weights and validate with it.",
    )
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Stop if val acc doesn't improve for N epochs.",
    )

    # System
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Resume
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint (.pt) to resume from.",
    )

    args = p.parse_args()

    if args.epochs > 50:
        raise ValueError("Limit epochs to 50 per requirement.")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    # Datasets / Loaders
    train_ds, val_ds = make_datasets(args.data_dir, args.img_size)
    num_classes = len(train_ds.classes)
    train_loader, val_loader = make_loaders(
        train_ds, val_ds, args.batch_size, args.num_workers, args.balanced_sampler
    )

    # Persist class info for test.py
    classes_path = os.path.join(args.output_dir, "classes.json")
    save_json({"classes": train_ds.classes}, classes_path)

    # Model
    model = build_model(num_classes=num_classes, finetune_from_imagenet=True)
    device = torch.device(args.device)
    model.to(device)

    # Loss (Cross-Entropy with label smoothing, as requested)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

    # Scheduler: warmup + cosine
    steps_per_epoch = max(len(train_loader) // max(args.accum_steps, 1), 1)
    scheduler, _, _ = build_warmup_cosine(
        optimizer,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        base_lr=args.lr,
    )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # EMA
    ema = ModelEMA(model, decay=0.999) if args.ema else None

    # Optionally resume
    start_epoch = 0
    best_acc = -1.0
    last_path = os.path.join(args.output_dir, "checkpoints", "last.pt")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", -1.0)
        if args.ema and "ema" in ckpt and ema is not None:
            ema.ema.load_state_dict(ckpt["ema"])
        print(
            f"Resumed from {args.resume} at epoch {start_epoch}, best_acc={best_acc:.4f}"
        )

    # Save run config
    save_json(
        {
            "args": vars(args),
            "num_classes": num_classes,
            "classes": train_ds.classes,
            "steps_per_epoch": len(train_loader),
        },
        os.path.join(args.output_dir, "run_config.json"),
    )

    # Training loop
    patience_counter = 0
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            ema,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            max_norm=args.max_grad_norm,
            accum_steps=args.accum_steps,
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, use_ema=args.ema, ema_model=ema
        )

        scheduler.step()

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_acc": best_acc,
            "classes": train_ds.classes,
        }
        if args.ema and ema is not None:
            ckpt["ema"] = ema.ema.state_dict()

        save_checkpoint(
            ckpt,
            is_best=is_best,
            ckpt_dir=os.path.join(args.output_dir, "checkpoints"),
            filename="last.pt",
        )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
            f"| val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% "
            f"| best_acc={best_acc*100:.2f}%"
        )

        if patience_counter >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch} (no val acc improvement for {patience_counter} epochs)."
            )
            break

    t1 = time.time()
    print(
        f"Training complete in {(t1 - t0)/60:.1f} min. Best val acc: {best_acc*100:.2f}%"
    )

    best_src = os.path.join(args.output_dir, "checkpoints", "best.pt")
    best_dst = os.path.join(args.output_dir, "best.pt")
    if os.path.exists(best_src):
        shutil.copyfile(best_src, best_dst)
        print(f"Best checkpoint copied to: {best_dst}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
