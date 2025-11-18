#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune ConvNeXt-L on a folder-structured fundus dataset.

This version avoids relying on `weights.meta["mean"]` since some TorchVision
builds omit it; instead it derives normalization stats from `weights.transforms()`
or falls back to ImageNet defaults.

Dataset layout (example):
  DATASET_ROOT/
    train/
      ClassA/ img1.jpg ...
      ClassB/ ...
    val/
      ClassA/ ...
      ClassB/ ...
    test/        # optional here; used by test.py

Key features:
- ConvNeXt-L (ImageNet-1k pretrained) + classifier head replaced
- Strong augments (RandAugment, ColorJitter, RandomErasing)
- MixUp / CutMix (soft-label CE), label smoothing
- Cosine LR with warmup (AdamW), AMP, optional EMA
- Channels-last memory format, deterministic seeding
- Saves: best.pt / last.pt / classes.json / history.json

Tested with Python 3.11, PyTorch 2.3.1, TorchVision 0.18.1
"""
import argparse, json, math, os, random, time
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision.transforms import InterpolationMode

# --------------------------- Utils ---------------------------


def set_seed(seed: Optional[int] = 42):
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_norm_stats(
    weights,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Try to read mean/std from weights; fall back to defaults if unavailable."""
    # 1) Try weights.meta
    try:
        meta = getattr(weights, "meta", None)
        if isinstance(meta, dict) and "mean" in meta and "std" in meta:
            return tuple(meta["mean"]), tuple(meta["std"])
    except Exception:
        pass
    # 2) Try to parse from weights.transforms()
    try:
        t = weights.transforms()
        from torchvision.transforms import Normalize

        stack = [t]
        while stack:
            mod = stack.pop()
            if hasattr(mod, "transforms"):
                stack.extend(list(getattr(mod, "transforms")))
            if isinstance(mod, Normalize):
                # mod.mean / mod.std could be list/tuple/tensor
                mean = tuple(
                    float(x)
                    for x in (
                        mod.mean
                        if isinstance(mod.mean, (list, tuple))
                        else mod.mean.tolist()
                    )
                )
                std = tuple(
                    float(x)
                    for x in (
                        mod.std
                        if isinstance(mod.std, (list, tuple))
                        else mod.std.tolist()
                    )
                )
                return mean, std
    except Exception:
        pass
    # 3) ImageNet defaults
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy that accepts soft targets (as in MixUp/CutMix)."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()


def mixup(images, labels, alpha: float):
    if alpha <= 0:
        raise ValueError("alpha must be > 0 for mixup")
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    b = images.size(0)
    index = torch.randperm(b, device=images.device)
    mixed = lam * images + (1 - lam) * images[index, :]
    return mixed, (lam, index)


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def cutmix(images, labels, alpha: float):
    if alpha <= 0:
        raise ValueError("alpha must be > 0 for cutmix")
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    b = images.size(0)
    index = torch.randperm(b, device=images.device)
    x1, y1, x2, y2 = rand_bbox(images.size(), lam)
    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
    return images, (lam, index)


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_large(weights=weights)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return model


def build_transforms(img_size: int, mean, std):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size, scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
            ),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(
                int(img_size * 1.14), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, eval_tfms


def warmup_cosine_scheduler(
    optimizer, warmup_epochs: int, total_epochs: int, start_factor=0.001
):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return start_factor + (1 - start_factor) * (
                current_epoch / max(1, warmup_epochs)
            )
        progress = (current_epoch - warmup_epochs) / max(
            1, total_epochs - warmup_epochs
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True
) -> Tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(images)
            loss = ce(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


# --------------------------- Main train ---------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune ConvNeXt-L on folder-structured fundus datasets"
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root of dataset with train/val[/test] subfolders",
    )
    ap.add_argument("--train_dir", type=str, default="train")
    ap.add_argument("--val_dir", type=str, default="val")
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save checkpoints and logs",
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.0,
        help=">0 enables MixUp with given alpha",
    )
    ap.add_argument(
        "--cutmix_alpha",
        type=float,
        default=0.0,
        help=">0 enables CutMix with given alpha",
    )
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="0 to disable EMA; e.g., 0.999 enables EMA",
    )
    ap.add_argument(
        "--amp", action="store_true", help="Enable mixed precision training (AMP)"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(args.seed)

    run_name = time.strftime("convnextL_%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir or (Path("./runs") / run_name))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{now()}] Device: {device} | AMP: {args.amp} | EMA: {args.ema_decay}")

    # Datasets
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
    mean, std = get_norm_stats(weights)
    train_tfms, eval_tfms = build_transforms(args.img_size, mean, std)

    train_path = Path(args.data_dir) / args.train_dir
    val_path = Path(args.data_dir) / args.val_dir
    train_ds = datasets.ImageFolder(train_path, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_path, transform=eval_tfms)

    classes = train_ds.classes
    class_to_idx = train_ds.class_to_idx
    with open(out_dir / "classes.json", "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)

    num_classes = len(classes)
    model = create_model(num_classes=num_classes, pretrained=True)
    model.to(device).to(memory_format=torch.channels_last)

    ema_model = None

    def _ema_avg_fn(ema_param, curr_param, num_averaged):
        return args.ema_decay * ema_param + (1.0 - args.ema_decay) * curr_param

    if args.ema_decay and args.ema_decay > 0.0:
        ema_model = AveragedModel(model, avg_fn=_ema_avg_fn)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = warmup_cosine_scheduler(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs
    )

    hard_ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    soft_ce = SoftTargetCrossEntropy()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_acc, best_epoch = 0.0, -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_correct, seen = 0.0, 0, 0
        start = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            labels = labels.to(device, non_blocking=True)

            use_mixup = args.mixup_alpha and args.mixup_alpha > 0.0
            use_cutmix = args.cutmix_alpha and args.cutmix_alpha > 0.0

            targets_soft = None
            if use_mixup and use_cutmix:
                if random.random() < 0.5:
                    mixed, (lam, index) = mixup(images, labels, args.mixup_alpha)
                else:
                    mixed, (lam, index) = cutmix(images, labels, args.cutmix_alpha)
                images = mixed
                y_a = one_hot(labels, num_classes)
                y_b = one_hot(labels[index], num_classes)
                targets_soft = lam * y_a + (1 - lam) * y_b
            elif use_mixup:
                mixed, (lam, index) = mixup(images, labels, args.mixup_alpha)
                images = mixed
                y_a = one_hot(labels, num_classes)
                y_b = one_hot(labels[index], num_classes)
                targets_soft = lam * y_a + (1 - lam) * y_b
            elif use_cutmix:
                mixed, (lam, index) = cutmix(images, labels, args.cutmix_alpha)
                images = mixed
                y_a = one_hot(labels, num_classes)
                y_b = one_hot(labels[index], num_classes)
                targets_soft = lam * y_a + (1 - lam) * y_b

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(images)
                loss = (
                    soft_ce(logits, targets_soft)
                    if targets_soft is not None
                    else hard_ce(logits, labels)
                )

            scaler.scale(loss).backward()
            if args.clip_grad_norm and args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.clip_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()

            if ema_model is not None:
                ema_model.update_parameters(model)

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                epoch_correct += (preds == labels).sum().item()
                seen += labels.size(0)
                epoch_loss += loss.item() * labels.size(0)

        scheduler.step()
        train_loss = epoch_loss / max(1, seen)
        train_acc = epoch_correct / max(1, seen)

        eval_model = ema_model if ema_model is not None else model
        vloss, vacc = evaluate(eval_model, val_loader, device, amp=args.amp)
        elapsed = time.time() - start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": vloss,
                "val_acc": vacc,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        print(
            f"[{now()}] Epoch {epoch+1}/{args.epochs} | {elapsed:.1f}s | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {vloss:.4f} acc {vacc:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
        )

        to_save = (ema_model.module if ema_model is not None else model).state_dict()
        torch.save(
            {
                "model_state": to_save,
                "classes": classes,
                "config": vars(args),
                "epoch": epoch,
            },
            out_dir / "last.pt",
        )

        if vacc > best_acc:
            best_acc, best_epoch = vacc, epoch
            torch.save(
                {
                    "model_state": to_save,
                    "classes": classes,
                    "config": vars(args),
                    "epoch": epoch,
                },
                out_dir / "best.pt",
            )

        with open(out_dir / "history.json", "w") as f:
            json.dump(
                {
                    "history": history,
                    "best_val_acc": best_acc,
                    "best_epoch": best_epoch,
                },
                f,
                indent=2,
            )

    print(
        f"[{now()}] Done. Best val acc: {best_acc:.4f} (epoch {best_epoch+1}). Artifacts in: {out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
