## train\_swinv2\_b.py

#!/usr/bin/env python3
"""
Fine-tune a Swin-V2-B classifier on an ImageFolder dataset (train/val split).

Dataset directory structure expected:

    DATA_DIR/
      train/
        class_a/ img1.jpg ...
        class_b/ ...
        ...
      val/
        class_a/ ...
        class_b/ ...
        ...

Test set is optional here; use test script separately.

Key features:
- TorchVision Swin-V2-B (ImageNet1K pretrained)
- AMP mixed precision (automatic)
- AdamW + cosine LR with warmup (step-wise)
- Label smoothing and optional Focal loss
- Optional class-balanced training via WeightedRandomSampler
- Early stopping & best-checkpoint saving
- Reproducibility controls
- Clear CLI to set data/output folders

Requirements (PyPI):
  torch >= 1.12, torchvision >= 0.14, numpy, pillow
Optional (for richer metrics later): scikit-learn
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

# --------------------- Utilities ---------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_class_samples(targets: List[int], num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for t in targets:
        counts[t] += 1
    return counts


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification.

    Args:
        gamma: focusing parameter (2.0 typical)
        weight: per-class weights (Tensor of shape [C]) or None
        label_smoothing: smoothing factor in [0, 1)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        # Label-smoothed CE
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        ce = -(true_dist * log_probs)
        if self.weight is not None:
            ce = ce * self.weight.unsqueeze(0)
        ce = ce.sum(dim=1)

        probs = torch.exp(-ce.detach())  # approximate p_t
        focal = (1 - probs) ** self.gamma * ce
        return focal.mean()


@dataclass
class TrainConfig:
    data_dir: str
    train_subdir: str
    val_subdir: str
    output_dir: str
    epochs: int
    batch_size: int
    accum_steps: int
    lr: float
    weight_decay: float
    warmup_epochs: float
    img_size: int
    num_workers: int
    seed: int
    label_smoothing: float
    use_focal: bool
    focal_gamma: float
    class_weight_mode: str  # none | inv_freq
    sampler_balanced: bool
    mixup_alpha: float
    cutmix_alpha: float
    amp: bool
    channels_last: bool
    save_every: int


# --------------------- Training / Eval ---------------------


def build_transforms(
    img_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]
):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),  # ophthalmic dx typically invariant to left/right for many tasks; toggle if needed
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15), antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, eval_tfms


def maybe_build_sampler(train_dataset, class_weight_mode: str, sampler_balanced: bool):
    targets = getattr(train_dataset, "targets", None)
    if targets is None:
        return None, None

    classes = train_dataset.classes
    num_classes = len(classes)
    counts = count_class_samples(targets, num_classes)

    class_weights = None
    if class_weight_mode == "inv_freq":
        cw = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            cw[c] = 1.0 / max(1, counts[c])
        cw = cw / cw.sum() * num_classes  # normalize around 1.0
        class_weights = torch.tensor(cw, dtype=torch.float32)

    sampler = None
    if sampler_balanced:
        sample_weights = np.zeros(len(targets), dtype=np.float32)
        for idx, t in enumerate(targets):
            sample_weights[idx] = 1.0 / max(1, counts[t])
        sampler = WeightedRandomSampler(
            sample_weights.tolist(), num_samples=len(sample_weights), replacement=True
        )

    return sampler, class_weights


def accuracy_top1(logits, target):
    pred = logits.argmax(dim=1)
    correct = (pred == target).sum().item()
    return correct / target.size(0)


def evaluate(model, loader, device, amp=True):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    ce = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
                loss = ce(logits, targets)
            total_loss += loss.item()
            total_correct += (logits.argmax(1) == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


# --------------------- Main ---------------------


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root folder containing train/ and val/",
    )
    p.add_argument("--train_subdir", type=str, default="train")
    p.add_argument("--val_subdir", type=str, default="val")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--accum_steps", type=int, default=1, help="Gradient accumulation steps"
    )

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--warmup_epochs",
        type=float,
        default=3.0,
        help="Warmup duration in epochs (can be fractional)",
    )

    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--focal_gamma", type=float, default=2.0)

    p.add_argument(
        "--class_weight_mode", type=str, default="none", choices=["none", "inv_freq"]
    )
    p.add_argument(
        "--sampler_balanced", action="store_true", help="Use class-balanced sampling"
    )

    # (Mixup/CutMix hooks kept for future use. Disabled by default.)
    p.add_argument("--mixup_alpha", type=float, default=0.0)
    p.add_argument("--cutmix_alpha", type=float, default=0.0)

    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (recommended on GPUs)",
    )
    p.add_argument(
        "--channels_last", action="store_true", help="Use channels-last memory format"
    )
    p.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save snapshot every N epochs (0=only best/last)",
    )

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model & weights
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std = weights.meta.get("std", [0.229, 0.224, 0.225])

    train_tfms, eval_tfms = build_transforms(args.img_size, mean, std)

    train_dir = os.path.join(args.data_dir, args.train_subdir)
    val_dir = os.path.join(args.data_dir, args.val_subdir)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)

    num_classes = len(train_ds.classes)

    sampler, class_weights = maybe_build_sampler(
        train_ds, args.class_weight_mode, args.sampler_balanced
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    model = swin_v2_b(weights=weights)
    # Replace classifier head
    in_feats = model.head.in_features
    model.head = nn.Linear(in_feats, num_classes)

    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    model.to(device)

    # Optimizer & LR scheduler (cosine with warmup, step-wise)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = math.ceil(args.epochs * len(train_loader) / max(1, args.accum_steps))
    warmup_steps = int(
        args.warmup_epochs * len(train_loader) / max(1, args.accum_steps)
    )
    warmup_steps = min(warmup_steps, max(1, total_steps // 2))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss
    cw_device = None
    if class_weights is not None:
        cw_device = class_weights.to(device)

    if args.use_focal:
        criterion = FocalLoss(
            gamma=args.focal_gamma,
            weight=cw_device,
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing, weight=cw_device
        )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val_acc = 0.0
    best_epoch = -1
    patience = max(5, int(0.2 * args.epochs))  # early stopping heuristic
    epochs_no_improve = 0

    # Save config & class mapping
    config = TrainConfig(
        data_dir=args.data_dir,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        class_weight_mode=args.class_weight_mode,
        sampler_balanced=args.sampler_balanced,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        amp=args.amp,
        channels_last=args.channels_last,
        save_every=args.save_every,
    )

    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump({"classes": train_ds.classes}, f, indent=2)

    # ----------------- Training Loop -----------------
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        seen = 0

        for i, (images, targets) in enumerate(train_loader):
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(images)
                loss = criterion(logits, targets)
                loss = loss / max(1, args.accum_steps)

            scaler.scale(loss).backward()

            if (i + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            # Stats
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size * max(1, args.accum_steps)
            running_acc += accuracy_top1(logits.detach(), targets) * batch_size
            seen += batch_size

        train_loss = running_loss / max(1, seen)
        train_acc = running_acc / max(1, seen)

        val_loss, val_acc = evaluate(model, val_loader, device, amp=args.amp)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # Save best
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_acc": val_acc,
            "train_acc": train_acc,
            "classes": train_ds.classes,
            "img_size": args.img_size,
            "weights_name": "Swin_V2_B_Weights.IMAGENET1K_V1",
        }
        torch.save(state, os.path.join(args.output_dir, "last.pth"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(state, os.path.join(args.output_dir, "model_best.pth"))
            print(
                f"  → New best model saved at epoch {epoch+1} with val acc {val_acc:.4f}"
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.save_every and ((epoch + 1) % args.save_every == 0):
            torch.save(state, os.path.join(args.output_dir, f"epoch_{epoch+1:03d}.pth"))

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered (no improvement for {patience} epochs). Best epoch was {best_epoch+1}."
            )
            break

    print(
        f"Training complete. Best val acc={best_val_acc:.4f} (epoch {best_epoch+1 if best_epoch>=0 else 'N/A'})."
    )


if __name__ == "__main__":
    main()
