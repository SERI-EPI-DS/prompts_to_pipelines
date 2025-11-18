#!/usr/bin/env python3
# Fine-tune ConvNeXt-L on folder-structured datasets.
# Compatible with: Python 3.11, PyTorch >= 2.2, TorchVision >= 0.18
#
# Folder layout (customizable via --train-folder/--val-folder/--test-folder):
#   DATA_ROOT/
#       train/
#           class_a/ img1.jpg ...
#           class_b/ ...
#       val/
#           class_a/ ...
#           class_b/ ...
#
# Example:
#   python train.py \
#     --data-root /path/to/dataset \
#     --train-folder train --val-folder val \
#     --outdir ./outputs/convnextl_run1 \
#     --epochs 50 --batch-size 32 --img-size 384 --mixup 0.2 --cutmix 1.0 --ema
#
# Notes:
# - Uses AdamW + cosine LR with warmup, AMP, RandAugment, Random Erasing,
#   optional Mixup/CutMix and EMA — strong modern recipe.
# - Saves: best.pth, last.pth, (optional) best_ema.pth, class_index.json, train_log.csv

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode
from torchvision.models import convnext_large, ConvNeXt_Large_Weights


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # keep fast; set False for determinism


def get_mean_std_from_weights(
    weights,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    # Robustly get mean/std; fallback to ImageNet defaults if meta missing
    try:
        meta = getattr(weights, "meta", None)
        if isinstance(meta, dict) and "mean" in meta and "std" in meta:
            return tuple(meta["mean"]), tuple(meta["std"])
    except Exception:
        pass
    # Fallback
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(img_size: int, mean, std):
    train_tfms = T.Compose(
        [
            T.Resize(
                int(img_size * 1.15),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ]
    )
    eval_tfms = T.Compose(
        [
            T.Resize(
                int(img_size * 1.15),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.CenterCrop(img_size),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean, std),
        ]
    )
    return train_tfms, eval_tfms


def replace_classifier_with(model: nn.Module, num_classes: int):
    # ConvNeXt in TorchVision has model.classifier as nn.Sequential([... Linear])
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        # find last Linear
        for i in range(len(model.classifier) - 1, -1, -1):
            if isinstance(model.classifier[i], nn.Linear):
                in_features = model.classifier[i].in_features
                model.classifier[i] = nn.Linear(in_features, num_classes)
                return model
    raise RuntimeError("Unexpected ConvNeXt classifier layout; cannot replace head.")


def param_groups_weight_decay(module: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@dataclass
class MixupCutmixCfg:
    mixup: float = 0.0
    cutmix: float = 0.0
    label_smoothing: float = 0.0
    num_classes: int = 1000


class MixupCutmix:
    """Simple Mixup/CutMix helper producing soft targets."""

    def __init__(self, cfg: MixupCutmixCfg):
        self.cfg = cfg

    def _sample_lambda(self, alpha: float) -> float:
        if alpha <= 0.0:
            return 1.0
        return float(torch._sample_dirichlet(torch.tensor([alpha, alpha])).max().item())

    def _rand_bbox(self, W: int, H: int, lam: float):
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = random.randint(0, W - 1)
        cy = random.randint(0, H - 1)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)
        return x1, y1, x2, y2

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        lam = 1.0
        use_mixup = self.cfg.mixup > 0.0 and random.random() < 0.5
        use_cutmix = self.cfg.cutmix > 0.0 and not use_mixup
        if use_mixup:
            lam = self._sample_lambda(self.cfg.mixup)
            index = torch.randperm(x.size(0), device=x.device)
            x = lam * x + (1.0 - lam) * x[index, :]
            y = self._soft_targets(y, index=index, lam=lam)
        elif use_cutmix:
            lam = self._sample_lambda(self.cfg.cutmix)
            index = torch.randperm(x.size(0), device=x.device)
            x1, y1, x2, y2 = self._rand_bbox(x.size(3), x.size(2), lam)
            x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
            # adjust lam based on exact area
            lam = 1.0 - (float((x2 - x1) * (y2 - y1)) / (x.size(-1) * x.size(-2)))
            y = self._soft_targets(y, index=index, lam=lam)
        else:
            # only label smoothing
            y = self._one_hot_smooth(y)
        return x, y

    def _one_hot_smooth(self, target: torch.Tensor):
        y = F.one_hot(target, num_classes=self.cfg.num_classes).float()
        if self.cfg.label_smoothing > 0.0:
            eps = self.cfg.label_smoothing
            y = y * (1 - eps) + eps / self.cfg.num_classes
        return y

    def _soft_targets(self, target: torch.Tensor, index: torch.Tensor, lam: float):
        y1 = F.one_hot(target, num_classes=self.cfg.num_classes).float()
        y2 = F.one_hot(target[index], num_classes=self.cfg.num_classes).float()
        if self.cfg.label_smoothing > 0.0:
            eps = self.cfg.label_smoothing
            y1 = y1 * (1 - eps) + eps / self.cfg.num_classes
            y2 = y2 * (1 - eps) + eps / self.cfg.num_classes
        return lam * y1 + (1 - lam) * y2


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor):
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss


class ModelEMA:
    """Exponential Moving Average of model parameters for evaluation stability."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        self.ema = self._clone_model(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device)

    def _clone_model(self, model):
        import copy

        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k] * (1.0 - d))

    def state_dict(self):
        return self.ema.state_dict()

    def to(self, device):
        self.ema.to(device)


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion=None
) -> Tuple[float, float]:
    model.eval()
    correct, n = 0, 0
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            if criterion is not None:
                running_loss += criterion(outputs, targets).item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            n += targets.size(0)
    top1 = 100.0 * correct / max(n, 1)
    avg_loss = running_loss / max(n, 1) if criterion is not None else 0.0
    return avg_loss, top1


def main():
    p = argparse.ArgumentParser(
        description="Fine-tune ConvNeXt-L on image folder datasets"
    )
    p.add_argument(
        "--data-root", type=str, required=True, help="Root folder containing splits"
    )
    p.add_argument("--train-folder", type=str, default="train")
    p.add_argument("--val-folder", type=str, default="val")
    p.add_argument(
        "--outdir", type=str, required=True, help="Where to save checkpoints and logs"
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=384, help="224 or 384 typically")
    p.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Base LR for batch-size 256; scaled linearly otherwise",
    )
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=5.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--pretrained", action="store_true", help="Start from ImageNet weights"
    )
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument(
        "--ema",
        action="store_true",
        help="Use EMA of weights for evaluation/checkpointing",
    )
    p.add_argument(
        "--balancesampler",
        action="store_true",
        help="Use class-weighted sampling on train set",
    )
    p.add_argument(
        "--accum-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Model & weights
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1 if args.pretrained else None
    model = convnext_large(weights=weights)
    # Replace head after we know num_classes

    mean, std = (
        get_mean_std_from_weights(weights)
        if weights is not None
        else ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    )
    train_tfms, eval_tfms = build_transforms(args.img_size, mean, std)

    # Datasets
    train_dir = Path(args.data_root) / args.train_folder
    val_dir = Path(args.data_root) / args.val_folder
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tfms)
    val_ds = datasets.ImageFolder(str(val_dir), transform=eval_tfms)

    num_classes = len(train_ds.classes)
    model = replace_classifier_with(model, num_classes)

    # Save class-index mapping for test-time
    class_index_path = outdir / "class_index.json"
    with open(class_index_path, "w") as f:
        json.dump(
            {
                "class_to_idx": train_ds.class_to_idx,
                "idx_to_class": {int(v): k for k, v in train_ds.class_to_idx.items()},
            },
            f,
            indent=2,
        )

    # Sampler (optional, balances class counts)
    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    if args.balancesampler:
        # compute weights per class
        counts = [0] * num_classes
        for _, target in train_ds.samples:
            counts[target] += 1
        class_weights = torch.tensor(
            [1.0 / max(c, 1) for c in counts], dtype=torch.double
        )
        sample_weights = [class_weights[target] for _, target in train_ds.samples]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **train_loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **train_loader_kwargs)

    val_loader = DataLoader(val_ds, shuffle=False, **train_loader_kwargs)

    device = torch.device(args.device)
    model.to(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Optimization
    # Linear LR scaling wrt batch size (reference BS=256)
    effective_bsz = args.batch_size * max(1, args.accum_steps)
    base_ref_bsz = 256
    lr = args.lr * (effective_bsz / base_ref_bsz)

    optim = AdamW(
        param_groups_weight_decay(model, args.weight_decay), lr=lr, betas=(0.9, 0.999)
    )

    # Cosine schedule with warmup
    def lr_lambda(cur_epoch):
        if cur_epoch < args.warmup_epochs:
            return float(cur_epoch) / max(args.warmup_epochs, 1e-8)
        progress = (cur_epoch - args.warmup_epochs) / max(
            args.epochs - args.warmup_epochs, 1e-8
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema = ModelEMA(model, decay=0.9999, device=device) if args.ema else None

    mix_cfg = MixupCutmixCfg(
        mixup=args.mixup,
        cutmix=args.cutmix,
        label_smoothing=args.label_smoothing,
        num_classes=num_classes,
    )
    aug = MixupCutmix(mix_cfg)

    # Training loop
    best_top1 = 0.0
    best_ema_top1 = 0.0
    log_path = outdir / "train_log.csv"
    with open(log_path, "w", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(
            ["epoch", "lr", "train_loss", "val_loss", "val_top1", "ema_val_top1"]
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            nseen = 0
            optim.zero_grad(set_to_none=True)

            for step, (images, targets) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # Augment (mixup/cutmix/smoothing) -> soft targets
                images, soft_targets = aug(images, targets)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = soft_cross_entropy(outputs, soft_targets) / args.accum_steps

                scaler.scale(loss).backward()

                if step % args.accum_steps == 0:
                    if args.grad_clip is not None and args.grad_clip > 0:
                        scaler.unscale_(optim)
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)

                    if ema is not None:
                        ema.update(model)

                running += loss.item() * images.size(0) * args.accum_steps
                nseen += images.size(0)

            scheduler.step()

            train_loss = running / max(1, nseen)
            # Evaluate (main model)
            val_loss, val_top1 = evaluate(
                model,
                val_loader,
                device,
                criterion=nn.CrossEntropyLoss(label_smoothing=0.0),
            )
            ema_top1 = None
            if ema is not None:
                ema_loss, ema_top1 = evaluate(
                    ema.ema, val_loader, device, criterion=nn.CrossEntropyLoss()
                )

            # Logging
            cur_lr = optim.param_groups[0]["lr"]
            writer.writerow(
                [
                    epoch,
                    f"{cur_lr:.6f}",
                    f"{train_loss:.4f}",
                    f"{val_loss:.4f}",
                    f"{val_top1:.2f}",
                    f"{'' if ema_top1 is None else f'{ema_top1:.2f}'}",
                ]
            )
            f_log.flush()

            # Checkpointing
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "num_classes": num_classes,
                    "args": vars(args),
                },
                outdir / "last.pth",
            )

            if val_top1 > best_top1:
                best_top1 = val_top1
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "num_classes": num_classes,
                        "args": vars(args),
                    },
                    outdir / "best.pth",
                )

            if ema is not None and ema_top1 is not None and ema_top1 > best_ema_top1:
                best_ema_top1 = ema_top1
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": ema.state_dict(),
                        "num_classes": num_classes,
                        "args": vars(args),
                    },
                    outdir / "best_ema.pth",
                )

            print(
                f"Epoch {epoch:03d}/{args.epochs} | LR {cur_lr:.6f} | Train {train_loss:.4f} | Val {val_loss:.4f} | Top1 {val_top1:.2f}% | EMA {'' if ema_top1 is None else f'{ema_top1:.2f}%'}"
            )

    print(
        "Done. Best Top-1 (model): %.2f%%  | Best Top-1 (EMA): %s"
        % (best_top1, "n/a" if not args.ema else f"{best_ema_top1:.2f}%")
    )


if __name__ == "__main__":
    main()
