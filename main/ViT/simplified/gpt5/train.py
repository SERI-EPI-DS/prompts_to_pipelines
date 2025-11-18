# =============================
# train.py — Fine-tune Swin-V2-B on foldered fundus dataset
# =============================
# Usage examples:
#   python train.py \
#     --data_dir /path/to/dataset \
#     --train_folder train --val_folder val \
#     --output_dir ./runs/swinv2b_exp1 \
#     --epochs 50 --batch_size 32
#
# Dataset structure expected:
#   dataset/
#     train/
#       classA/ img1.jpg ...
#       classB/ ...
#     val/
#       classA/ ...
#       classB/ ...
# (Optionally a test/ folder for the testing script.)
#
# This script implements: data aug (RandAugment), Mixup/CutMix, label smoothing,
# AMP, cosine LR w/ warmup, EMA, early stopping, and best-checkpoint saving.

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.swin_transformer import swin_v2_b, Swin_V2_B_Weights
from torch.optim.lr_scheduler import LambdaLR

# ----------------------
# Utils
# ----------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(string_: str):
    import re

    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


@dataclass
class TrainConfig:
    data_dir: str
    train_folder: str = "train"
    val_folder: str = "val"
    output_dir: str = "./runs/swinv2b"
    pretrained: bool = True
    image_size: int = 256  # Swin-V2-B commonly uses 256; must be multiple of 32
    batch_size: int = 32
    num_workers: int = 8
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.8
    ema_decay: float = 0.999
    early_stop_patience: int = 10
    seed: int = 42
    grad_clip_norm: float = 1.0
    resume: Optional[str] = None  # path to checkpoint to resume from


# ----------------------
# Mixup/CutMix helpers
# ----------------------


def rand_bbox(W, H, lam):
    # returns bbox coords for CutMix
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)
    return x1, y1, x2, y2


def mixup_data(x, y, alpha: float):
    if alpha <= 0:
        return x, y, 1.0
    lam = random.betavariate(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam


def cutmix_data(x, y, alpha: float):
    if alpha <= 0:
        return x, y, 1.0
    lam = random.betavariate(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    # adjust lambda to match pixel ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return x, (y_a, y_b), lam


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        # target is one-hot or soft distribution
        log_probs = F.log_softmax(logits, dim=1)
        return -(target * log_probs).sum(dim=1).mean()


class ModelEMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.ema = self._clone_model(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _clone_model(self, model):
        import copy

        ema = copy.deepcopy(model)
        return ema

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k].detach() * (1.0 - d))


# ----------------------
# Data
# ----------------------


def build_transforms(cfg: TrainConfig, weights):
    # Eval transforms from weights (guarantees correct resize/crop/normalize)
    if weights is not None:
        try:
            eval_tfms = weights.transforms()
            mean = weights.meta.get("mean", (0.485, 0.456, 0.406))
            std = weights.meta.get("std", (0.229, 0.224, 0.225))
        except Exception:
            eval_tfms = transforms.Compose(
                [
                    transforms.Resize(
                        cfg.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(cfg.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
    else:
        eval_tfms = transforms.Compose(
            [
                transforms.Resize(
                    cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(cfg.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    # Strong train aug: RandomResizedCrop + RandAugment + ColorJitter + HFlip
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                cfg.image_size,
                scale=(0.7, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tfms, eval_tfms


# ----------------------
# Training / Eval loops
# ----------------------


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


def train_one_epoch(
    model,
    ema,
    loader,
    optimizer,
    device,
    epoch,
    num_classes,
    cfg: TrainConfig,
    loss_ce: nn.Module,
    loss_soft: SoftTargetCrossEntropy,
    scaler: torch.cuda.amp.GradScaler,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        use_mix = random.random() < cfg.mix_prob
        if use_mix and (cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0):
            if cfg.cutmix_alpha > 0 and random.random() < 0.5:
                images, (ya, yb), lam = cutmix_data(images, targets, cfg.cutmix_alpha)
            else:
                images, (ya, yb), lam = mixup_data(images, targets, cfg.mixup_alpha)
            mixed_targets = one_hot(ya, num_classes) * lam + one_hot(
                yb, num_classes
            ) * (1 - lam)
        else:
            mixed_targets = None

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            if mixed_targets is not None:
                loss = loss_soft(logits, mixed_targets)
            else:
                loss = loss_ce(logits, targets)

        scaler.scale(loss).backward()
        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_targets = []
    loss_total = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss_total += ce(logits, targets).item()
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    acc = (preds == targets).float().mean().item()

    # Macro F1
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    avg_loss = loss_total / len(targets)
    return avg_loss, acc, macro_f1


# ----------------------
# Main
# ----------------------


def get_args():
    p = argparse.ArgumentParser(description="Fine-tune Swin-V2-B on fundus images")
    p.add_argument("--data_dir", type=str, required=True, help="Root dataset directory")
    p.add_argument("--train_folder", type=str, default="train")
    p.add_argument("--val_folder", type=str, default="val")
    p.add_argument("--output_dir", type=str, default="./runs/swinv2b")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--cutmix_alpha", type=float, default=1.0)
    p.add_argument("--mix_prob", type=float, default=0.8)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Initialize randomly instead of ImageNet1K pretrained",
    )
    p.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume"
    )
    return p.parse_args()


def build_model(num_classes: int, pretrained: bool = True):
    weights = Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None
    model = swin_v2_b(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model, weights


def make_schedulers(optimizer, num_epochs: int, warmup_epochs: int):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        # cosine from 1 -> 0
        progress = (current_epoch - warmup_epochs) / float(
            max(1, num_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def save_config(cfg: TrainConfig, classes: List[str]):
    os.makedirs(cfg.output_dir, exist_ok=True)
    conf = {
        "config": cfg.__dict__,
        "classes": classes,
    }
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w") as f:
        json.dump(conf, f, indent=2)


def main():
    args = get_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        train_folder=args.train_folder,
        val_folder=args.val_folder,
        output_dir=args.output_dir,
        pretrained=not args.no_pretrained,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mix_prob,
        ema_decay=args.ema_decay,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        grad_clip_norm=args.grad_clip_norm,
        resume=args.resume,
    )

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_dir = os.path.join(cfg.data_dir, cfg.train_folder)
    val_dir = os.path.join(cfg.data_dir, cfg.val_folder)

    # Model & weights
    model, weights = build_model(
        num_classes=2, pretrained=cfg.pretrained
    )  # temp num_classes, updated after dataset
    train_tfms, eval_tfms = build_transforms(cfg, weights)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)

    classes = train_ds.classes
    num_classes = len(classes)

    # Rebuild model head now that we know class count
    model, weights = build_model(num_classes=num_classes, pretrained=cfg.pretrained)
    model.to(device)

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Optim, sched, losses, EMA
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = make_schedulers(optimizer, cfg.epochs, cfg.warmup_epochs)

    loss_ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    loss_soft = SoftTargetCrossEntropy()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    ema = ModelEMA(model, decay=cfg.ema_decay)

    # Save config & class mapping
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "classes.json"), "w") as f:
        json.dump(
            {"classes": classes, "class_to_idx": train_ds.class_to_idx}, f, indent=2
        )
    save_config(cfg, classes)

    start_epoch = 0
    best_metric = -1.0

    # Optional resume
    if cfg.resume is not None and os.path.isfile(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        if "ema" in ckpt and ckpt["ema"] is not None:
            ema.ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", -1.0)
        print(f"Resumed from {cfg.resume} at epoch {start_epoch}")

    # Training loop
    patience = cfg.early_stop_patience
    history = []

    for epoch in range(start_epoch, cfg.epochs):
        for g in optimizer.param_groups:
            pass  # LR controlled by scheduler

        train_loss, train_acc = train_one_epoch(
            model,
            ema,
            train_loader,
            optimizer,
            device,
            epoch,
            num_classes,
            cfg,
            loss_ce,
            loss_soft,
            scaler,
        )
        scheduler.step()

        # Evaluate with EMA weights
        val_loss, val_acc, val_f1 = evaluate(ema.ema, val_loader, device, num_classes)

        history.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
            }
        )
        print(
            f"Epoch {epoch+1}/{cfg.epochs} | LR {optimizer.param_groups[0]['lr']:.6f} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} macroF1 {val_f1:.4f}"
        )

        # Checkpointing
        metric = val_f1  # choose macro-F1 as primary
        is_best = metric > best_metric
        if is_best:
            best_metric = metric
            torch.save(
                {
                    "model": ema.ema.state_dict(),
                    "classes": classes,
                    "best_metric": best_metric,
                },
                os.path.join(cfg.output_dir, "best.pt"),
            )

        # Save last
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.ema.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
            },
            os.path.join(cfg.output_dir, "last.pt"),
        )

        # Early stopping
        if not is_best:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break
        else:
            patience = cfg.early_stop_patience

    # Save history
    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
