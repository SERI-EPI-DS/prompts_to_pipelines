# =============================
# file: train.py
# =============================
import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision.transforms import InterpolationMode

# -----------------------------
# Utilities
# -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    train_dir: str = "train"
    val_dir: str = "val"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    pretrained: bool = True
    freeze_backbone: bool = False
    workers: int = 8
    seed: int = 42
    grad_clip_norm: float = 1.0
    amp_dtype: str = "fp16"  # choices: fp16/bf16
    save_every: int = 0  # 0 -> only best & last
    resume: str = ""


# -----------------------------
# Model / Data
# -----------------------------


def build_transforms(image_size: int, mean: List[float], std: List[float]):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize(
                int(image_size * 1.14), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, val_tf


def create_model(num_classes: int, pretrained: bool, freeze_backbone: bool):
    weights = ConvNeXt_Large_Weights.DEFAULT if pretrained else None
    model = convnext_large(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier.2"):
                p.requires_grad = False
    return model, weights


# -----------------------------
# Training / Evaluation
# -----------------------------


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).sum().item()
        return correct / targets.numel()


def train_one_epoch(
    model,
    loader,
    device,
    optimizer,
    scaler,
    loss_fn,
    grad_clip_norm: float,
    amp_dtype: torch.dtype,
):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            logits = model(images)
            loss = loss_fn(logits, targets)
        scaler.scale(loss).backward()
        if grad_clip_norm and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy_top1(logits.detach(), targets) * images.size(0)
        n += images.size(0)

    return running_loss / n, running_acc / n


def evaluate(model, loader, device, loss_fn, amp_dtype: torch.dtype):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(images)
                loss = loss_fn(logits, targets)
            running_loss += loss.item() * images.size(0)
            running_acc += accuracy_top1(logits, targets) * images.size(0)
            n += images.size(0)

    return running_loss / n, running_acc / n


# -----------------------------
# Main
# -----------------------------


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Fine-tune ConvNeXt-L on fundus photographs"
    )
    p.add_argument(
        "--data_root",
        required=True,
        type=str,
        help="Path to dataset root containing train/val/test folders",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to save checkpoints and logs",
    )
    p.add_argument("--train_dir", default="train", type=str)
    p.add_argument("--val_dir", default="val", type=str)
    p.add_argument("--image_size", default=224, type=int)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--lr", default=5e-4, type=float)
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--warmup_epochs", default=3, type=int)
    p.add_argument("--label_smoothing", default=0.1, type=float)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.set_defaults(pretrained=True)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--workers", default=8, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--grad_clip_norm", default=1.0, type=float)
    p.add_argument("--amp_dtype", default="fp16", choices=["fp16", "bf16"], type=str)
    p.add_argument(
        "--save_every",
        default=0,
        type=int,
        help="Save checkpoint every N epochs (0 to disable)",
    )
    p.add_argument(
        "--resume", default="", type=str, help="Path to a checkpoint to resume from"
    )

    args = p.parse_args()
    return TrainConfig(**vars(args))


def main():
    cfg = parse_args()

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = get_device()
    torch.set_float32_matmul_precision("high")

    # Build datasets
    # Use ImageNet normalization by default or from weights
    tmp_weights = ConvNeXt_Large_Weights.DEFAULT if cfg.pretrained else None
    if tmp_weights is not None and hasattr(tmp_weights, "meta"):
        mean = tmp_weights.meta.get("mean", IMAGENET_MEAN)
        std = tmp_weights.meta.get("std", IMAGENET_STD)
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    train_tf, val_tf = build_transforms(cfg.image_size, mean, std)

    train_path = os.path.join(cfg.data_root, cfg.train_dir)
    val_path = os.path.join(cfg.data_root, cfg.val_dir)
    train_set = datasets.ImageFolder(train_path, transform=train_tf)
    val_set = datasets.ImageFolder(val_path, transform=val_tf)

    num_classes = len(train_set.classes)
    class_names = train_set.classes  # alphabetical order by ImageFolder
    class_index = {c: i for i, c in enumerate(class_names)}

    # Save class index mapping for test-time
    with open(os.path.join(cfg.output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=cfg.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, cfg.batch_size // 2),
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=cfg.workers > 0,
    )

    model, _ = create_model(num_classes, cfg.pretrained, cfg.freeze_backbone)
    model.to(device)
    model.to(memory_format=torch.channels_last)

    # Optimizer & Schedulers
    if cfg.freeze_backbone:
        params_to_opt = [p for p in model.parameters() if p.requires_grad]
    else:
        # Slightly higher LR on classifier head
        head_params = list(model.classifier[2].parameters())
        backbone_params = [
            p for n, p in model.named_parameters() if not n.startswith("classifier.2")
        ]
        params_to_opt = [
            {"params": backbone_params, "lr": cfg.lr * 0.5},
            {"params": head_params, "lr": cfg.lr},
        ]

    optimizer = optim.AdamW(params_to_opt, lr=cfg.lr, weight_decay=cfg.weight_decay)

    main_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[cfg.warmup_epochs],
        )
    else:
        scheduler = main_scheduler

    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_acc = 0.0
    start_epoch = 0

    # Resume if requested
    if cfg.resume and os.path.isfile(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(
            ckpt.get("optimizer_state_dict", optimizer.state_dict())
        )
        scheduler.load_state_dict(
            ckpt.get("scheduler_state_dict", scheduler.state_dict())
        )
        scaler.load_state_dict(ckpt.get("scaler_state_dict", scaler.state_dict()))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(
            f"Resumed from {cfg.resume} at epoch {start_epoch}, best_acc={best_acc:.4f}"
        )

    # Save config
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    log_path = os.path.join(cfg.output_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr_head,lr_backbone\n")

    amp_dtype = torch.float16 if cfg.amp_dtype == "fp16" else torch.bfloat16

    for epoch in range(start_epoch, cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            loss_fn,
            cfg.grad_clip_norm,
            amp_dtype,
        )
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn, amp_dtype)
        scheduler.step()

        # Current LRs (handle param groups)
        if (
            isinstance(optimizer.param_groups, list)
            and len(optimizer.param_groups) >= 2
        ):
            lr_backbone = optimizer.param_groups[0]["lr"]
            lr_head = optimizer.param_groups[1]["lr"]
        else:
            lr_backbone = optimizer.param_groups[0]["lr"]
            lr_head = lr_backbone

        print(
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} | lr_head={lr_head:.2e} lr_backbone={lr_backbone:.2e}"
        )

        # Logging
        with open(log_path, "a") as f:
            f.write(
                f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{lr_head:.8f},{lr_backbone:.8f}\n"
            )

        # Save last
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_acc": best_acc,
            "class_names": class_names,
            "image_size": cfg.image_size,
        }
        torch.save(ckpt, os.path.join(cfg.output_dir, "last.pt"))

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, os.path.join(cfg.output_dir, "best.pt"))
            print(f"✔ Saved new best checkpoint (val_acc={best_acc:.4f})")

        # Optional periodic saves
        if cfg.save_every and ((epoch + 1) % cfg.save_every == 0):
            torch.save(ckpt, os.path.join(cfg.output_dir, f"epoch_{epoch+1}.pt"))

    print("Training complete. Best val acc=%.4f" % best_acc)


if __name__ == "__main__":
    main()
