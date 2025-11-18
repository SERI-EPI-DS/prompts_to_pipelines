# ===============================
# train.py — fine‑tune a timm vision model (default Swin‑V2‑B) on fundus photographs
# ===============================
"""
Key changes v3
~~~~~~~~~~~~~~
* **Robust model resolver** – if the requested `--model_name` is missing from your local `timm`, we now:
  1. Warn that it’s unavailable and suggest upgrading ( `pip install -U timm` ).
  2. Fall back (in priority order) to another Swin‑V2, then Swin‑V1, then ViT, then ResNet‑50 – so training can continue.
* Added `--min_timm` flag (default **0.9.4**) to enforce a modern enough version; script aborts early with a clear message instead of crashing mid‑initialisation.
* Both **train.py** and **test.py** share the same `safe_create_model()` helper for consistency.

Usage (single‑GPU example)
-------------------------
python train.py \ 
    --train_dir /path/to/dataset/train \
    --val_dir   /path/to/dataset/val   \
    --output_dir ./results             \
    --model_name swinv2_base_window12_192_22k \
    --epochs 30 --batch_size 32 --lr 3e-4

"""

import argparse, os, random, sys, time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from packaging import version

import timm  # rely on whatever version user has

# ----------------------------------------------------
# Helper utilities
# ----------------------------------------------------


def check_timm(min_version="0.9.4"):
    """Abort if timm is too old for any Swin‑V2 weights."""
    if version.parse(timm.__version__) < version.parse(min_version):
        sys.exit(
            f"[ERROR] timm>={min_version} required for Swin‑V2.  You have timm=={timm.__version__}.  Run:  pip install -U timm"
        )


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_create_model(model_name: str, num_classes: int):
    """Return a valid timm model even if the preferred one is missing."""
    available = timm.list_models(pretrained=True)
    if model_name in available:
        return timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    print(f"[WARN] timm has no pretrained '{model_name}'.  Falling back…")
    fallback_priority = [
        "swinv2_small_window8_256",
        "swinv2_tiny_window8_256",
        "swin_base_patch4_window7_224",
        "swin_tiny_patch4_window7_224",
        "vit_base_patch16_224",
        "resnet50",
        "resnet18",
    ]
    for cand in fallback_priority:
        if cand in available:
            print(
                f"[INFO] Falling back to '{cand}' ({'pretrained' if allow_pretrained else 'random init'})."
            )
            try:
                return _create(cand, num_classes, pretrained=allow_pretrained)
            except RuntimeError:
                return _create(cand, num_classes, pretrained=False)

    # 3️⃣  Last-ditch: torchvision’s ResNet-18
    from torchvision.models import resnet18

    print(
        "[WARN] No usable timm backbones found; using torchvision.resnet18 (random init)."
    )
    model = resnet18(num_classes=num_classes)
    return model


# ----------------------------------------------------
# Argument parsing
# ----------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine‑tune a timm model on colour fundus photographs (ImageFolder layout)"
    )
    # data dirs
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--val_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    # model & hparams
    p.add_argument("--model_name", type=str, default="swinv2_base_window12_192_22k")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min_timm",
        type=str,
        default="0.9.4",
        help="Minimum timm version; abort if lower.",
    )
    return p.parse_args()


# ----------------------------------------------------
# Data transforms
# ----------------------------------------------------


def build_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, val_tf


# ----------------------------------------------------
# Main training loop
# ----------------------------------------------------


def main():
    args = parse_args()
    check_timm(args.min_timm)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | requested model: {args.model_name}")

    # data
    train_tf, val_tf = build_transforms(args.img_size)
    train_set = datasets.ImageFolder(args.train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(args.val_dir, transform=val_tf)

    num_classes = len(train_set.classes)
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

    # model (robust creation)
    model = safe_create_model(args.model_name, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs)
    )
    scaler = GradScaler()

    # dirs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = correct = total = 0
        start = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        epoch_loss /= total

        # validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        if epoch > args.warmup_epochs:
            scheduler.step()
        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss {epoch_loss:.4f} | train {train_acc:.4f} | val {val_acc:.4f} | {(time.time()-start)/60:.1f} m"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "classes": train_set.classes,
                    "img_size": args.img_size,
                    "model_name": args.model_name,
                },
                ckpt_path,
            )
            print(f"  ↳ saved best (val {best_acc:.4f}) to {ckpt_path}")

    print("Training done. Best val‑acc:", best_acc)


if __name__ == "__main__":
    main()
