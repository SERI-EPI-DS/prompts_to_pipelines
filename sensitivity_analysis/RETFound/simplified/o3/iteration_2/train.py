# ============================
# train.py
# Fine‑tunes the RETFound ViT‑L model for fundus image diagnosis.
# Requires: Python 3.11+, PyTorch 2.3+, TorchVision 0.18+, timm 0.9+, torchmetrics 1.4+
# --------------------------------------------
"""
Example usage
-------------
python train.py \
    --data_dir /path/to/dataset \
    --output_dir ./outputs \
    --epochs 50 \
    --batch_size 32 \
    --lr 3e-5 \
    --checkpoint rmaphoh/RETFound-SSL-ViT-Large-16-224

`data_dir` must contain `train`, `val`, `test` sub‑folders, each with one folder per class.
`checkpoint` can be a local .pth/.safetensors file or a HuggingFace model repo.
"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm  # vision models
from torchmetrics.classification import AUROC
from torch.cuda.amp import GradScaler, autocast

# --------------------------------------------------
# Utility
# --------------------------------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_loaders(root: Path, img_size: int, batch_size: int, num_workers: int):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.ImageFolder(root / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(root / "val", transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes


def build_model(num_classes: int, checkpoint: str | None):
    # create vanilla ViT‑Large/16 backbone
    model = timm.create_model(
        "vit_large_patch16_224", pretrained=False, num_classes=num_classes
    )

    if checkpoint:
        # supports local path or HF hub repo id
        try:
            state = torch.hub.load_state_dict_from_url(
                checkpoint, map_location="cpu", trust_repo=True
            )
        except Exception:
            state = torch.load(checkpoint, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        # Strip classification head keys
        filtered = {k: v for k, v in state.items() if not k.startswith("head")}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(
            f"Loaded checkpoint with {len(filtered)} params (missing={len(missing)}, unexpected={len(unexpected)})"
        )
    return model


# --------------------------------------------------
# Training / Evaluation loops
# --------------------------------------------------


def evaluate(loader, model, device, criterion, num_classes):
    model.eval()
    loss_meter = 0.0
    correct = 0
    total = 0
    auroc = AUROC(task="multiclass", num_classes=num_classes).to(device)
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss_meter += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += imgs.size(0)
            auroc.update(outputs.softmax(dim=1), targets)
    return loss_meter / total, correct / total, auroc.compute().item()


def train_one_epoch(
    epoch, model, loader, criterion, optimizer, scaler, device, log_interval=50
):
    model.train()
    running = 0.0
    correct = 0
    total = 0
    for step, (imgs, targets) in enumerate(loader):
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += imgs.size(0)

        if (step + 1) % log_interval == 0:
            print(
                f"Epoch [{epoch}] Step [{step+1}/{len(loader)}] Loss: {running/total:.4f} Acc: {correct/total:.4f}"
            )
    return running / total, correct / total


def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=Path(args.output_dir) / "training.log",
        level=logging.INFO,
        format="%(asctime)s \u2502 %(levelname)s \u2502 %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, classes = get_data_loaders(
        Path(args.data_dir), args.img_size, args.batch_size, args.num_workers
    )
    model = build_model(len(classes), args.checkpoint).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler()
    best_auc = 0.0
    metrics_history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc, val_auc = evaluate(
            val_loader, model, device, criterion, len(classes)
        )
        scheduler.step()

        logger.info(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "lr": scheduler.get_last_lr()[0],
                }
            )
        )
        metrics_history.append((epoch, train_loss, val_loss, val_acc, val_auc))

        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = Path(args.output_dir) / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "classes": classes,
                },
                ckpt_path,
            )
            logger.info(
                f"\u2605 Saved best model to {ckpt_path} (AUROC={best_auc:.4f})"
            )

    # save metrics
    with open(Path(args.output_dir) / "training_metrics.json", "w") as fp:
        json.dump(metrics_history, fp, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine‑tune RETFound ViT‑Large classifier")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Folder with train/val/test sub‑directories",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save checkpoints & logs",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path or HuggingFace repo id of RETFound weights",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
