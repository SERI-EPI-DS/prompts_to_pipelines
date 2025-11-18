#!/usr/bin/env python3
"""
Fine-tune Swin-V2-B on a fundus-image classification task.

Typical use:
python train.py \
    --data_dir /path/to/dataset \
    --output_dir ./runs/exp1 \
    --epochs 50 \
    --batch_size 32
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import Swin_V2_B_Weights as SwinWeights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Swin-V2-B.")
    parser.add_argument(
        "--data_dir", required=True, help="Root folder with train/val/test subfolders."
    )
    parser.add_argument(
        "--output_dir", default="./output", help="Where to save checkpoints & logs."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="If set, only train the classification head.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandAugment(),
            transforms.ToTensor(),
            SwinWeights.IMAGENET1K_V1.transforms(),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            SwinWeights.IMAGENET1K_V1.transforms(),
        ]
    )
    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(Path(data_dir) / "val", transform=val_tfms)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dl, val_dl, train_ds.classes


def build_model(num_classes: int, freeze_backbone: bool = False):
    model = models.swin_v2_b(weights=SwinWeights.IMAGENET1K_V1)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("head."):
                param.requires_grad_(False)
    return model


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    return (output.argmax(1) == target).float().mean().item()


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    epoch_loss = acc = n = 0
    criterion = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        epoch_loss += loss.item() * bs
        acc += accuracy(logits, labels) * bs
        n += bs
    return epoch_loss / n, acc / n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    epoch_loss = acc = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        bs = imgs.size(0)
        epoch_loss += loss.item() * bs
        acc += accuracy(logits, labels) * bs
        n += bs
    return epoch_loss / n, acc / n


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, class_names = build_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    model = build_model(len(class_names), args.freeze_backbone).to(device)
    optim = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    log = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optim, scaler, device)
        val_loss, val_acc = validate(model, val_dl, device)
        scheduler.step()

        log.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4%} time={time.time()-t0:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"model": model.state_dict(), "classes": class_names},
                Path(args.output_dir) / "best_model.pth",
            )

    # save training log
    with open(Path(args.output_dir) / "history.json", "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
