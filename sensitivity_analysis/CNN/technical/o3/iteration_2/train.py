#!/usr/bin/env python3
"""
Fine-tunes a ConvNeXt-L model on colour fundus photographs (patched).

Fixes
-----
* Sets TORCHVISION_DISABLE_ONNX_RUNTIME=1 to bypass a torchvision-onnxruntime
  import bug that raised “_ARRAY_API not found”.
* Removes the hue term from ColorJitter to avoid an OverflowError in
  torchvision.transforms.adjust_hue.
"""

import os

# --------------------------------------------------------------------------- #
# Work-around: prevent torchvision from importing onnxruntime at import time. #
# --------------------------------------------------------------------------- #
os.environ.setdefault("TORCHVISION_DISABLE_ONNX_RUNTIME", "1")

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# --------------------------------------------------------------------------- #
# CLI                                                                       #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ConvNeXt-L on fundus images")
    p.add_argument("--data_dir", required=True, help="Root with train/ and val/")
    p.add_argument("--results_dir", required=True, help="Where to store checkpoints")
    p.add_argument("--epochs", type=int, default=50, help="≤ 50")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", help="Enable mixed-precision")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data                                                                      #
# --------------------------------------------------------------------------- #
def dataloaders(root: str, batch: int, workers: int):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.ImageFolder(Path(root) / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(Path(root) / "val", transform=val_tfms)

    train_ld = DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True
    )
    val_ld = DataLoader(
        val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True
    )
    return train_ld, val_ld, train_ds.classes, train_ds.class_to_idx


# --------------------------------------------------------------------------- #
# Model                                                                     #
# --------------------------------------------------------------------------- #
def build_model(num_classes: int) -> nn.Module:
    try:
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        net = models.convnext_large(weights=weights)
    except AttributeError:
        net = models.convnext_large(pretrained=True)

    in_feat = net.classifier[2].in_features
    net.classifier[2] = nn.Linear(in_feat, num_classes)
    return net


# --------------------------------------------------------------------------- #
# Train / eval loops                                                        #
# --------------------------------------------------------------------------- #
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    tot_loss, tot_correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        preds = logits.argmax(1)
        tot_loss += loss.item() * x.size(0)
        tot_correct += (preds == y).sum().item()
        n += x.size(0)
    return tot_loss / n, tot_correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss, tot_correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(1)
        tot_loss += loss.item() * x.size(0)
        tot_correct += (preds == y).sum().item()
        n += x.size(0)
    return tot_loss / n, tot_correct / n


# --------------------------------------------------------------------------- #
# Main                                                                      #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    train_ld, val_ld, class_names, class_to_idx = dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    with open(Path(args.results_dir) / "class_indices.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = amp.GradScaler(enabled=args.amp)

    best_acc, best_epoch = 0.0, -1
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_epoch(
            model, train_ld, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_ld, criterion, device)
        scheduler.step()
        dt = time.perf_counter() - t0
        print(
            f"Epoch {epoch+1}/{args.epochs} | {dt:.1f}s "
            f"Train {tr_loss:.4f}/{tr_acc:.4f}  "
            f"Val {val_loss:.4f}/{val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "epoch": best_epoch,
                    "val_acc": best_acc,
                },
                Path(args.results_dir) / "model_best.pth",
            )

    print(f"Done. Best val acc {best_acc:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
