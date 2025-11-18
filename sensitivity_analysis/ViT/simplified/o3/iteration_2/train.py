# =====================
# train.py — fine‑tune Swin‑V2‑B (robust to older TorchVision)
# =====================
"""
Example
-------
python train.py \
  --data_dir /path/to/dataset \
  --output_dir ./outputs \
  --epochs 40 \
  --batch_size 32

Changes (2025‑09‑22)
--------------------
* Added **graceful fallback** when `Swin_V2_B_Weights` or its metadata are unavailable (older TorchVision).
* Optional `--no_pretrained` flag to disable ImageNet initialisation.
* Mean/STD now default to ImageNet values if weight metadata missing.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import swin_v2_b

# Optional – TorchVision ≥ 0.16 provides these weights.
try:
    from torchvision.models import Swin_V2_B_Weights  # type: ignore
except ImportError:  # older TorchVision
    Swin_V2_B_Weights = None  # type: ignore


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _imagenet_stats() -> tuple[list[float], list[float]]:
    """Return ImageNet mean / std (RGB)."""
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_data_loaders(data_dir: Path, batch_size: int, num_workers: int = 4):
    # Obtain normalisation values from weight meta if available, else default.
    if Swin_V2_B_Weights and hasattr(Swin_V2_B_Weights, "IMAGENET1K_V1"):
        mean = Swin_V2_B_Weights.IMAGENET1K_V1.meta.get("mean", _imagenet_stats()[0])
        std = Swin_V2_B_Weights.IMAGENET1K_V1.meta.get("std", _imagenet_stats()[1])
    else:
        mean, std = _imagenet_stats()

    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfm)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tfm)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dl, val_dl, train_ds.classes


def build_model(num_classes: int, use_pretrained: bool):
    weights = None
    if (
        use_pretrained
        and Swin_V2_B_Weights
        and hasattr(Swin_V2_B_Weights, "IMAGENET1K_V1")
    ):
        weights = Swin_V2_B_Weights.IMAGENET1K_V1
    model = swin_v2_b(weights=weights)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def evaluate(model: nn.Module, dl: DataLoader, device: torch.device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, classes = get_data_loaders(
        data_dir, args.batch_size, args.num_workers
    )

    # Persist class‑index mapping for inference.
    (output_dir / "class_indices.json").write_text(
        json.dumps({i: c for i, c in enumerate(classes)}, indent=2)
    )

    model = build_model(len(classes), use_pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 50 == 0 or step == len(train_dl):
                print(
                    f"Epoch[{epoch}/{args.epochs}] step[{step}/{len(train_dl)}] loss: {running_loss/step:.4f}"
                )

        scheduler.step()
        val_acc = evaluate(model, val_dl, device)
        print(f"Epoch {epoch}: val_acc = {val_acc:.4f}")
        torch.save(model.state_dict(), output_dir / "last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best.pt")
            print("[+] New best model saved.")

    print(f"Training complete. Best val accuracy: {best_acc:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine‑tune Swin‑V2‑B on colour fundus photographs"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Dataset root containing train/val folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save checkpoints & logs",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--no_pretrained", action="store_true", help="Disable ImageNet initialisation"
    )

    train(parser.parse_args())
