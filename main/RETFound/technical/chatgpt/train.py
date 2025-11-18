"""
Training script for fine-tuning the RETFound MAE vision transformer on a
classification task.  This script expects a directory layout with separate
sub‑folders for training and validation data under a single root.  Each class
should reside in its own folder.  The model weights produced by fine‑tuning
are saved to the specified output directory.

Usage example:

```
python train.py \
    --data_root /path/to/dataset \
    --output_dir /path/to/save/results \
    --weights /path/to/RETFound_CFP_weights.pth \
    --epochs 30 \
    --batch_size 32 \
    --lr 5e-5
```

The script will automatically determine the number of classes from the
sub‑directories of the training folder and will adapt the classifier head
accordingly.  A cosine annealing learning rate schedule is employed and
mixed‑precision training is enabled via ``torch.cuda.amp``.  Validation
performance (accuracy) is monitored at the end of every epoch and the best
performing model is saved to ``finetuned_model.pth`` in the output directory.

Note: this script relies on the ``RETFound_mae`` definition provided in the
RETFound repository.  The default patch size (16), embedding dimension
(1024) and depth (24) originate from the model definition【463435877362505†L259-L279】.
"""

import argparse
import csv
import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
Attempt to import the RETFound Vision Transformer.  If the RETFound package
cannot be found on the Python path, this block looks for a folder named
``RETFound`` relative to this script's location and temporarily adds it to
``sys.path``.  This fallback accommodates cases where the repository has
been cloned alongside this script but is not installed as a Python package.
"""
try:
    from RETFound.models_vit import RETFound_mae  # type: ignore
except ImportError:
    import sys
    import pathlib

    # Compute a candidate path two directories above this file (project/code -> project -> root)
    _current_file = pathlib.Path(__file__).resolve()
    # Search upwards for a directory named 'RETFound'
    for parent in _current_file.parents:
        candidate = parent / "RETFound"
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
            try:
                from models_vit import RETFound_mae  # type: ignore

                break
            except ImportError:
                # Remove the injected path if import fails and continue searching
                sys.path.pop(0)
                continue
    else:
        # If we exit the loop without breaking, raise a helpful error
        raise ImportError(
            "Unable to locate RETFound.models_vit. Please set PYTHONPATH to include "
            "the RETFound repository or clone it adjacent to this script."
        )


def set_seed(seed: int) -> None:
    """Helper to fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create the training and validation transformations.

    Args:
        img_size: The target size (height and width) for the images.

    Returns:
        A tuple ``(train_transform, val_transform)``.
    """
    # Data augmentation for training: random resize/crop, horizontal flip,
    # rotation and colour jitter provide robustness to viewpoint and
    # illumination changes.
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            # Standard ImageNet normalisation values; these work well for natural
            # images and are a reasonable default for colour fundus photos.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # For validation, apply deterministic resizing and centre cropping to
    # minimise distribution shift between training and evaluation.
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),  # resize shorter side
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


def prepare_data_loaders(
    data_root: str, batch_size: int, img_size: int
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Prepare training and validation dataloaders.

    Args:
        data_root: Root directory containing ``train`` and ``val`` sub‑folders.
        batch_size: Number of images per batch.
        img_size: Target size for image transformations.

    Returns:
        Tuple of training DataLoader, validation DataLoader and mapping from
        class names to indices.
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Expected 'train' and 'val' folders inside {data_root}. "
            f"Got train_dir={train_dir}, val_dir={val_dir}"
        )
    train_transform, val_transform = build_transforms(img_size)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    class_to_idx = train_dataset.class_to_idx
    # Set persistent workers for stability on GPU servers and pin memory for
    # faster host‑to‑device transfers.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader, class_to_idx


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model on a validation set.

    Returns the average loss and accuracy.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
            total += labels.size(0)
    avg_loss = running_loss / total if total > 0 else 0.0
    avg_acc = (running_corrects.double() / total).item() if total > 0 else 0.0
    return avg_loss, avg_acc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    """Train the model for a single epoch and return the average loss."""
    model.train()
    running_loss = 0.0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
    return running_loss / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine‑tune RETFound MAE model for classification"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing train/val/test folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save models and metrics",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to RETFound pre‑trained weights (RETFound_CFP_weights.pth)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of epochs for fine‑tuning (default: 50)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Mini‑batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Initial learning rate (default: 5e‑5)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for optimizer (default: 0.05)",
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Input image size (default: 224)"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for cross‑entropy loss (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_to_idx = prepare_data_loaders(
        args.data_root, args.batch_size, args.img_size
    )
    num_classes = len(class_to_idx)

    # Instantiate the backbone model.  The ``img_size`` argument adjusts
    # positional embeddings to the chosen resolution.
    model = RETFound_mae(img_size=args.img_size)
    # Replace the classification head with a new fully connected layer sized
    # according to the number of classes.
    in_features = model.head.in_features if hasattr(model, "head") else model.embed_dim
    model.head = nn.Linear(in_features, num_classes)

    # Load pre‑trained weights if provided.  We disable strict loading so that
    # missing keys (e.g. classifier head) do not trigger an error.
    if args.weights is not None:
        print(f"Loading pre‑trained weights from {args.weights}")
        state_dict = torch.load(args.weights, map_location="cpu")
        # Strip 'module.' prefix if present (from DataParallel checkpoints).
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            cleaned_state_dict[new_key] = v
        msg = model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"Loaded weights with message: {msg}")

    model = model.to(device)

    # Define loss function with label smoothing and optimizer.
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    best_epoch = -1
    metrics_path = os.path.join(args.output_dir, "training_metrics.csv")
    # Write header for metrics CSV
    with open(metrics_path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        duration = time.time() - start_time
        print(
            f"Epoch {epoch+1:03d}/{args.epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, time={duration:.1f}s"
        )
        # Append metrics
        with open(metrics_path, "a", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_acc])
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_path = os.path.join(args.output_dir, "finetuned_model.pth")
            torch.save(model.state_dict(), best_path)
    print(
        f"Training complete. Best validation accuracy {best_acc:.4f} achieved at epoch {best_epoch}."
    )


if __name__ == "__main__":
    main()
