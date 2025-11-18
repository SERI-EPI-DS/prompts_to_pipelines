#!/usr/bin/env python3
"""
train.py
==========

This script fine‑tunes a Swin‑V2‑B classifier on a custom dataset of
colour fundus photographs.  It constructs the required datasets and
dataloaders, applies sensible augmentations during training, trains
the model with cross‑entropy loss and label smoothing and saves the
best model weights to disk.  Hyperparameters such as the number of
epochs, learning rate, batch size and locations of the data and
results directories can be supplied via command‑line arguments.

The training procedure is designed for a modern GPU (e.g. an RTX 3090
with 24 GB of VRAM) and utilises mixed precision to accelerate
training while conserving memory.  It also relies on the pretrained
weights for Swin‑V2‑B provided by TorchVision when available.

Example usage
-------------

Assuming your dataset lives under ``/data`` with ``train`` and
``val`` subfolders and you want to save outputs under ``/workspace/results``
you can run:

    python train.py --data_dir /data --results_dir /workspace/results \
        --epochs 50 --batch_size 16 --lr 1e-4

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def parse_args() -> argparse.Namespace:
    """Parse the command‑line arguments for training.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fine‑tune a Swin‑V2‑B classifier on a dataset of fundus "
            "photographs.  The data directory must contain 'train' and 'val' "
            "subdirectories organised in the standard ImageFolder layout."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root of the dataset containing train/val/test folders",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path where models and logs will be saved",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini‑batch size used during training (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate for the optimiser (default: 3e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay (L2 penalty) for the optimiser (default: 1e-2)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading (default: 4)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Amount of label smoothing to apply in the loss (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed random number generators for reproducible results.

    Parameters
    ----------
    seed : int
        The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When using deterministic algorithms the performance might be lower but
    # results become reproducible. For this fine‑tuning we use deterministic
    # behaviour for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(
    data_dir: str, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, list[str]]:
    """Create training and validation dataloaders.

    Parameters
    ----------
    data_dir : str
        Root directory containing 'train' and 'val' subfolders.
    batch_size : int
        Number of images per mini‑batch.
    num_workers : int
        Number of worker processes to use for data loading.

    Returns
    -------
    Tuple[DataLoader, DataLoader, list[str]]
        A tuple containing the training dataloader, validation dataloader
        and the list of class names.
    """
    # Define image normalisation constants for ImageNet
    # TorchVision's Swin‑V2 weights use mean and standard deviation as follows
    # 【885833164807516†L155-L162】.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Apply a range of augmentations during training.  RandomResizedCrop
    # introduces scale variation, horizontal/vertical flips help the model
    # generalise to mirrored eyes, random rotation covers rotated fundus
    # images and colour jitter adjusts brightness and colour distribution.
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=224, scale=(0.8, 1.0), ratio=(0.75, 1.3333)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # For validation we follow the inference transforms recommended for
    # Swin‑V2‑B weights, resizing to 256 and centre cropping to 224, then
    # normalising【885833164807516†L155-L162】.
    val_transforms = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        raise FileNotFoundError(
            f"Expected 'train' and 'val' directories inside {data_dir}, "
            f"but they were not found."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes


def build_model(num_classes: int) -> nn.Module:
    """Instantiate a Swin‑V2‑B model for fine‑tuning.

    Parameters
    ----------
    num_classes : int
        The number of target classes.

    Returns
    -------
    nn.Module
        The initialised model ready for training.
    """
    # Try to load pretrained weights if available.  TorchVision 0.18
    # exposes Swin_V2_B_Weights and using DEFAULT leverages
    # ImageNet pretrained weights.  If weights cannot be imported
    # (e.g. due to version mismatch) we fall back to None.
    weights = None
    try:
        from torchvision.models import Swin_V2_B_Weights

        weights = Swin_V2_B_Weights.DEFAULT
    except (ImportError, AttributeError):
        # The Swin_V2_B_Weights class may not exist in older TorchVision
        # versions; in that case we continue without loading pretrained
        # weights.  Training will still work but may require more epochs
        # to converge.
        weights = None

    model = models.swin_v2_b(weights=weights)
    # Replace the classification head.  The original head outputs 1000
    # features by default (for ImageNet).  We swap it for a new linear
    # layer with output dimension equal to the number of classes.
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    """Perform one epoch of training.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    loader : DataLoader
        The training dataloader.
    criterion : nn.Module
        The loss function.
    optimizer : optim.Optimizer
        The optimiser to update the model parameters.
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler for mixed precision training.
    device : torch.device
        The device on which computations are performed.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the average loss and accuracy over the epoch.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model on a validation set.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loader : DataLoader
        The validation dataloader.
    criterion : nn.Module
        The loss function.
    device : torch.device
        The device on which computations are performed.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)
    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Create results directory
    results_path = Path(args.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare dataloaders
    train_loader, val_loader, class_names = create_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    num_classes = len(class_names)

    # Build model
    model = build_model(num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss, optimiser and scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # It's good practice to only optimise parameters that require gradients
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params_to_update, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = results_path / "best_model.pth"
    # Store training history
    history = []  # list of tuples: (epoch, train_loss, train_acc, val_loss, val_acc)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append((epoch, train_loss, train_acc, val_loss, val_acc))
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                best_model_path,
            )

    # Save final checkpoint
    final_model_path = results_path / "last_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "class_names": class_names,
            "epoch": args.epochs,
            "val_acc": val_acc,
        },
        final_model_path,
    )

    # Write training history to a CSV file for later analysis
    history_path = results_path / "training_history.csv"
    try:
        import csv

        with open(history_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
            )
            writer.writerows(history)
    except Exception as e:
        # Don't crash if writing fails; just emit a warning
        print(f"Warning: failed to save training history: {e}")

    print(
        f"Training complete. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}. "
        f"Best model saved to {best_model_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
