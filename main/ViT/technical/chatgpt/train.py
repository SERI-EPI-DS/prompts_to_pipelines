#!/usr/bin/env python3
"""
train.py
-----------
Script to fine‑tune a Swin‑V2‑B model for image classification on a custom
dataset organised into training, validation and test splits.  The dataset
should follow the standard ImageFolder layout under the provided data root
directory:

    root/train/<class_name>/<image_files>
    root/val/<class_name>/<image_files>

This script constructs appropriate torchvision datasets and dataloaders,
applies a suite of data augmentations during training, and uses label
smoothing cross‑entropy loss to supervise the network.  The model weights
initialise from an ImageNet‑1K pretrained checkpoint provided by
torchvision (``Swin_V2_B_Weights.IMAGENET1K_V1``【990691079611788†L90-L161】).  Training runs for a maximum of
``--epochs`` epochs (default: 30) with an AdamW optimiser and cosine
annealing learning rate schedule.  The best performing model on the
validation set (based on top‑1 accuracy) is stored to the output
directory.

Usage:
    python train.py --data_root /path/to/data --output_dir ./results

Optional arguments allow control over batch size, number of epochs,
learning rate, weight decay and label smoothing.  For example:

    python train.py \
        --data_root /data/fundus \
        --output_dir ./project/results \
        --batch_size 32 \
        --epochs 50 \
        --lr 3e-5 \
        --label_smoothing 0.1

The script will create the output directory if it does not exist and
write the best model checkpoint to ``best_model.pth`` within that
directory.
"""

import argparse
import csv
import logging
import os
import sys
import time
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import swin_v2_b, Swin_V2_B_Weights


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine‑tune Swin‑V2‑B on a custom dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing 'train' and 'val' folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store checkpoints and logs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of images per batch (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (default: 50, max allowed: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate for the AdamW optimiser (default: 3e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for AdamW optimiser (default: 0.05)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for the cross‑entropy loss (default: 0.1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Constrain epochs to the maximum allowed value of 50
    if args.epochs > 50:
        print(
            f"Number of epochs {args.epochs} exceeds the maximum allowed 50. "
            f"Clipping to 50."
        )
        args.epochs = 50
    return args


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def get_data_loaders(
    data_root: str, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    """Create DataLoaders for the training and validation sets.

    Args:
        data_root: Root directory containing 'train' and 'val' subdirectories.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of train DataLoader, validation DataLoader, number of classes and list of class names.
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    # Normalisation values used by ImageNet pretrained models【990691079611788†L158-L162】
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training augmentations: resize, random crop, flips, colour jitter and RandAugment
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                (272, 272), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomResizedCrop(256, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Validation transforms: resize and centre crop to 256
    val_transforms = transforms.Compose(
        [
            transforms.Resize(
                (272, 272), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes, class_names


def build_model(num_classes: int) -> nn.Module:
    """Load a pretrained Swin‑V2‑B model and replace the classification head.

    Args:
        num_classes: Number of target classes.

    Returns:
        Model ready for fine‑tuning on the specified number of classes.
    """
    # Load ImageNet‑1K pretrained weights【990691079611788†L90-L161】
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    model = swin_v2_b(weights=weights)

    # Replace the classification head to match the number of classes
    # The Swin transformer stores the classification head in the attribute 'head'
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for a single epoch.

    Args:
        model: Model to train.
        dataloader: DataLoader for the training data.
        criterion: Loss function.
        optimizer: Optimiser for updating model parameters.
        scaler: GradScaler for mixed precision training.
        device: Device on which to run computations.

    Returns:
        Tuple of average loss and accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backpropagation with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model on a validation set.

    Args:
        model: Model to evaluate.
        dataloader: DataLoader for the validation data.
        criterion: Loss function.
        device: Device on which to run computations.

    Returns:
        Tuple of average validation loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main() -> None:
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(args.output_dir, "training.log")
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    # Seed for reproducibility
    set_seed(args.seed)

    # DataLoaders
    logging.info(f"Loading data from {args.data_root}...")
    train_loader, val_loader, num_classes, class_names = get_data_loaders(
        args.data_root, args.batch_size, args.num_workers
    )
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Class names: {class_names}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Model
    model = build_model(num_classes)
    model.to(device)

    # Loss and optimiser
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        logging.info(f"Epoch {epoch}/{args.epochs}")

        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        duration = time.time() - start_time
        logging.info(
            f"Epoch {epoch}: Train loss={train_loss:.4f}, Train acc={train_acc:.4f}, "
            f"Val loss={val_loss:.4f}, Val acc={val_acc:.4f}, Duration={duration/60:.2f} min"
        )

        # Save the model if it improves on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "num_classes": num_classes,
                    "class_names": class_names,
                },
                best_model_path,
            )
            logging.info(
                f"Saved new best model at epoch {epoch} with val acc {val_acc:.4f}"
            )

    logging.info(
        f"Training complete. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}"
    )


if __name__ == "__main__":
    main()
