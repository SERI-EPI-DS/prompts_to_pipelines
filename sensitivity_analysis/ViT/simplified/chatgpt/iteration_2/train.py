#!/usr/bin/env python3
"""
train_swinv2.py
================

This script fine‑tunes a Swin Transformer V2 base model for image classification.

The code is built with PyTorch and the `timm` library to access a state‑of‑the‑art
SwinV2 architecture. It assumes your dataset is laid out with one folder per
class inside a training, validation and optional test directory, for example::

    my_dataset/
        train/
            class_a/
                image1.png
                ...
            class_b/
                ...
        val/
            class_a/
                ...
            class_b/
                ...

You can override the names of the train/val subfolders using the
``--train-folder`` and ``--val-folder`` command line arguments if your data
structure differs.

The script exposes several configurable parameters, such as batch size,
learning rate and number of epochs. It uses sensible defaults and performs
basic data augmentation for the training set (random resized crops and
horizontal flips) while using deterministic centre crops during validation.

During training, the script tracks the best validation accuracy and saves
checkpoints in the specified ``--output-dir`` (default: ``./checkpoints``). It
also logs loss and accuracy after each epoch. If a GPU is available, it will
automatically move the model and tensors to GPU for faster training.

Usage example (using default folder names):

    python train_swinv2.py --data-dir /path/to/my_dataset --output-dir ./models

To change folder names or other hyperparameters:

    python train_swinv2.py \
        --data-dir /data/fundus_dataset \
        --train-folder train_images \
        --val-folder validation_images \
        --epochs 20 \
        --batch-size 16 \
        --learning-rate 1e-4 \
        --model-name swinv2_base_patch4_window16_256

Requirements:
* PyTorch ≥ 1.10
* timm ≥ 0.6.12
* torchvision
* Pillow

"""

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for this script. Install with `pip install timm`."
    ) from e


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine‑tune a Swin Transformer V2 model for image classification."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory of the dataset containing train/val/test folders.",
    )
    parser.add_argument(
        "--train-folder",
        type=str,
        default="train",
        help="Name of the subfolder inside data_dir containing training images.",
    )
    parser.add_argument(
        "--val-folder",
        type=str,
        default="val",
        help="Name of the subfolder inside data_dir containing validation images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save trained models and logs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini‑batch size for training and validation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 penalty).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="swinv2_base_window16_256",
        help=(
            "Name of the SwinV2 model to use from timm. "
            "Try models like 'swinv2_base_window16_256' or 'swinv2_base_window8_256'. "
            "If your timm version appends a training dataset suffix (e.g. '.ms_in1k'), include it."
        ),
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Input image size for the model. Recommended: 192, 224 or 256.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print loss and accuracy every N batches during training.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a model checkpoint to resume training from.",
    )
    return parser.parse_args()


def create_dataloaders(
    data_dir: str,
    train_folder: str,
    val_folder: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders.

    Parameters
    ----------
    data_dir: str
        Root dataset directory containing the train and validation folders.
    train_folder: str
        Name of the training subdirectory.
    val_folder: str
        Name of the validation subdirectory.
    img_size: int
        Input image size (height and width) for the model.
    batch_size: int
        Batch size for training and validation.
    num_workers: int
        Number of worker processes used by DataLoader.

    Returns
    -------
    train_loader: DataLoader
        PyTorch DataLoader for the training set.
    val_loader: DataLoader
        PyTorch DataLoader for the validation set.
    """
    # Normalization statistics from ImageNet; Swin models are pretrained on ImageNet-22k
    # and often use these stats. If your data distribution differs greatly, consider
    # recomputing mean and std on your dataset.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Train transforms: random resized crop and horizontal flip for augmentation.
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    # Validation transforms: resize then centre crop.
    val_transforms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_path = os.path.join(data_dir, train_folder)
    val_path = os.path.join(data_dir, val_folder)

    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"Training folder not found: {train_path}")
    if not os.path.isdir(val_path):
        raise FileNotFoundError(f"Validation folder not found: {val_path}")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transforms)

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

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> Tuple[float, float]:
    """Train the model for one epoch.

    Returns average loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

        # Print training progress
        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / total
            avg_acc = running_corrects / total
            logging.info(
                f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on the validation set."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def main() -> None:
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting training with arguments: %s", args)

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        train_folder=args.train_folder,
        val_folder=args.val_folder,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(train_loader.dataset.classes)
    logging.info(f"Number of classes: {num_classes}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Build the model
    logging.info(f"Loading model {args.model_name} (pretrained)")
    # Try to create the requested model. Some timm versions use a suffix such as
    # `.ms_in1k` to denote ImageNet-1k pretraining. If the requested model is
    # unknown, automatically attempt to append `.ms_in1k` as a fallback.
    try:
        model = timm.create_model(
            args.model_name,
            pretrained=True,
            num_classes=num_classes,
            img_size=args.img_size,
        )
    except RuntimeError as e:
        if "Unknown model" in str(e) and not args.model_name.endswith(".ms_in1k"):
            fallback_name = f"{args.model_name}.ms_in1k"
            logging.warning(
                f"Model '{args.model_name}' not found in timm. Trying fallback '{fallback_name}'."
            )
            try:
                model = timm.create_model(
                    fallback_name,
                    pretrained=True,
                    num_classes=num_classes,
                    img_size=args.img_size,
                )
                args.model_name = fallback_name  # Update for saving checkpoint
            except Exception:
                # List similar models to aid the user
                available = [m for m in timm.list_models("swinv2") if "base" in m]
                raise RuntimeError(
                    f"Could not find model '{args.model_name}' or fallback '{fallback_name}'. "
                    f"Ensure you are using a timm version that supports SwinV2. "
                    f"Available SwinV2 base models include: {available}"
                ) from None
        else:
            raise
    model.to(device)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # Optional: Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_acc = 0.0
    # Optionally resume from a checkpoint
    if args.resume is not None and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint.get("scheduler_state_dict", {}))
        start_epoch = checkpoint.get("epoch", 0)
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        logging.info(
            f"Resumed training from {args.resume} (epoch {start_epoch}, best val acc {best_val_acc:.4f})"
        )

    # Training loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.log_interval,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        logging.info(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save checkpoint if this is the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / f"best_model_epoch{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "classes": train_loader.dataset.classes,
                    "img_size": args.img_size,
                    "model_name": args.model_name,
                },
                checkpoint_path,
            )
            logging.info(
                f"Saved new best model to {checkpoint_path} with val acc {best_val_acc:.4f}"
            )


if __name__ == "__main__":
    main()
