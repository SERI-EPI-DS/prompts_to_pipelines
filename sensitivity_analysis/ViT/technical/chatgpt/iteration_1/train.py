"""
train.py
-----------

This script fine‑tunes a Swin‑V2‑B classifier on a custom image dataset.  It
supports configurable input and output directories via CLI arguments and
automatically adapts the classifier head to the number of classes in the
training set.  The training pipeline includes common data augmentations,
uses label‑smoothed cross‑entropy loss, employs the AdamW optimizer with
cosine annealing learning rate scheduling, and leverages automatic mixed
precision (AMP) for improved performance on NVIDIA GPUs.  The script
tracks validation accuracy and saves the weights of the best performing
model into the specified results directory.  A list of class names is
persisted to `classes.txt` in the results folder to ensure consistency
between training and testing.

Usage example:

```
python train.py --data-dir /path/to/data \
               --output-dir /path/to/results \
               --epochs 30 \
               --batch-size 32
```

The data directory must contain `train` and `val` subfolders, each of which
should be organised as class‑specific subdirectories following the
``torchvision.datasets.ImageFolder`` convention.  The results directory will
be created if it does not already exist.
"""

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Fine‑tune Swin‑V2‑B classifier on a custom dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing 'train' and 'val' subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save model weights and logs",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini‑batch size for training and validation",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for cross entropy loss",
    )
    return parser.parse_args()


def create_dataloaders(
    data_dir: str, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, list]:
    """
    Create training and validation dataloaders and return them along with the
    class names.  Applies data augmentations consistent with the Swin‑V2
    model's expected input normalization.

    Args:
        data_dir: Path to the dataset root containing 'train' and 'val' folders.
        batch_size: Number of images per mini‑batch.
        num_workers: Number of worker processes for data loading.

    Returns:
        A tuple of (train_loader, val_loader, class_names).
    """
    # Use the built‑in weight configuration to obtain normalization stats
    weights = models.Swin_V2_B_Weights.DEFAULT
    # Some versions of torchvision do not expose mean/std in the meta
    # attribute, so provide ImageNet defaults when missing
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    try:
        meta = weights.meta  # type: ignore[attr-defined]
        mean = meta.get("mean", default_mean)
        std = meta.get("std", default_std)
    except Exception:
        mean, std = default_mean, default_std

    # Training transforms: random resized crop, horizontal/vertical flips,
    # color jitter and normalization.  These augmentations can improve
    # generalisation for medical imaging tasks such as fundus photography.
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=256,
                scale=(0.8, 1.0),
                ratio=(0.8, 1.25),
                interpolation=transforms.InterpolationMode.BICUBIC,
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

    # Validation transforms: resize followed by centre crop and normalisation.
    val_transforms = transforms.Compose(
        [
            transforms.Resize(
                size=272, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

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
    """
    Load the pretrained Swin‑V2‑B model and replace the classifier head to
    accommodate the number of classes in the target dataset.

    Args:
        num_classes: Number of output classes for classification.

    Returns:
        A PyTorch neural network model ready for fine‑tuning.
    """
    # Load model with ImageNet pretrained weights
    weights = models.Swin_V2_B_Weights.DEFAULT
    model = models.swin_v2_b(weights=weights)
    # The classifier head is a linear layer named 'head'
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[float, float]:
    """
    Train the model for a single epoch.

    Returns:
        A tuple of (average_loss, accuracy).
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
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set.

    Returns:
        A tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()

    # Ensure the results directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataloaders and get class names
    train_loader, val_loader, class_names = create_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    # Save class names for later use in testing
    classes_path = os.path.join(args.output_dir, "classes.txt")
    with open(classes_path, "w") as f:
        for cls in class_names:
            f.write(cls + "\n")

    # Build the model and move to device
    model = build_model(num_classes=len(class_names))
    model = model.to(device)

    # Define loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimiser and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_acc = 0.0
    # Resume from checkpoint if provided
    if args.resume is not None and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_acc = checkpoint.get("best_acc", 0.0)
        print(f"Resumed training from epoch {start_epoch} with best_acc={best_acc:.4f}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_acc": best_acc,
                    "classes": class_names,
                },
                checkpoint_path,
            )
            print(f"Saved new best model with Val Acc: {best_acc:.4f}")

        scheduler.step()

    # Always save the final model after training completes
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": args.epochs,
            "best_acc": best_acc,
            "classes": class_names,
        },
        final_model_path,
    )
    print(f"Training completed. Best Val Acc: {best_acc:.4f}. Final model saved.")


if __name__ == "__main__":
    main()
