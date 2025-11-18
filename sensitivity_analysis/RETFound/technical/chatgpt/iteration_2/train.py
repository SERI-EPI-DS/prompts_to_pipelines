#!/usr/bin/env python3
"""
Fine‑tune a RETFound ViT‑L classifier on a custom colour fundus dataset.

This script prepares PyTorch data loaders for the expected folder layout,
builds a Vision Transformer based on the RETFound foundation model,
loads the provided pre‑trained weights, trains the model using
cross‑entropy loss with label smoothing, and writes out the best
model parameters.  It makes minimal assumptions about the training
hardware; if a CUDA device is available it will be used automatically.

Folder layout (relative to `--data_path`):

  data/
    train/
      class_1/  image files...
      class_2/  image files...
      ...
    val/
      class_1/  image files...
      class_2/  image files...
      ...
    test/     (not touched during training)

The `RETFound` repository should live next to the `project` folder as
described in the problem statement.  This script appends the path to
that repository to `sys.path` so that it can import the `models_vit`
module.  All configuration parameters can be overridden on the
command line; sensible defaults are provided for common use cases.

Example:

    python train.py --data_path ../data \
                    --weights ../RETFound/RETFound_CFP_weights.pth \
                    --output_dir ../project/results

This will fine‑tune the model for up to 50 epochs and write
`best_model.pth` into the specified output directory.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with sensible defaults.
    """
    parser = argparse.ArgumentParser(description="Fine‑tune RETFound ViT‑L classifier")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "data"),
        help="Root folder containing train/val/test subdirectories.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent
            / "RETFound"
            / "RETFound_CFP_weights.pth"
        ),
        help="Path to RETFound foundation model weights (.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent / "project" / "results"
        ),
        help="Directory in which to store training outputs and best model.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Square input image size (pixels) for the Vision Transformer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Mini‑batch size for training and validation.  Adjust according to GPU memory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.  Will stop earlier if interrupted.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 penalty) for the optimizer.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for cross‑entropy loss.",
    )
    return parser.parse_args()


def build_transforms(input_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Construct data augmentation and evaluation transforms.

    Parameters
    ----------
    input_size : int
        Size to which images will be resized.

    Returns
    -------
    Tuple[transforms.Compose, transforms.Compose]
        Training and validation transforms respectively.
    """
    # The mean/std used here follow ImageNet statistics.  The RETFound model
    # was trained using a similar normalisation so these values are
    # appropriate for fine‑tuning.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                input_size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, val_transform


def load_datasets(
    data_path: str, input_size: int, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    """Load training and validation datasets.

    Parameters
    ----------
    data_path : str
        Root directory of the dataset containing `train` and `val` subfolders.
    input_size : int
        Desired spatial resolution for input images.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of worker threads for data loading.

    Returns
    -------
    Tuple[DataLoader, DataLoader, int, List[str]]
        Training loader, validation loader, number of classes and list of
        class names in alphabetical order.
    """
    train_transform, val_transform = build_transforms(input_size)
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory '{train_dir}' not found")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory '{val_dir}' not found")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
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
        drop_last=False,
    )
    return train_loader, val_loader, num_classes, class_names


def build_model(num_classes: int, input_size: int, weights_path: str) -> nn.Module:
    """Construct the Vision Transformer and load RETFound weights.

    Parameters
    ----------
    num_classes : int
        Number of output classes for the classification head.
    input_size : int
        Input image size for the vision transformer.
    weights_path : str
        Path to the pre‑trained RETFound weights file.

    Returns
    -------
    nn.Module
        A Vision Transformer ready for fine‑tuning.
    """
    # Insert RETFound modules into the import path.  The repository is
    # expected to be located two levels above this script inside `RETFound`.
    script_dir = Path(__file__).resolve().parent
    retfound_dir = script_dir.parent.parent / "RETFound"
    if not retfound_dir.is_dir():
        raise FileNotFoundError(
            f"Expected RETFound directory at '{retfound_dir}', but it was not found."
        )
    sys.path.insert(0, str(retfound_dir))
    try:
        from models_vit import RETFound_mae  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import RETFound models. Ensure the RETFound repository "
            "is present adjacent to the project directory."
        ) from exc
    # Instantiate the base Vision Transformer with classification head
    model = RETFound_mae(
        img_size=input_size,
        num_classes=num_classes,
        drop_path_rate=0.0,
        global_pool=True,
    )
    # Load pre‑trained weights if available
    if weights_path and os.path.isfile(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        # Some checkpoints wrap the model weights in a dict under 'model'
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        # Remove mismatched classifier weights to allow reinitialisation
        for key in ["head.weight", "head.bias"]:
            if key in state_dict and state_dict[key].shape[0] != num_classes:
                del state_dict[key]
        # Interpolate missing position embeddings is handled internally by model
        _ = model.load_state_dict(state_dict, strict=False)
    # Reinitialise the classification head for our number of classes
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        # Initialise with small random weights (similar to timm default)
        nn.init.trunc_normal_(model.head.weight, std=2e-5)
        if model.head.bias is not None:
            nn.init.zeros_(model.head.bias)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run a single epoch of training.

    Returns
    -------
    Tuple[float, float]
        Average loss and accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / total
    avg_acc = running_correct / total if total > 0 else 0.0
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation set.

    Returns
    -------
    Tuple[float, float]
        Average loss and accuracy on the validation set.
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / total
    avg_acc = running_correct / total if total > 0 else 0.0
    return avg_loss, avg_acc


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare data loaders
    train_loader, val_loader, num_classes, class_names = load_datasets(
        args.data_path, args.input_size, args.batch_size, args.num_workers
    )
    # Build model
    model = build_model(num_classes, args.input_size, args.weights)
    model.to(device)
    # Loss and optimiser
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine annealing learning rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch + 1:2d}/{args.epochs}: "
            f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.4f}",
            flush=True,
        )
        # Save the best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "input_size": args.input_size,
                },
                save_path,
            )
    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
