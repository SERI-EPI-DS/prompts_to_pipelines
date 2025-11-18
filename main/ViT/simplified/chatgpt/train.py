"""
train_swinv2.py
================

This script provides a convenient entry point for fine‑tuning a Swin
Transformer V2 base model on a custom image classification dataset.  It
relies on the `timm` library to instantiate a pre‑trained model and
leverages PyTorch for the training loop.  The expected dataset layout
is ``root/train``, ``root/val`` and ``root/test`` with each split
containing sub‑directories named after the class labels and filled with
images.  When run, the script will create the necessary data
transformations, perform the training/validation loop, record the best
checkpoint based on validation accuracy and save a file mapping
class indices to their human readable names.

Swin Transformers operate on image patches and apply windowed self
attention across shifted windows to capture both local and global
context.  Their hierarchical design allows them to operate on
arbitrary image sizes while maintaining linear computational
complexity【38878464741457†L120-L129】.  In the context of this
script we fine‑tune the ``swinv2_base_window12to24_192to384.ms_in22k_ft_in1k``
checkpoint, which was pre‑trained on ImageNet‑22k and fine‑tuned on
ImageNet‑1k by the original authors【739641659479825†L55-L59】.  You can
replace the model name with any other model supported by
``timm.create_model`` if you desire.

Example usage::

    python train_swinv2.py --data_dir /path/to/dataset --output_dir ./runs

To train on a GPU just ensure that CUDA is available; the script will
automatically select it.  See ``python train_swinv2.py --help`` for
additional configurable parameters.
"""

import argparse
import os
import copy
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
from timm.data import resolve_model_data_config, create_transform


def parse_args() -> argparse.Namespace:
    """Parse command line options.

    Returns
    -------
    argparse.Namespace
        Populated namespace with attributes corresponding to the command
        line options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fine‑tune a Swin Transformer V2 base model on a custom dataset. "
            "Your data directory should contain 'train' and 'val' folders with "
            "sub‑directories per class."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset root directory containing train/val/test splits.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help=(
            "Directory where checkpoints and logs will be saved. "
            "If it doesn't exist it will be created."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
        help=(
            "Name of the pre‑trained model to fine‑tune. "
            "Any model that can be loaded via timm.create_model is valid."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of images per batch during training/validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs to run.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Base learning rate for the optimizer.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes used to load data.",
    )
    parser.add_argument(
        "--freeze_layers",
        action="store_true",
        help=(
            "Freeze all layers except the classification head. "
            "Useful for quick adaptation with small datasets."
        ),
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Optional path to a checkpoint (.pth) to resume training from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device to train on (e.g. 'cuda', 'cpu'). If not specified, "
            "the script will pick CUDA if available."
        ),
    )
    return parser.parse_args()


def load_data(
    root_dir: str,
    transform_train,
    transform_val,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create training and validation data loaders.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root containing 'train' and 'val' subdirectories.
    transform_train : callable
        Transformation applied to training images.
    transform_val : callable
        Transformation applied to validation images.
    batch_size : int
        Batch size for both data loaders.
    num_workers : int
        Number of worker processes for data loading.

    Returns
    -------
    tuple[DataLoader, DataLoader, list[str]]
        Data loaders for training and validation splits and a list of class
        names (ordered by index).
    """
    train_path = os.path.join(root_dir, "train")
    val_path = os.path.join(root_dir, "val")

    if not os.path.isdir(train_path) or not os.path.isdir(val_path):
        raise FileNotFoundError(
            "Couldn't find 'train' and 'val' folders inside the provided data directory"
        )

    train_dataset = ImageFolder(train_path, transform=transform_train)
    val_dataset = ImageFolder(val_path, transform=transform_val)
    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, class_names


def build_model(
    model_name: str, num_classes: int, freeze_layers: bool
) -> torch.nn.Module:
    """Create and optionally freeze a Swin V2 model.

    Parameters
    ----------
    model_name : str
        Name of the model as understood by ``timm.create_model``.
    num_classes : int
        Number of output classes for the classification head.
    freeze_layers : bool
        If ``True`` all parameters except the classifier head will be frozen.

    Returns
    -------
    torch.nn.Module
        Instantiated model.
    """
    # Create model with pre‑trained weights and correct number of classes
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    if freeze_layers:
        for name, param in model.named_parameters():
            # Only parameters belonging to the classifier head remain trainable
            if "head" not in name and "fc" not in name:
                param.requires_grad = False
    return model


def save_class_mapping(class_names: list[str], output_dir: str) -> None:
    """Persist class index to label mapping for later inference.

    Parameters
    ----------
    class_names : list[str]
        Ordered list of class names as returned by ``ImageFolder.classes``.
    output_dir : str
        Directory into which the mapping file will be written.
    """
    mapping_path = os.path.join(output_dir, "class_labels.txt")
    with open(mapping_path, "w", encoding="utf-8") as f:
        for idx, name in enumerate(class_names):
            f.write(f"{idx},{name}\n")


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Perform one training epoch and return the average loss.

    Parameters
    ----------
    model : torch.nn.Module
        Model being trained.
    data_loader : DataLoader
        Loader for the training data.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model weights.
    device : torch.device
        Device on which computations are performed.

    Returns
    -------
    float
        Average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        num_samples += images.size(0)
    return running_loss / max(num_samples, 1)


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a validation set.

    Parameters
    ----------
    model : torch.nn.Module
        Model being evaluated.
    data_loader : DataLoader
        Loader for the validation data.
    criterion : torch.nn.Module
        Loss function.
    device : torch.device
        Device on which computations are performed.

    Returns
    -------
    tuple[float, float]
        A tuple containing (average loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
    avg_loss = running_loss / max(num_samples, 1)
    accuracy = correct / max(num_samples, 1)
    return avg_loss, accuracy


def main() -> None:
    """Entry point for script execution."""
    args = parse_args()
    # Determine device
    device_str = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load a temporary model to fetch its default data configuration.  We
    # instantiate without altering num_classes because we only need the
    # associated normalization and augmentation parameters.
    tmp_model = timm.create_model(args.model_name, pretrained=True)
    data_config = resolve_model_data_config(tmp_model)
    train_transform = create_transform(**data_config, is_training=True)
    val_transform = create_transform(**data_config, is_training=False)

    # Build data loaders
    train_loader, val_loader, class_names = load_data(
        args.data_dir, train_transform, val_transform, args.batch_size, args.num_workers
    )
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Instantiate the model and optionally resume from a checkpoint
    model = build_model(args.model_name, num_classes, args.freeze_layers)
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint)
        print(f"Resumed training from checkpoint {args.resume_checkpoint}")
    model.to(device)

    # Prepare loss function and optimizer.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_model_weights: Optional[dict] = None

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_acc:.4f}"
        )
        # Keep track of the best performing model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(
                best_model_weights,
                os.path.join(args.output_dir, "best_model.pth"),
            )
            print(f"Saved new best model with accuracy {best_acc:.4f}")
        scheduler.step()

    # Save the final model weights regardless of performance
    torch.save(model.state_dict(), os.path.join(args.output_dir, "last_model.pth"))
    if best_model_weights is not None:
        # Restore best weights before saving mapping
        model.load_state_dict(best_model_weights)
    save_class_mapping(class_names, args.output_dir)
    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
