"""
Training script for fine-tuning a ConvNeXt‑L classifier on a custom image
classification dataset.  This script expects the dataset to be organised into
``train`` and ``val`` subdirectories under a user specified root folder, with
each subdirectory containing one folder per class.  During training the
pre‑trained ConvNeXt‑Large network from TorchVision is fine‑tuned using
cross‑entropy loss with optional label smoothing.  Model weights yielding
the best validation accuracy are saved along with the class mapping.

Usage example:

.. code-block:: bash

    python train.py \
        --data_root /path/to/data \
        --output_dir /path/to/results \
        --epochs 40 --batch_size 16 --lr 1e-4

This will load images from ``/path/to/data/train`` and ``/path/to/data/val``,
train for up to 40 epochs, and write the best model checkpoint into
``/path/to/results``.

The script automatically detects the number of classes from the training
subdirectory names and replaces the final classification head of the
ConvNeXt‑Large network accordingly.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models
from torchvision.transforms import InterpolationMode, autoaugment


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        A namespace containing the parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine‑tune ConvNeXt‑L on a custom dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=(
            "Path to the root of the dataset.  This directory must contain "
            "'train' and 'val' subdirectories, each with one folder per class."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where training artefacts such as model weights will be stored.",
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
        default=16,
        help="Number of samples per training batch (default: 16).  Adjust based on GPU memory.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for the optimiser (default: 1e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 penalty) coefficient (default: 1e-4)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor applied to the cross‑entropy loss (default: 0.1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for the data loaders (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def build_transforms(
    img_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Construct the training and validation transform pipelines.

    Parameters
    ----------
    img_size : int, optional
        Size of the square crop fed to the network (default: 224).  ConvNeXt
        models are typically trained on 224×224 crops.

    Returns
    -------
    Tuple[transforms.Compose, transforms.Compose]
        The training and validation transform pipelines.
    """
    # Normalisation values borrowed from ImageNet, which the pre‑trained
    # ConvNeXt weights were trained on.  Using these ensures that the
    # network sees similarly distributed inputs during fine‑tuning.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose(
        [
            # Randomly crop and resize the image.  We use bicubic interpolation
            # and a reasonably wide scale range to provide diversity in zoom
            # levels.  AutoAugment is employed to add further stochastic
            # augmentations designed for ImageNet‑like datasets.
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_tfms = transforms.Compose(
        [
            # For validation (and later testing) we use a deterministic pipeline
            # with a slightly larger resize followed by a centre crop.  Using
            # bicubic interpolation yields smoother results that match the
            # pre‑training recipe.
            transforms.Resize(
                int(img_size * 256 / 224),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, val_tfms


def create_model(num_classes: int) -> nn.Module:
    """Initialise the ConvNeXt‑Large model with a new classification head.

    Parameters
    ----------
    num_classes : int
        Number of target classes in the dataset.

    Returns
    -------
    nn.Module
        A ConvNeXt‑Large model ready for training on the specified number of
        classes.  The model is returned with pre‑trained ImageNet weights
        loaded and the final linear layer replaced to match ``num_classes``.
    """
    # Load the pre‑trained model.  We explicitly set ``weights`` to the
    # ImageNet1K weights.  Should this raise an exception due to missing
    # weights (unlikely in the provided environment), the model will fall
    # back to random initialisation.
    try:
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
    except AttributeError:
        # For torchvision versions prior to 0.18, the weights enum may live
        # directly under convnext_large.  We catch this gracefully.
        weights = None
    model = models.convnext_large(weights=weights)

    # Retrieve the number of features feeding into the classification head.  In
    # ConvNeXt models the classifier comprises several layers; the final
    # linear layer lives at index 2 of the classifier sequential container.
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        in_features = model.classifier[-1].in_features  # last layer
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        # Defensive programming: if the model structure changes in future
        # releases, fall back to replacing the entire classifier with a simple
        # linear head.
        in_features = model.get_classifier().in_features  # type: ignore[attr-defined]
        model.fc = nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]
    return model


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute the classification accuracy for a batch of predictions.

    Parameters
    ----------
    output : torch.Tensor
        The model outputs (logits) of shape (batch_size, num_classes).
    target : torch.Tensor
        The ground truth labels of shape (batch_size,).

    Returns
    -------
    float
        The proportion of correctly predicted samples in the batch.
    """
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = preds.eq(target).sum().item()
        total = target.size(0)
        return correct / total


def save_class_mapping(mapping: Dict[str, int], output_dir: str) -> None:
    """Persist the class_to_idx mapping to disk as a JSON file.

    The mapping is used later when loading the model for inference.  JSON is
    chosen because it is human readable and language agnostic.

    Parameters
    ----------
    mapping : Dict[str, int]
        Mapping from class name (folder name) to integer index.
    output_dir : str
        Directory where the JSON file will be written.  The file is named
        'class_to_idx.json'.
    """
    os.makedirs(output_dir, exist_ok=True)
    mapping_path = os.path.join(output_dir, "class_to_idx.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    """Run a single training epoch over the provided dataloader.

    Parameters
    ----------
    model : nn.Module
        The neural network being trained.
    dataloader : DataLoader
        The data loader providing batches of training samples.
    criterion : nn.Module
        The loss function.
    optimizer : optim.Optimizer
        Optimiser used to update model parameters.
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler for mixed precision training.
    device : torch.device
        The device (CPU or CUDA) on which tensors should reside.

    Returns
    -------
    Tuple[float, float]
        The mean training loss and accuracy across all batches for the epoch.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        running_total += batch_size

    mean_loss = running_loss / running_total
    mean_acc = running_correct / running_total
    return mean_loss, mean_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on the validation dataset.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    dataloader : DataLoader
        Data loader for the validation set.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Device on which computations should be performed.

    Returns
    -------
    Tuple[float, float]
        The mean loss and accuracy across the validation set.
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            running_total += batch_size

    mean_loss = running_loss / running_total
    mean_acc = running_correct / running_total
    return mean_loss, mean_acc


def main() -> None:
    """Entry point for training the ConvNeXt‑Large model.

    This function orchestrates the loading of data, initialisation of the
    network, training loop, validation, and checkpoint saving.
    """
    args = parse_args()

    # Set random seeds for reproducibility across PyTorch, NumPy and Python.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Prepare output directory.  We include a timestamp to avoid
    # inadvertently overwriting previous experiments when the same directory
    # name is reused.
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"train_log_{timestamp}.txt")

    # Build data augmentation pipelines.
    train_tfms, val_tfms = build_transforms(img_size=224)

    # Define dataset directories.
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Expected subdirectories 'train' and 'val' inside data_root; got {args.data_root}"
        )

    # Create datasets and data loaders.
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_tfms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_tfms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Save the class mapping so we can interpret predictions later.
    save_class_mapping(train_dataset.class_to_idx, args.output_dir)

    # Create the model and move it to the appropriate device.
    num_classes = len(train_dataset.classes)
    model = create_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function with label smoothing.  Label smoothing can
    # improve model calibration and robustness.
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Set up the optimiser and learning rate scheduler.  AdamW is a modern
    # adaptive optimiser that often works well on fine‑tuning tasks.  A
    # cosine annealing schedule helps lower the learning rate smoothly over
    # the course of training.
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Automatic mixed precision.  This can accelerate training and reduce
    # memory usage on NVIDIA GPUs without degrading model quality.
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    # Open a log file to record progress.  Logging both to console and file
    # helps with later analysis.
    with open(log_file, "w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            # Train for one epoch
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device
            )
            # Evaluate on the validation set
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            # Step the scheduler
            scheduler.step()

            # Write progress information
            msg = (
                f"Epoch [{epoch + 1}/{args.epochs}]"
                f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
                f"  val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                f"  lr={optimizer.param_groups[0]['lr']:.6f}"
            )
            print(msg)
            log.write(msg + "\n")
            log.flush()

            # Save the model if it improved
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_to_idx": train_dataset.class_to_idx,
                    },
                    best_model_path,
                )

        # Final summary
        summary = f"Training complete. Best validation accuracy: {best_acc:.4f}"
        print(summary)
        log.write(summary + "\n")


if __name__ == "__main__":
    main()
