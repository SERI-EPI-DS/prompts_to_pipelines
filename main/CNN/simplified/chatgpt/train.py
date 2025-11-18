"""
train_convnext.py
===================

This script fine‑tunes a ConvNeXt‑L model on a custom image classification
dataset. The dataset is expected to follow the typical folder structure used
by the PyTorch ``ImageFolder`` class: each split (e.g. ``train`` and
``val``) resides inside its own directory and, within each split, every
class has its own subfolder containing the images.  The ``ImageFolder``
utility will automatically infer class names from the subdirectory names
【51527509087205†L524-L536】, which greatly simplifies dataset management.

To achieve state‑of‑the‑art performance while remaining simple to use, the
script leverages a pre‑trained ConvNeXt‑L backbone from the PyTorch
``torchvision`` library. The ``ConvNeXt`` family was proposed by Meta
Research in their paper *“A ConvNet for the 2020s”*.  The authors showed
that these pure convolutional models, built from standard ConvNet
components, compete favorably with Vision Transformers in terms of
accuracy and scalability, achieving up to 87.8 % top‑1 ImageNet
accuracy【750288342696214†L23-L28】.  The official weights package
provides an inference transform that rescales images, center crops to
224×224 pixels and normalizes pixel values using ImageNet’s mean
and standard deviation【297744549850868†L160-L167】.  These normalization
statistics are widely used in computer vision and appear in many
recommendations, including Microsoft’s ONNX tutorial, which suggests
normalizing RGB channels with mean ``[0.485, 0.456, 0.406]`` and
standard deviation ``[0.229, 0.224, 0.225]``【668952771882936†L710-L721】.

The script exposes several command‑line arguments to control the
training process, such as the dataset location, number of epochs,
batch size, learning rate and output directory.  It logs losses and
accuracies, saves the best model (based on validation accuracy) and
optionally resumes training from a previous checkpoint.  A typical
training run might look like this:

.. code-block:: bash

   python train_convnext.py \
       --data-dir /path/to/dataset \
       --output-dir ./results \
       --epochs 20 \
       --batch-size 16 \
       --learning-rate 1e-4

Make sure to install the necessary dependencies before running the
script (for example ``pip install torch torchvision timm tqdm scikit‑learn``).
"""

import argparse
import os
import time
from datetime import datetime
from typing import Tuple, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    input_size: int = 224,
    use_randaugment: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create training and validation dataloaders.

    Args:
        data_dir: Root directory containing ``train`` and ``val`` subfolders.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        input_size: Side length of the input crop.  224 is standard for
            ConvNeXt models【297744549850868†L160-L167】.
        use_randaugment: Whether to apply RandAugment for additional
            augmentation.

    Returns:
        A tuple of (train_loader, val_loader, class_names).
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory '{train_dir}' does not exist")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory '{val_dir}' does not exist")

    # Normalization statistics taken from ImageNet【297744549850868†L160-L167】【668952771882936†L710-L721】
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = [
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
    ]
    # Optionally add RandAugment for stronger augmentations
    if use_randaugment:
        try:
            train_transforms.append(transforms.RandAugment())
        except AttributeError:
            # RandAugment may not be available in older torchvision versions
            pass
    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    val_transforms = transforms.Compose(
        [
            transforms.Resize(
                int(input_size * 1.12)
            ),  # e.g. 224 -> 250; typical ratio used by ImageNet
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=train_dir, transform=transforms.Compose(train_transforms)
    )
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)
    class_names = train_dataset.classes

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

    return train_loader, val_loader, class_names


def initialize_model(num_classes: int, input_size: int = 224) -> nn.Module:
    """Load a pre‑trained ConvNeXt‑L model and adapt it to the given number of classes.

    The model uses pre‑trained weights trained on ImageNet‑1K.  The final
    classification layer is replaced by a new ``Linear`` layer with
    ``num_classes`` outputs.  The returned model is ready for fine‑tuning.

    Args:
        num_classes: Number of output classes.
        input_size: Input size (224 or 384). Only affects the model's internal
            positional encodings for some architectures; ConvNeXt is fully
            convolutional so any reasonable input size works.

    Returns:
        A ``torch.nn.Module`` representing the fine‑tunable model.
    """
    # Load pretrained ConvNeXt‑L.  The weights default to ImageNet‑1K
    # (IMAGENET1K_V1) which follow the improved training recipe.  The
    # corresponding inference transform resizes, crops and normalizes
    # images using mean=[0.485,0.456,0.406] and std=[0.229,0.224,0.225]
    # 【297744549850868†L160-L167】.
    weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
    model = models.convnext_large(weights=weights)
    # Replace the final classification head
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        # The classifier is a Sequential: LayerNorm -> Linear
        last_layer = model.classifier[-1]
        if isinstance(last_layer, nn.Linear):
            in_features = last_layer.in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unexpected classifier structure in ConvNeXt model")
    else:
        raise RuntimeError("ConvNeXt model does not have a classifier attribute")
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch.

    Returns:
        A tuple (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation set.

    Returns:
        A tuple (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def save_checkpoint(
    state: Dict[str, object],
    is_best: bool,
    output_dir: str,
    epoch: int,
) -> None:
    """Save model and training state to disk.

    Args:
        state: Dictionary containing model and optimizer state.
        is_best: If True, also save a copy as 'best_model.pth'.
        output_dir: Directory to write checkpoint files into.
        epoch: Current epoch number.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(state, filename)
    if is_best:
        best_path = os.path.join(output_dir, "best_model.pth")
        torch.save(state["model_state_dict"], best_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine‑tune a ConvNeXt‑L classifier")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing 'train' and 'val' subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save logs and models",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--input-size", type=int, default=224, help="Input image size (224 or 384)"
    )
    parser.add_argument(
        "--no-augment", action="store_true", help="Disable RandAugment during training"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataloaders and get class names
    train_loader, val_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        use_randaugment=not args.no_augment,
    )

    num_classes = len(class_names)
    model = initialize_model(num_classes=num_classes, input_size=args.input_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_acc = 0.0

    if args.resume is not None and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint.get("scheduler_state_dict", {}))
        start_epoch = checkpoint.get("epoch", 1) + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        print(
            f"[INFO] Resumed from checkpoint '{args.resume}' at epoch {start_epoch - 1}"
        )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Open log file
    log_file_path = os.path.join(args.output_dir, "training_log.csv")
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"[INFO] Epoch {epoch}/{args.epochs}")
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc,
                "class_names": class_names,
            },
            is_best=is_best,
            output_dir=args.output_dir,
            epoch=epoch,
        )

        # Append to log
        with open(log_file_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_acc:.4f},{val_loss:.6f},{val_acc:.4f}\n"
            )

        epoch_duration = time.time() - epoch_start
        print(f"[INFO] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"[INFO] Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
        print(f"[INFO] Epoch took {epoch_duration/60:.2f} minutes\n")

    print(f"[INFO] Training completed. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
