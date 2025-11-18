"""
train_swinv2_classifier.py
===========================

This script fine‑tunes a SwinV2 Base ("Swin‑V2‑B") vision transformer on an image
classification dataset.  Datasets should follow the common `ImageFolder`
layout where each split (``train``, ``val`` and optionally ``test``) contains
sub‑directories named after the class labels with images inside.  At a high
level the Swin Transformer builds hierarchical feature maps by merging
small image patches into progressively coarser windows and applies
self‑attention only within these local windows, resulting in linear
complexity relative to the input size【555662335336767†L67-L74】.  SwinV2
introduces several improvements over the original Swin Transformer,
including a residual‑post‑norm architecture with cosine attention,
log‑spaced continuous position bias for transferring models across
resolutions, and SimMIM self‑supervised pre‑training for improved
stability【555662335336767†L67-L80】.  These architectural features allow
SwinV2 models to scale well and perform competitively on a wide range of
vision tasks.

To adapt a pre‑trained SwinV2 model to your classification task the
script replaces the final classification head with a new head sized for
the number of classes in your dataset and fine‑tunes the weights on your
training data.  The underlying implementation uses the `timm` library
(`PyTorch Image Models`), which provides a large collection of pre‑trained
models.  Recent versions of `timm` add patch and position embedding
resizing support so Swin models can be instantiated for different input
resolutions without losing the benefit of pre‑trained weights【608117212520077†L166-L184】.  You
can therefore pass a custom `--img_size` to match your fundus image
resolution if desired.

Usage example:

```
python train_swinv2_classifier.py \
    --data_dir /path/to/dataset \
    --model_name swinv2_base_patch4_window16_256 \
    --img_size 256 \
    --batch_size 16 \
    --epochs 30 \
    --lr 3e-5 \
    --weight_decay 0.05 \
    --output_dir /path/to/output
```

The arguments are explained below.  The script will write the best model
checkpoint (based on validation accuracy), a JSON file containing the
class‑to‑index mapping and a training log (CSV) into the specified
``output_dir``.

Dependencies
------------

This script depends on PyTorch, torchvision, timm and tqdm.  You can
install them via pip:

```
pip install torch torchvision timm tqdm scikit‑learn
```

Note that training vision transformers on high resolution images can be
memory intensive; enabling mixed precision training via ``--use_amp``
reduces memory consumption and speeds up training on modern GPUs.

"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine‑tune a SwinV2 Base classifier on an image dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root directory containing 'train' and 'val' folders.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_patch4_window12_192_22k",
        help=(
            "Name of the pre‑trained model from timm to fine‑tune. "
            "If the specified model is unavailable in your timm installation, the script will attempt "
            "to fall back to another Swin or SwinV2 Base model."
        ),
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help=(
            "Input image size (height/width) for the model. If not specified, "
            "the default input size associated with the chosen model is used."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini‑batch size for training and validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate for the AdamW optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay (L2 penalty) for the optimizer.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where checkpoints and logs will be saved.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of subprocesses for data loading.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable mixed precision training to reduce memory usage.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help=(
            "If set, training will stop early if validation accuracy does not "
            "improve for this number of epochs."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create transformation pipelines for training and validation.

    Parameters
    ----------
    img_size : int
        Target size (height and width) for input images.

    Returns
    -------
    tuple
        A pair of torchvision transforms: (train_transform, val_transform).
    """
    # Mean and std values from ImageNet (RGB) for normalisation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose(
        [
            transforms.Resize(img_size + 32),  # enlarge then crop
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, val_transform


def prepare_dataloaders(
    data_dir: str,
    train_tf: transforms.Compose,
    val_tf: transforms.Compose,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Prepare PyTorch DataLoaders for training and validation.

    Parameters
    ----------
    data_dir : str
        Root directory containing ``train`` and ``val`` subdirectories.
    train_tf : callable
        Transformations applied to training images.
    val_tf : callable
        Transformations applied to validation images.
    batch_size : int
        Batch size for DataLoader.
    num_workers : int
        Number of worker processes for data loading.

    Returns
    -------
    tuple
        (train_loader, val_loader, class_names)
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_tf)

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


def create_model(model_name: str, num_classes: int, img_size: int = None) -> nn.Module:
    """Instantiate a SwinV2 model from timm and adapt its classifier.

    Parameters
    ----------
    model_name : str
        Name of the timm model to load.
    num_classes : int
        Number of output classes for classification head.
    img_size : int, optional
        Desired input image size.  If provided, the model will resize its
        positional embeddings to fit this size【608117212520077†L166-L184】.

    Returns
    -------
    torch.nn.Module
        A pre‑trained SwinV2 model with a new classification head.
    """
    # Create the model with pretrained weights.  Passing img_size triggers
    # timm to adapt the patch and position embedding to the requested size【608117212520077†L166-L184】.
    model = timm.create_model(
        model_name,
        pretrained=True,
        img_size=img_size,
        num_classes=num_classes,
    )
    return model


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model on a validation or test set.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    loader : DataLoader
        DataLoader providing images and labels.
    device : torch.device
        Device on which to run evaluation.

    Returns
    -------
    tuple
        (avg_loss, accuracy) where avg_loss is the mean cross‑entropy loss and
        accuracy is the fraction of correct predictions.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
) -> float:
    """Train the model for one epoch and return the average training loss.

    Parameters
    ----------
    model : torch.nn.Module
        Model being trained.
    loader : DataLoader
        DataLoader for the training set.
    optimizer : torch.optim.Optimizer
        Optimizer for gradient descent.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    device : torch.device
        Device on which training is executed.
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler used for mixed precision training.
    use_amp : bool
        Whether mixed precision is enabled.

    Returns
    -------
    float
        Average training loss over the epoch.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        if scheduler is not None:
            scheduler.step()
        pbar.set_postfix(loss=loss.item())
    avg_loss = running_loss / total
    return avg_loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First verify the requested model exists and perform fallback if necessary
    available_models = timm.list_models()
    if args.model_name not in available_models:
        fallback_name = None
        # Progressive fallback through SwinV2 base, SwinV2, Swin v1 base, any Swin, then ResNet50
        candidates = [m for m in available_models if "swinv2_base" in m]
        if candidates:
            fallback_name = candidates[0]
        else:
            candidates = [m for m in available_models if "swinv2" in m]
            if candidates:
                fallback_name = candidates[0]
            else:
                candidates = [m for m in available_models if m.startswith("swin_base")]
                if candidates:
                    fallback_name = candidates[0]
                else:
                    candidates = [m for m in available_models if "swin" in m]
                    if candidates:
                        fallback_name = candidates[0]
        if fallback_name is None:
            # Use a very common CNN as last resort
            if "resnet50" in available_models:
                fallback_name = "resnet50"
            else:
                raise RuntimeError(
                    f"The requested model '{args.model_name}' is not available and no suitable fallback was found. "
                    "Please update timm or specify an available model name."
                )
        print(
            f"Warning: model '{args.model_name}' not found in timm. "
            f"Falling back to '{fallback_name}'."
        )
        args.model_name = fallback_name

    # If img_size not specified, use model default.  Fallback verification has already occurred.
    if args.img_size is None:
        temp_model = timm.create_model(args.model_name, pretrained=False)
        input_size = temp_model.default_cfg.get("input_size", (3, 224, 224))
        args.img_size = input_size[1]
        del temp_model

    # Build transforms and dataloaders
    train_tf, val_tf = build_transforms(args.img_size)
    train_loader, val_loader, class_names = prepare_dataloaders(
        args.data_dir, train_tf, val_tf, args.batch_size, args.num_workers
    )

    num_classes = len(class_names)

    # Save class index mapping
    class_index_path = os.path.join(args.output_dir, "class_indices.json")
    with open(class_index_path, "w", encoding="utf-8") as f:
        json.dump({i: cls for i, cls in enumerate(class_names)}, f, indent=2)

    # Create and move model to device
    model = create_model(args.model_name, num_classes, args.img_size)
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = None
    if args.epochs > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader) * args.epochs
        )
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_acc = 0.0
    epochs_no_improve = 0
    log_path = os.path.join(args.output_dir, "training_log.csv")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("epoch,train_loss,val_loss,val_accuracy\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            scaler,
            args.use_amp,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc:.4f}")

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_acc:.4f}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            best_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "img_size": args.img_size,
                    "class_indices_path": class_index_path,
                },
                best_path,
            )
            print(f"Saved new best model to {best_path}")
        else:
            epochs_no_improve += 1
        if (
            args.early_stop_patience is not None
            and epochs_no_improve >= args.early_stop_patience
        ):
            print(
                f"Validation accuracy has not improved for {args.early_stop_patience} epochs; stopping early."
            )
            break

    print(f"Training completed. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
