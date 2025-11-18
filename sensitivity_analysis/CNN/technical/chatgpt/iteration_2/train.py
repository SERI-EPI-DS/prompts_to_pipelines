#!/usr/bin/env python3
"""
Fine‑tune a ConvNeXt‑L model for image classification.

This script trains a ConvNeXt Large (ConvNeXt‑L) model on a custom image
classification dataset using PyTorch and TorchVision.  The expected dataset
layout matches that produced by ``torchvision.datasets.ImageFolder``.  A root
directory contains ``train``, ``val`` and ``test`` sub‑directories, each of
which contains a directory per class with that class’s images.  The training
script will read the training and validation splits, apply a suite of
augmentations, fine‑tune the model, and write the best checkpoint to disk.

Key features:

* Uses TorchVision's ConvNeXt‑L model with ImageNet‑1K pretrained weights.
* Replaces the final classification layer to match the number of dataset
  classes.
* Implements label smoothing using PyTorch’s built‑in ``CrossEntropyLoss``
  (the ``label_smoothing`` argument softens the ground truth distribution
  to reduce overconfidence【701448864621765†L2661-L2666】【479432953449643†L419-L435】).
* Optionally applies mixup and cutmix augmentations, which have been shown to
  improve accuracy on modern convnets【479432953449643†L437-L454】.
* Tracks validation accuracy and saves the model weights that obtain the
  highest accuracy.

The default hyper‑parameters are chosen to work on a single RTX 3090 with
24 GB of VRAM but may be overridden on the command line.

Example usage:

```
python train.py --data_dir /path/to/data --results_dir ./results \
    --epochs 50 --batch_size 8 --learning_rate 3e-4
```

"""

import argparse
import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import datasets, models


def set_seed(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply the mixup augmentation to a batch of images and labels.

    Mixup forms a convex combination of two examples drawn uniformly at random
    from the batch.  Each output image is ``lam * x_i + (1 - lam) * x_j`` and
    the targets are similarly mixed.  See Zhang et al., "mixup: Beyond
    Empirical Risk Minimization" for details.

    Args:
        x: Batch of input images, shape (N, C, H, W).
        y: Tensor of integer class labels, shape (N,).
        alpha: Hyper‑parameter controlling the strength of mixing.  A larger
            value makes the distribution of lambda values more uniform.

    Returns:
        mixed_x: Tensor containing the mixed images.
        y_a: Original labels for the first term.
        y_b: Original labels for the second term.
        lam: Lambda value sampled from Beta(alpha, alpha).
    """
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply the cutmix augmentation to a batch of images and labels.

    Cutmix composites a rectangle from a second image onto the first image.
    The area of the rectangle is determined by a Beta distribution and the
    labels are combined proportionally to the area.  See Yun et al., "CutMix:
    Regularization Strategy to Train Strong Classifiers with Localizable
    Features" for details.

    Args:
        x: Batch of input images, shape (N, C, H, W).
        y: Tensor of integer class labels, shape (N,).
        alpha: Hyper‑parameter controlling the strength of mixing.

    Returns:
        cutmix_x: Tensor containing the cutmix images.
        y_a: Labels for the base images.
        y_b: Labels for the patches.
        lam: Effective lambda (proportion of the base image kept).
    """
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)
    y_a, y_b = y, y[index]
    # Compute bounding box for the patch
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    # Uniformly sample the centre of the patch
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    # Compute box boundaries
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    # Create patched images
    x_cutmix = x.clone()
    x_cutmix[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    # Adjust lambda to exactly match the area ratio
    lam_adjusted = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    return x_cutmix, y_a, y_b, lam_adjusted


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: data.DataLoader,
    device: torch.device,
    epoch: int,
    epochs: int,
    use_mixup: bool,
    mixup_alpha: float,
    use_cutmix: bool,
    cutmix_alpha: float,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
) -> Tuple[float, float]:
    """Train the model for a single epoch.

    Returns the average loss and accuracy over the epoch.
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    num_samples = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if use_mixup and use_cutmix:
            # Randomly select between mixup and cutmix with equal probability
            if random.random() < 0.5:
                inputs_mix, targets_a, targets_b, lam = mixup_data(
                    inputs, labels, mixup_alpha
                )
                outputs = model(inputs_mix)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                    outputs, targets_b
                )
                # For accuracy, pick the dominant label
                preds = outputs.argmax(dim=1)
                # Use targets_a when lam >= 0.5, else targets_b
                blended_targets = torch.where(lam >= 0.5, targets_a, targets_b)
                running_corrects += (preds == blended_targets).sum().item()
            else:
                inputs_cm, targets_a, targets_b, lam = cutmix_data(
                    inputs, labels, cutmix_alpha
                )
                outputs = model(inputs_cm)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                    outputs, targets_b
                )
                preds = outputs.argmax(dim=1)
                blended_targets = torch.where(lam >= 0.5, targets_a, targets_b)
                running_corrects += (preds == blended_targets).sum().item()
        elif use_mixup:
            inputs_mix, targets_a, targets_b, lam = mixup_data(
                inputs, labels, mixup_alpha
            )
            outputs = model(inputs_mix)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                outputs, targets_b
            )
            preds = outputs.argmax(dim=1)
            blended_targets = torch.where(lam >= 0.5, targets_a, targets_b)
            running_corrects += (preds == blended_targets).sum().item()
        elif use_cutmix:
            inputs_cm, targets_a, targets_b, lam = cutmix_data(
                inputs, labels, cutmix_alpha
            )
            outputs = model(inputs_cm)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                outputs, targets_b
            )
            preds = outputs.argmax(dim=1)
            blended_targets = torch.where(lam >= 0.5, targets_a, targets_b)
            running_corrects += (preds == blended_targets).sum().item()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item() * inputs.size(0)
        num_samples += inputs.size(0)
    epoch_loss = running_loss / num_samples
    epoch_acc = running_corrects / num_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation or test set.

    Returns the average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    num_samples = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        num_samples += inputs.size(0)
    epoch_loss = running_loss / num_samples
    epoch_acc = running_corrects / num_samples
    return epoch_loss, epoch_acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ConvNeXt‑L model on a classification dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing train, val, and test folders.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (max 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=2e-5,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for CrossEntropyLoss",
    )
    parser.add_argument(
        "--mixup", action="store_true", help="Enable mixup data augmentation"
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.2,
        help="Alpha parameter for the Beta distribution in mixup",
    )
    parser.add_argument(
        "--cutmix", action="store_true", help="Enable cutmix data augmentation"
    )
    parser.add_argument(
        "--cutmix_alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for the Beta distribution in cutmix",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    assert args.epochs <= 50, "Maximum number of epochs must not exceed 50"

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations
    # Training augmentation: random resized crop, horizontal flip, rotation, color jitter
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Validation transformation: resize then center crop.  TorchVision's pretrained
    # ConvNeXt models use resize to 232 and center crop to 224 with ImageNet
    # normalization【436985033393033†L160-L167】.
    val_transforms = transforms.Compose(
        [
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Datasets
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # DataLoaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")

    # Initialize model with pretrained weights
    # According to the TorchVision docs, convnext_large accepts a weights
    # parameter and uses ImageNet‑1K pretrained weights when weights=DEFAULT【436985033393033†L92-L122】.
    model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
    # Replace the classification head to match our number of classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.to(device)

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer and scheduler
    # We use AdamW with weight decay; the new training recipe recommends tuning
    # weight decay and excluding normalization parameters from weight decay【479432953449643†L459-L473】.
    # Here we follow that advice by setting a small weight decay and excluding
    # parameters whose names contain 'norm' or 'bias' from weight decay.
    def get_optimizer_params(model: nn.Module, weight_decay: float):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "norm"]):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    optimizer = optim.AdamW(
        get_optimizer_params(model, args.weight_decay), lr=args.learning_rate
    )

    # Cosine annealing with warm restarts or step scheduler could be used; here we
    # adopt a cosine schedule which smoothly decays the learning rate to zero
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_path = os.path.join(args.results_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            device,
            epoch,
            args.epochs,
            use_mixup=args.mixup,
            mixup_alpha=args.mixup_alpha,
            use_cutmix=args.cutmix,
            cutmix_alpha=args.cutmix_alpha,
            scheduler=scheduler,
        )
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_names": train_dataset.classes,
                    "val_accuracy": val_acc,
                },
                checkpoint_path,
            )
    print(
        f"Training complete. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}."
    )


if __name__ == "__main__":
    main()
