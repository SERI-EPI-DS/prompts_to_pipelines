#!/usr/bin/env python3
"""
train.py
===========

This script fine‑tunes a ConvNeXt‑L model on a dataset of color fundus
photographs.  It expects the data directory to contain ``train`` and
``val`` subfolders, each of which contains one subfolder per class with
the corresponding images.  The script trains the network for up to
a user specified number of epochs, evaluates on the validation set
after each epoch and saves the weights that achieve the highest
validation accuracy.  A JSON file containing the list of class names is
saved alongside the best model so that the test script can recover the
original class ordering.

Usage example::

    python train.py \
        --data_root /path/to/dataset \
        --results_dir /path/to/save/results \
        --epochs 50 \
        --batch_size 8 \
        --lr 2e-5

The script uses CUDA if available and falls back to CPU otherwise.  If
CUDA is available on multiple devices, the ``--gpu`` flag can be
provided to select which GPU index to use (default is 0).
"""

import argparse
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights


def set_random_seeds(seed: int = 42) -> None:
    """Helper to set seeds for reproducibility.

    Args:
        seed: Seed value to set across random, numpy and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For performance reasons we do not set deterministic mode; ConvNeXt uses
    # batchnorm and other operations that are safe for non‑deterministic runs.
    # However, we enable benchmark to allow cuDNN to find efficient kernels.
    torch.backends.cudnn.benchmark = True


def get_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    """Prepare training and validation dataloaders.

    Args:
        data_root: Root directory containing ``train`` and ``val`` folders.
        batch_size: Number of images per batch.
        num_workers: Number of worker processes for data loading.

    Returns:
        A dictionary mapping ``'train'`` and ``'val'`` to their respective
        data loaders.
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Expected 'train' and 'val' directories inside {data_root}, "
            f"but found: train exists={os.path.isdir(train_dir)}, val exists={os.path.isdir(val_dir)}"
        )

    # Standard ImageNet normalization values.  ConvNeXt models are trained
    # on images resized to 224×224 with these statistics.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Data augmentations for training.  RandomResizedCrop with a scale
    # between 0.8 and 1.0 encourages robustness to slight zoom and
    # cropping.  ColorJitter introduces brightness/contrast variations,
    # which are reasonable for fundus photographs.  Random horizontal
    # flips are also included to account for left/right eye differences.
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # For validation we avoid random augmentations and simply resize the
    # shorter side to 256 and perform a center crop to 224.  This
    # procedure matches the standard evaluation pipeline used by
    # ConvNeXt on ImageNet.
    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=train_transforms),
        "val": datasets.ImageFolder(val_dir, transform=val_transforms),
    }

    dataloaders = {
        phase: DataLoader(
            image_datasets[phase],
            batch_size=batch_size,
            shuffle=(phase == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        for phase in ["train", "val"]
    }

    return dataloaders, image_datasets


def build_model(num_classes: int) -> nn.Module:
    """Load a ConvNeXt‑L model pre‑trained on ImageNet and adapt the classifier.

    Args:
        num_classes: Number of target classes for the dataset.

    Returns:
        A ``nn.Module`` ready for training on the specified number of classes.
    """
    # Load pretrained ConvNeXt‑L weights from ImageNet.  Using
    # ``weights=ConvNeXt_Large_Weights.DEFAULT`` automatically downloads
    # weights on the first run and caches them for subsequent runs.
    weights = ConvNeXt_Large_Weights.DEFAULT
    model = convnext_large(weights=weights)

    # Replace the final classifier to match the dataset classes.  In
    # torchvision 0.18, ConvNeXt models expose ``classifier`` as either
    # a Sequential(LayerNorm, Linear) or directly a Linear layer.  We
    # inspect the classifier and replace the final linear layer.
    if isinstance(model.classifier, nn.Sequential):
        # The last element should be a Linear layer.
        last_layer = model.classifier[-1]
        if not isinstance(last_layer, nn.Linear):
            raise RuntimeError(
                "Unexpected classifier structure; cannot replace final layer"
            )
        in_features = last_layer.in_features
        new_layers = list(model.classifier.children())
        new_layers[-1] = nn.Linear(in_features, num_classes)
        model.classifier = nn.Sequential(*new_layers)
    elif isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError("ConvNeXt classifier has an unknown type; cannot adapt it")

    return model


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    dataset_sizes: Dict[str, int],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int,
    results_dir: str,
) -> nn.Module:
    """Train the model and save the best performing weights.

    Args:
        model: The neural network to train.
        dataloaders: Dict of DataLoader objects for training and validation.
        dataset_sizes: Dict mapping phases to number of samples.
        criterion: Loss function with label smoothing.
        optimizer: Optimizer used for weight updates.
        scheduler: Learning rate scheduler invoked after each epoch.
        device: Device on which to perform computations.
        num_epochs: Maximum number of epochs to train.
        results_dir: Directory in which to save the best weights.

    Returns:
        The model loaded with the best weights observed over the course of
        training.
    """
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                # Forward pass.  Track gradients only in the training phase.
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model if it yields a better validation accuracy
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                # Save the model weights whenever a new best is found
                weights_path = os.path.join(results_dir, "best_model.pth")
                torch.save(best_model_wts, weights_path)
                print(
                    f"New best model saved with accuracy {best_acc:.4f} at epoch {epoch + 1}"
                )

        print()

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine‑tune a ConvNeXt‑L classifier on a fundus dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the root of the dataset containing train/val/test folders",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory where training outputs (weights, class names) will be stored",
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
        default=8,
        help="Mini‑batch size for training and validation (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Initial learning rate for the AdamW optimizer (default: 2e‑5)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay (L2 penalty) for the optimizer (default: 0.05)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading (default: 4)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for CrossEntropyLoss (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use if multiple GPUs are available (default: 0)",
    )
    args = parser.parse_args()

    # Ensure the results directory exists
    os.makedirs(args.results_dir, exist_ok=True)

    # Set seeds for reproducibility
    set_random_seeds(args.seed)

    # Prepare data loaders
    dataloaders, image_datasets = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dataset_sizes = {phase: len(image_datasets[phase]) for phase in ["train", "val"]}
    class_names: List[str] = image_datasets["train"].classes

    # Save class names for later use in the test script
    class_file_path = os.path.join(args.results_dir, "class_names.json")
    with open(class_file_path, "w") as f:
        json.dump(class_names, f)

    # Build the model and move it to the appropriate device
    model = build_model(num_classes=len(class_names))
    device_str = (
        f"cuda:{args.gpu}"
        if torch.cuda.is_available() and torch.cuda.device_count() > args.gpu
        else "cpu"
    )
    device = torch.device(device_str)
    model = model.to(device)

    # Define loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Define optimizer and scheduler.  AdamW is well suited for fine‑tuning
    # transformer style architectures like ConvNeXt.  CosineAnnealingLR
    # gradually reduces the learning rate over the course of training.
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train the model
    trained_model = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        results_dir=args.results_dir,
    )

    # Save the final model (although the best model is already saved)
    final_weights_path = os.path.join(args.results_dir, "last_model.pth")
    torch.save(trained_model.state_dict(), final_weights_path)
    print(f"Training complete.  Final model saved to {final_weights_path}.")


if __name__ == "__main__":
    main()
