#!/usr/bin/env python3
"""
ConvNext-L Fine-tuning Script for Ophthalmology Fundus Image Classification

This script implements state-of-the-art fine-tuning techniques for medical image classification
using ConvNext-L architecture with PyTorch.

Author: Manus AI
Date: 2025-06-19
"""

import argparse
import os
import sys
import time
import logging
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torchvision
from torchvision import transforms, datasets
import torchvision.models as models

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


class FundusDataset(Dataset):
    """Custom dataset class for fundus photographs."""

    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to the data directory
            transform: Optional transform to be applied on images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        self._load_samples()

    def _load_samples(self) -> None:
        """Load all image samples and create class mappings."""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist")

        # Get all class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        class_dirs.sort()

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.data_dir}")

        # Create class mappings
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Load all samples
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            # Supported image extensions
            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in extensions:
                    self.samples.append((str(img_path), class_idx))

        if not self.samples:
            raise ValueError(f"No valid images found in {self.data_dir}")

        print(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        for i, cls_name in enumerate(self.classes):
            count = sum(1 for _, label in self.samples if label == i)
            print(f"  {cls_name}: {count} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset."""
        img_path, label = self.samples[idx]

        try:
            # Load image
            from PIL import Image

            image = Image.open(img_path).convert("RGB")

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                black_image = Image.new("RGB", (224, 224), (0, 0, 0))
                image = self.transform(black_image)
            else:
                image = torch.zeros(3, 224, 224)
            return image, label


def get_transforms(
    input_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation transforms.

    Args:
        input_size: Input image size

    Returns:
        Tuple of (train_transform, val_transform)
    """
    # ImageNet statistics for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training transforms with augmentation
    train_transform = transforms.Compose(
        [
            transforms.Resize((int(input_size * 1.15), int(input_size * 1.15))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # Suitable for fundus images
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Validation transforms without augmentation
    val_transform = transforms.Compose(
        [
            transforms.Resize((int(input_size * 1.15), int(input_size * 1.15))),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, val_transform


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create ConvNext-L model.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        ConvNext-L model
    """
    # Load ConvNext-L model
    if pretrained:
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        model = models.convnext_large(weights=weights)
    else:
        model = models.convnext_large(weights=None)

    # Replace classifier
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model


def create_optimizer_and_scheduler(
    model: nn.Module,
    lr_backbone: float,
    lr_classifier: float,
    epochs: int,
    steps_per_epoch: int,
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: The model to optimize
        lr_backbone: Learning rate for backbone
        lr_classifier: Learning rate for classifier
        epochs: Total number of epochs
        steps_per_epoch: Number of steps per epoch

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Separate parameters for different learning rates
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if "classifier" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    # Create optimizer with different learning rates
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": classifier_params, "lr": lr_classifier},
        ],
        weight_decay=0.01,
    )

    # Create cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-6
    )

    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: The model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        device: Device to train on
        epoch: Current epoch number
        logger: Logger instance

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        scheduler.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        accuracy = 100.0 * correct / total
        pbar.set_postfix(
            {
                "Loss": f"{avg_loss:.4f}",
                "Acc": f"{accuracy:.2f}%",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}",
            }
        )

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    logger.info(f"Epoch {epoch+1} Train - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    class_names: List[str],
) -> Tuple[float, float, np.ndarray, List[int], List[int]]:
    """
    Validate for one epoch.

    Args:
        model: The model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        logger: Logger instance
        class_names: List of class names

    Returns:
        Tuple of (average_loss, accuracy, confusion_matrix, all_targets, all_predictions)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Store for metrics calculation
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{accuracy:.2f}%"})

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)

    logger.info(f"Epoch {epoch+1} Val - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

    return avg_loss, accuracy, cm, all_targets, all_predictions


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    accuracy: float,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> None:
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }

    # Save last checkpoint
    torch.save(checkpoint, checkpoint_dir / "last_epoch.pth")

    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, checkpoint_dir / "best_model.pth")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Path,
) -> None:
    """Plot and save training curves."""
    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "b-", label="Training Loss")
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(epochs, train_accs, "b-", label="Training Accuracy")
    ax2.plot(epochs, val_accs, "r-", label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], save_path: Path
) -> None:
    """Plot and save confusion matrix."""
    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="ConvNext-L Fine-tuning for Ophthalmology"
    )

    # Data arguments
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to data root directory"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to results directory"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of epochs (default: 50)"
    )
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=1e-4,
        help="Learning rate for backbone (default: 1e-4)",
    )
    parser.add_argument(
        "--lr_classifier",
        type=float,
        default=1e-3,
        help="Learning rate for classifier (default: 1e-3)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )

    # Parse arguments
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Setup paths
    data_root = Path(args.data_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create all necessary subdirectories
    (results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(results_dir / "logs")
    logger.info(f"Starting training with arguments: {args}")

    # Check data directories
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists():
        raise ValueError(f"Training directory {train_dir} does not exist")
    if not val_dir.exists():
        raise ValueError(f"Validation directory {val_dir} does not exist")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(
            f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = FundusDataset(train_dir, transform=train_transform)
    val_dataset = FundusDataset(val_dir, transform=val_transform)

    # Verify same classes
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Training and validation datasets have different classes")

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # Create model
    logger.info("Creating model...")
    model = create_model(num_classes, pretrained=True)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, args.lr_backbone, args.lr_classifier, args.epochs, len(train_loader)
    )

    # Create loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Create gradient scaler for mixed precision
    scaler = GradScaler()

    # Initialize tracking variables
    start_epoch = 0
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("accuracy", 0.0)
        logger.info(
            f"Resumed from epoch {start_epoch}, best accuracy: {best_val_acc:.2f}%"
        )

    # Training loop
    logger.info("Starting training...")
    start_time = time.time()

    # Linear probing phase (first 5 epochs)
    linear_probing_epochs = min(5, args.epochs)

    for epoch in range(start_epoch, args.epochs):
        # Linear probing: freeze backbone for first few epochs
        if epoch < linear_probing_epochs:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            logger.info(f"Epoch {epoch+1}: Linear probing mode (backbone frozen)")
        else:
            # Full fine-tuning: unfreeze all parameters
            for param in model.parameters():
                param.requires_grad = True
            if epoch == linear_probing_epochs:
                logger.info(f"Epoch {epoch+1}: Switching to full fine-tuning mode")

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            logger,
        )

        # Validate
        val_loss, val_acc, cm, val_targets, val_preds = validate_epoch(
            model, val_loader, criterion, device, epoch, logger, class_names
        )

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Check if best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")

        # Save checkpoint
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            val_loss,
            val_acc,
            results_dir / "checkpoints",
            is_best,
        )

        # Save metrics to CSV
        metrics_df = pd.DataFrame(
            {
                "epoch": range(1, len(train_losses) + 1),
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_acc": train_accs,
                "val_acc": val_accs,
            }
        )
        metrics_df.to_csv(results_dir / "logs" / "metrics.csv", index=False)

        # Plot training curves
        plot_training_curves(
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            results_dir / "plots" / "training_curves.png",
        )

        # Plot confusion matrix for best model
        if is_best:
            plot_confusion_matrix(
                cm, class_names, results_dir / "plots" / "confusion_matrix.png"
            )

            # Save classification report
            report = classification_report(
                val_targets, val_preds, target_names=class_names, output_dict=True
            )
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(results_dir / "logs" / "classification_report.csv")

        # Early stopping check (optional)
        if epoch >= 10:  # Start checking after 10 epochs
            recent_val_losses = val_losses[-5:]  # Last 5 epochs
            if len(recent_val_losses) == 5 and all(
                recent_val_losses[i] >= recent_val_losses[i - 1] for i in range(1, 5)
            ):
                logger.info("Early stopping triggered - validation loss not improving")
                break

    # Training completed
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/3600:.2f} hours")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save final summary
    summary = {
        "total_epochs": epoch + 1,
        "best_val_accuracy": best_val_acc,
        "total_training_time_hours": total_time / 3600,
        "num_classes": num_classes,
        "class_names": class_names,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

    import json

    with open(results_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training summary saved to training_summary.json")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
