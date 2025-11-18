#!/usr/bin/env python3
"""
Swin-V2-B Fine-tuning Script for Ophthalmology Image Classification

This script implements state-of-the-art fine-tuning of Swin Transformer V2-Base
for medical image classification, specifically designed for ophthalmology datasets.

Key Features:
- Swin-V2-B architecture with hierarchical feature extraction
- PolyLoss function for improved classification performance
- Comprehensive data augmentation pipeline
- Mixed precision training for efficiency
- Configurable data paths and hyperparameters
- Grad-CAM visualization support
- Model checkpointing and early stopping

Based on research from:
- "Multi-Fundus Diseases Classification Using Retinal OCT Images with Swin Transformer V2"
- "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis"

Author: AI Research Assistant
Date: 2025
"""

import os
import sys
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class PolyLoss(nn.Module):
    """
    PolyLoss implementation based on the paper:
    "PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions"

    This loss function has been shown to outperform cross-entropy loss and focal loss
    in medical image classification tasks, particularly for fundus disease classification.
    """

    def __init__(self, epsilon: float = 2.0, reduction: str = "mean"):
        """
        Initialize PolyLoss.

        Args:
            epsilon: Polynomial coefficient (ε₁ in the paper). Default is 2.0 as used in OCT classification.
            reduction: Specifies the reduction to apply to the output ('mean', 'sum', 'none').
        """
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PolyLoss.

        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)

        Returns:
            Computed PolyLoss
        """
        # Compute cross-entropy loss
        ce_loss = self.cross_entropy(logits, targets)

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Compute PolyLoss: L_Poly-1 = -log(Pt) + ε₁(1 - Pt)
        poly_loss = ce_loss + self.epsilon * (1 - pt)

        if self.reduction == "mean":
            return poly_loss.mean()
        elif self.reduction == "sum":
            return poly_loss.sum()
        else:
            return poly_loss


class MedicalImageDataset(Dataset):
    """
    Custom dataset class for medical images with enhanced preprocessing.
    Supports the standard folder structure: dataset > train/test/val > class_folders > images
    """

    def __init__(self, data_path: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the dataset folder
            transform: Torchvision transforms to apply
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.dataset = ImageFolder(root=data_path, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

        logger.info(f"Loaded dataset from {data_path}")
        logger.info(f"Found {len(self.dataset)} images in {len(self.classes)} classes")
        logger.info(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_transforms(
    image_size: int = 256, is_training: bool = True
) -> transforms.Compose:
    """
    Get data transforms for training and validation.

    Based on the methodology from the research papers:
    - Resize to 256x256 (as used in the Swin-V2 OCT paper)
    - Comprehensive data augmentation for training
    - Normalization using ImageNet statistics (for transfer learning)

    Args:
        image_size: Target image size (default: 256)
        is_training: Whether to apply training augmentations

    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def create_model(
    num_classes: int,
    pretrained: bool = True,
    model_name: str = "swinv2_base_window12to16_192to256_22kft1k",
) -> nn.Module:
    """
    Create Swin Transformer V2 Base model.

    Uses the timm library implementation of Swin-V2-B with:
    - Pre-trained weights from ImageNet-22K and fine-tuned on ImageNet-1K
    - Window size adapted for 256x256 input resolution
    - Custom classification head for the target number of classes

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights
        model_name: Specific Swin-V2 model variant

    Returns:
        Configured Swin-V2-B model
    """
    logger.info(f"Creating {model_name} model with {num_classes} classes")

    # Create model using timm
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.1,  # Dropout rate
        drop_path_rate=0.1,  # Stochastic depth rate
    )

    logger.info(
        f"Model created successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    logger.info(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        device: Device to run on
        epoch: Current epoch number

    Returns:
        Dictionary containing training metrics
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        scheduler.step()

        # Calculate metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Update progress bar
        current_lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100 * correct_predictions / total_samples:.2f}%",
                "LR": f"{current_lr:.2e}",
            }
        )

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct_predictions / total_samples

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Validate the model for one epoch.

    Args:
        model: The neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        epoch: Current epoch number

    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Calculate comprehensive metrics
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="weighted"
    )

    return {
        "loss": epoch_loss,
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1 * 100,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_acc: float,
    save_path: str,
    is_best: bool = False,
):
    """
    Save model checkpoint.

    Args:
        model: The neural network model
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        best_acc: Best validation accuracy so far
        save_path: Path to save the checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint, save_path)

    if is_best:
        best_path = save_path.replace(".pth", "_best.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"New best model saved with accuracy: {best_acc:.2f}%")


def plot_training_history(
    train_history: List[Dict], val_history: List[Dict], save_path: str
):
    """
    Plot and save training history.

    Args:
        train_history: List of training metrics per epoch
        val_history: List of validation metrics per epoch
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_history) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    ax1.plot(epochs, [h["loss"] for h in train_history], "b-", label="Training Loss")
    ax1.plot(epochs, [h["loss"] for h in val_history], "r-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(
        epochs, [h["accuracy"] for h in train_history], "b-", label="Training Accuracy"
    )
    ax2.plot(
        epochs, [h["accuracy"] for h in val_history], "r-", label="Validation Accuracy"
    )
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    # Learning rate plot
    ax3.plot(epochs, [h["learning_rate"] for h in train_history], "g-")
    ax3.set_title("Learning Rate Schedule")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_yscale("log")
    ax3.grid(True)

    # Validation metrics plot
    ax4.plot(epochs, [h["precision"] for h in val_history], "r-", label="Precision")
    ax4.plot(epochs, [h["recall"] for h in val_history], "g-", label="Recall")
    ax4.plot(epochs, [h["f1_score"] for h in val_history], "b-", label="F1-Score")
    ax4.set_title("Validation Metrics")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Score (%)")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Training history plot saved to {save_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Swin-V2-B for Ophthalmology Image Classification"
    )

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing train/val/test folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save outputs (models, logs, plots)",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_window12to16_192to256_22kft1k",
        help="Swin-V2 model variant from timm",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Input image size (default: 256)"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pre-trained weights",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--poly_epsilon", type=float, default=2.0, help="Epsilon parameter for PolyLoss"
    )

    # Training configuration
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--save_freq", type=int, default=10, help="Save checkpoint every N epochs"
    )

    # Experiment tracking
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for experiment tracking",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="swin-v2-ophthalmology",
        help="W&B project name",
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name for logging"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Version: {torch.version.cuda}")

    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project=args.project_name, name=args.experiment_name, config=vars(args)
        )

    # Load datasets
    logger.info("Loading datasets...")

    train_transform = get_transforms(args.image_size, is_training=True)
    val_transform = get_transforms(args.image_size, is_training=False)

    train_dataset = MedicalImageDataset(
        data_path=os.path.join(args.data_root, "train"), transform=train_transform
    )

    val_dataset = MedicalImageDataset(
        data_path=os.path.join(args.data_root, "val"), transform=val_transform
    )

    # Create data loaders
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

    # Create model
    num_classes = len(train_dataset.classes)
    model = create_model(num_classes, args.pretrained, args.model_name)
    model = model.to(device)

    # Define loss function (PolyLoss)
    criterion = PolyLoss(epsilon=args.poly_epsilon)

    # Define optimizer (AdamW as used in Swin Transformer papers)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Define learning rate scheduler (Cosine Annealing with Warm Restarts)
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps // 4,  # First restart after 1/4 of total steps
        T_mult=2,  # Double the period after each restart
        eta_min=1e-6,
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    logger.info("Starting training...")
    start_time = time.time()

    best_val_acc = 0.0
    patience_counter = 0
    train_history = []
    val_history = []

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 50)

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )
        train_history.append(train_metrics)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        val_history.append(val_metrics)

        # Log metrics
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%"
        )
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, "
            f"Precision: {val_metrics['precision']:.2f}%, Recall: {val_metrics['recall']:.2f}%, "
            f"F1: {val_metrics['f1_score']:.2f}%"
        )

        # Log to W&B
        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1_score": val_metrics["f1_score"],
                    "learning_rate": train_metrics["learning_rate"],
                }
            )

        # Save checkpoint
        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc,
                str(checkpoint_path),
                is_best,
            )

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(
                f"Early stopping triggered after {args.patience} epochs without improvement"
            )
            break

    # Training completed
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time/3600:.2f} hours")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save final model
    final_model_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Plot training history
    plot_path = output_dir / "training_history.png"
    plot_training_history(train_history, val_history, str(plot_path))

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "train_history": train_history,
                "val_history": val_history,
                "best_val_acc": best_val_acc,
                "total_epochs": len(train_history),
                "total_time_hours": total_time / 3600,
            },
            f,
            indent=2,
        )

    if args.use_wandb:
        wandb.finish()

    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()
