#!/usr/bin/env python3
"""
Advanced ConvNext-L Training Script for Ophthalmology Fundus Image Classification

This script implements state-of-the-art techniques for medical image classification:
- ConvNext-L architecture with pre-trained weights
- Advanced data augmentation strategies
- Mixed precision training
- Learning rate scheduling with warmup
- Class balancing and weighted loss
- Comprehensive evaluation metrics
- Model checkpointing and early stopping
- TensorBoard logging

Author: AI Research Assistant
Date: 2025
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import torchvision
from torchvision import transforms, datasets
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.scheduler import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class AdvancedDataAugmentation:
    """
    Advanced data augmentation pipeline for medical images
    Based on state-of-the-art techniques from recent research
    """

    def __init__(self, image_size=224, severity="medium"):
        self.image_size = image_size
        self.severity = severity

    def get_training_transforms(self):
        """Get training transforms with advanced augmentation"""
        if self.severity == "light":
            transforms_list = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        elif self.severity == "medium":
            transforms_list = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        else:  # heavy
            transforms_list = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=20),
                transforms.RandomAffine(
                    degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)
                ),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
                transforms.RandomApply([transforms.RandomPosterize(bits=4)], p=0.2),
                transforms.RandomApply(
                    [transforms.RandomSolarize(threshold=128)], p=0.2
                ),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]

        return transforms.Compose(transforms_list)

    def get_validation_transforms(self):
        """Get validation transforms (no augmentation)"""
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class ConvNextClassifier(nn.Module):
    """
    ConvNext-L classifier with custom head for medical image classification
    """

    def __init__(self, num_classes, pretrained=True, dropout_rate=0.3):
        super(ConvNextClassifier, self).__init__()

        # Load ConvNext-Large model
        self.backbone = timm.create_model(
            "convnext_large_in22k",
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="avg",
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Custom classification head with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(self.feature_dim // 2, num_classes),
        )

        # Initialize classifier weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize classifier weights using Xavier initialization"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    """

    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class MetricsCalculator:
    """
    Comprehensive metrics calculator for medical image classification
    """

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate comprehensive classification metrics"""
        metrics = {}

        # Basic metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from sklearn.metrics import cohen_kappa_score

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )
        kappa = cohen_kappa_score(y_true, y_pred)

        metrics.update(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "kappa": kappa,
            }
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(y_true, y_pred, average=None)
        )

        for i, class_name in enumerate(self.class_names):
            metrics[f"{class_name}_precision"] = precision_per_class[i]
            metrics[f"{class_name}_recall"] = recall_per_class[i]
            metrics[f"{class_name}_f1"] = f1_per_class[i]

        # AUC if probabilities are provided
        if y_prob is not None and self.num_classes > 2:
            try:
                auc = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
                metrics["auc"] = auc
            except ValueError:
                logger.warning("Could not calculate AUC score")

        return metrics


class EarlyStopping:
    """Early stopping implementation"""

    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.best_weights = model.state_dict().copy()

        return False


def load_data(data_dir, batch_size=32, num_workers=4, augmentation_severity="medium"):
    """
    Load and prepare datasets with advanced augmentation
    """
    logger.info(f"Loading data from {data_dir}")

    # Initialize augmentation
    augmentation = AdvancedDataAugmentation(severity=augmentation_severity)

    # Define transforms
    train_transform = augmentation.get_training_transforms()
    val_transform = augmentation.get_validation_transforms()

    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"), transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"), transform=val_transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"), transform=val_transform
    )

    # Calculate class weights for balanced training
    class_counts = np.bincount(train_dataset.targets)
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_dataset.targets), y=train_dataset.targets
    )

    # Create weighted sampler for balanced training
    sample_weights = [class_weights[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}"
    )
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    logger.info(f"Class names: {train_dataset.classes}")

    return train_loader, val_loader, test_loader, train_dataset.classes, class_weights


def train_epoch(
    model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        if scheduler is not None:
            scheduler.step_update(epoch * len(train_loader) + batch_idx)

        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Log progress
        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%"
            )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device, num_classes, class_names):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            with autocast():
                output = model(data)
                loss = criterion(output, target)

            running_loss += loss.item()

            # Get predictions and probabilities
            probabilities = F.softmax(output, dim=1)
            _, predicted = output.max(1)

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate metrics
    metrics_calc = MetricsCalculator(num_classes, class_names)
    metrics = metrics_calc.calculate_metrics(
        all_targets, all_predictions, np.array(all_probabilities)
    )

    epoch_loss = running_loss / len(val_loader)

    return epoch_loss, metrics, all_targets, all_predictions


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)

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
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="ConvNext-L Training for Ophthalmology"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--augmentation",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Augmentation severity",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="focal",
        choices=["ce", "focal", "label_smooth"],
        help="Loss function type",
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, class_names, class_weights = load_data(
        args.data_dir, args.batch_size, args.num_workers, args.augmentation
    )

    num_classes = len(class_names)

    # Create model
    model = ConvNextClassifier(num_classes, pretrained=True, dropout_rate=args.dropout)
    model = model.to(device)

    # Define loss function
    if args.loss_type == "focal":
        criterion = FocalLoss(alpha=1, gamma=2)
    elif args.loss_type == "label_smooth":
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Define scheduler with warmup
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=args.lr * 0.01,
        warmup_t=5,
        warmup_lr_init=args.lr * 0.1,
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))

    # Training loop
    best_val_acc = 0.0
    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        logger.info(f"Resumed from epoch {start_epoch}")

    logger.info("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )

        # Validation
        val_loss, val_metrics, val_targets, val_predictions = validate_epoch(
            model, val_loader, criterion, device, num_classes, class_names
        )

        val_acc = val_metrics["accuracy"] * 100

        # Update scheduler
        scheduler.step(epoch)

        # Log metrics
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f"Metrics/{metric_name}", metric_value, epoch)

        logger.info(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_names": class_names,
                    "num_classes": num_classes,
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )

            # Save confusion matrix for best model
            save_confusion_matrix(
                val_targets,
                val_predictions,
                class_names,
                os.path.join(args.output_dir, "confusion_matrix_best.png"),
            )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "class_names": class_names,
                "num_classes": num_classes,
            },
            os.path.join(args.output_dir, "checkpoint.pth"),
        )

        # Early stopping
        if early_stopping(val_acc, model):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_loss, test_metrics, test_targets, test_predictions = validate_epoch(
        model, test_loader, criterion, device, num_classes, class_names
    )

    # Save final results
    results = {
        "test_metrics": test_metrics,
        "class_names": class_names,
        "training_args": vars(args),
    }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save final confusion matrix
    save_confusion_matrix(
        test_targets,
        test_predictions,
        class_names,
        os.path.join(args.output_dir, "confusion_matrix_test.png"),
    )

    # Print final results
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f'Test accuracy: {test_metrics["accuracy"]*100:.2f}%')
    logger.info(f'Test F1-score: {test_metrics["f1_score"]:.4f}')
    logger.info(f'Test Kappa: {test_metrics["kappa"]:.4f}')

    writer.close()


if __name__ == "__main__":
    main()
