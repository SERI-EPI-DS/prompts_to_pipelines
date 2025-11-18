#!/usr/bin/env python3
"""
Swin-V2-B Fine-tuning Script for Ophthalmology Classification
Author: AI Assistant
Description: Fine-tunes a Swin-V2-B model on fundus photographs for medical diagnosis classification
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MedicalImageDataset(Dataset):
    """Custom dataset for medical images with enhanced preprocessing"""

    def __init__(self, root_dir: str, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx or {}
        self.classes = []

        self._load_samples()

    def _load_samples(self):
        """Load all image samples and create class mappings"""
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Dataset directory {self.root_dir} does not exist")

        # Get all class directories
        class_dirs = [
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]
        class_dirs.sort()

        if not self.class_to_idx:
            self.class_to_idx = {
                cls_name: idx for idx, cls_name in enumerate(class_dirs)
            }

        self.classes = list(self.class_to_idx.keys())

        # Load all samples
        for class_name in class_dirs:
            if class_name not in self.class_to_idx:
                continue

            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                ):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

        logger.info(
            f"Loaded {len(self.samples)} samples from {len(self.classes)} classes"
        )
        for cls_name, cls_idx in self.class_to_idx.items():
            count = sum(1 for _, idx in self.samples if idx == cls_idx)
            logger.info(f"  {cls_name}: {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a dummy image and label
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label


def get_transforms(
    input_size: int = 224, is_training: bool = True
) -> transforms.Compose:
    """Get data transforms for training or validation"""

    if is_training:
        # Training transforms with medical-appropriate augmentations
        transform = transforms.Compose(
            [
                transforms.Resize((input_size + 32, input_size + 32)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
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
        # Validation/test transforms
        transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return transform


class SwinV2Classifier(nn.Module):
    """Swin-V2-B model for medical image classification"""

    def __init__(
        self, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.2
    ):
        super(SwinV2Classifier, self).__init__()

        # Load pretrained Swin-V2-B
        if pretrained:
            self.backbone = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        else:
            self.backbone = swin_v2_b(weights=None)

        # Get the number of features from the classifier
        num_features = self.backbone.head.in_features

        # Replace the classifier head
        self.backbone.head = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(num_features, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)


class EarlyStopping:
    """Early stopping utility"""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.1,
) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            logger.info(
                f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, "
                f"Acc: {100.*correct/total:.2f}%"
            )

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def save_training_plots(
    train_losses: List[float],
    train_accs: List[float],
    val_losses: List[float],
    val_accs: List[float],
    save_path: str,
):
    """Save training history plots"""
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(train_losses, label="Training Loss", color="blue")
    ax1.plot(val_losses, label="Validation Loss", color="red")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(train_accs, label="Training Accuracy", color="blue")
    ax2.plot(val_accs, label="Validation Accuracy", color="red")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train Swin-V2-B for ophthalmology classification"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to the data root directory"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        required=True,
        help="Path to the project root directory",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory name within project root",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup directories
    data_root = Path(args.data_root)
    project_root = Path(args.project_root)
    results_dir = project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists():
        raise ValueError(f"Training directory {train_dir} does not exist")
    if not val_dir.exists():
        raise ValueError(f"Validation directory {val_dir} does not exist")

    # Get transforms
    train_transform = get_transforms(args.input_size, is_training=True)
    val_transform = get_transforms(args.input_size, is_training=False)

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = MedicalImageDataset(str(train_dir), transform=train_transform)
    val_dataset = MedicalImageDataset(
        str(val_dir), transform=val_transform, class_to_idx=train_dataset.class_to_idx
    )

    num_classes = len(train_dataset.classes)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {train_dataset.classes}")

    # Save class mapping
    class_mapping = {
        "class_to_idx": train_dataset.class_to_idx,
        "idx_to_class": {v: k for k, v in train_dataset.class_to_idx.items()},
        "classes": train_dataset.classes,
    }
    with open(results_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)

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
    logger.info("Creating model...")
    model = SwinV2Classifier(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # Setup loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Setup learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    # Setup early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)

    # Training history
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(args.max_epochs):
        epoch_start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.label_smoothing
        )

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Record history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start_time

        logger.info(
            f"Epoch {epoch+1}/{args.max_epochs} ({epoch_time:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
            f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "class_mapping": class_mapping,
                    "args": vars(args),
                },
                results_dir / "best_model.pth",
            )
            logger.info(
                f"New best model saved with validation accuracy: {val_acc:.2f}%"
            )

        # Check early stopping
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.1f} minutes")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save final model
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "class_mapping": class_mapping,
            "args": vars(args),
        },
        results_dir / "final_model.pth",
    )

    # Save training history
    history = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc,
        "total_epochs": epoch + 1,
        "total_time": total_time,
    }

    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save training plots
    save_training_plots(
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        str(results_dir / "training_history.png"),
    )

    logger.info(f"Training completed. Results saved to {results_dir}")


if __name__ == "__main__":
    main()
