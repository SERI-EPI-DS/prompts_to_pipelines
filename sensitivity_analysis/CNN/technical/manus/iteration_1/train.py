#!/usr/bin/env python3
"""
Fine-tuning script for ConvNext-L classifier on ophthalmology fundus images.
Implements state-of-the-art training techniques for optimal performance.
"""

import argparse
import os
import time
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchvision.transforms as transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from PIL import Image
import matplotlib.pyplot as plt


class FundusDataset(Dataset):
    """Custom dataset for fundus images with proper preprocessing."""

    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        # Build class mapping
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx

            # Collect all images in this class
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tiff",
                    ".bmp",
                ]:
                    self.samples.append((str(img_path), idx))

        print(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        print(f"Classes: {self.classes}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(
    input_size: int = 224, is_training: bool = True
) -> transforms.Compose:
    """Get image transforms for training or validation."""

    if is_training:
        # Training transforms with strong augmentation
        return transforms.Compose(
            [
                transforms.Resize((input_size + 32, input_size + 32)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            ]
        )
    else:
        # Validation transforms
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)

        # Create smoothed targets
        targets = torch.zeros_like(log_preds).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.smoothing) * targets + self.smoothing / n_classes

        loss = (-targets * log_preds).sum(dim=-1).mean()
        return loss


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.early_stop


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create ConvNext-L model with custom classifier head."""

    if pretrained:
        model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
    else:
        model = convnext_large(weights=None)

    # Replace classifier head - ConvNext expects specific structure
    # The original classifier is: AdaptiveAvgPool2d -> LayerNorm -> Flatten -> Linear
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # Global average pooling
        nn.Flatten(1),  # Flatten to [batch_size, features]
        nn.LayerNorm(in_features),  # LayerNorm after flattening
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )

    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch."""

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 50 == 0:
            print(
                f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%"
            )

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    val_acc: float,
    checkpoint_path: str,
    class_to_idx: Dict[str, int],
) -> None:
    """Save model checkpoint."""

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc": val_acc,
            "class_to_idx": class_to_idx,
        },
        checkpoint_path,
    )


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str,
) -> None:
    """Plot and save training history."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot losses
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ConvNext-L for ophthalmology classification"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing train/val folders",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save results and model weights",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of epochs (default: 50)",
    )
    parser.add_argument(
        "--input_size", type=int, default=224, help="Input image size (default: 224)"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (default: 0.1)",
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(
            f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Create datasets
    train_transform = get_transforms(args.input_size, is_training=True)
    val_transform = get_transforms(args.input_size, is_training=False)

    train_dataset = FundusDataset(
        os.path.join(args.data_root, "train"), transform=train_transform
    )
    val_dataset = FundusDataset(
        os.path.join(args.data_root, "val"), transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    num_classes = len(train_dataset.classes)
    model = create_model(num_classes, pretrained=True)
    model = model.to(device)

    print(f"Model created with {num_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Loss function and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    # Use different learning rates for backbone and classifier
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if "classifier" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": args.learning_rate * 0.1,
            },  # Lower LR for backbone
            {
                "params": classifier_params,
                "lr": args.learning_rate,
            },  # Higher LR for classifier
        ],
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    print(f"\nStarting training for {args.max_epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.max_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch + 1}/{args.max_epochs} ({epoch_time:.1f}s)")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.2e}")
        print("-" * 60)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_path = results_dir / "best_model.pth"
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                val_acc,
                str(best_checkpoint_path),
                train_dataset.class_to_idx,
            )
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

        # Save latest checkpoint
        latest_checkpoint_path = results_dir / "latest_model.pth"
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            val_acc,
            str(latest_checkpoint_path),
            train_dataset.class_to_idx,
        )

        # Early stopping check
        if early_stopping(val_acc):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save training history plot
    plot_path = results_dir / "training_history.png"
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs, str(plot_path)
    )

    # Save training configuration and results
    config = {
        "args": vars(args),
        "num_classes": num_classes,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "best_val_acc": best_val_acc,
        "total_epochs": len(train_losses),
        "total_training_time": total_time,
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
    }

    config_path = results_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training configuration saved to {config_path}")
    print(f"Best model saved to {best_checkpoint_path}")
    print(f"Training history plot saved to {plot_path}")


if __name__ == "__main__":
    main()
