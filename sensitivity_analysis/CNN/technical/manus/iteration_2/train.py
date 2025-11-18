#!/usr/bin/env python3
"""
Fine-tuning script for ConvNext-L classifier on ophthalmology fundus images.
ULTRA-STABLE VERSION: Completely removes ColorJitter and optimizes memory usage.
"""

import argparse
import os
import time
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import timm


# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class FundusDataset(Dataset):
    """Custom dataset for fundus images with ultra-stable augmentations."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform=None,
        class_to_idx: Optional[Dict] = None,
    ):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []
        self.classes = []

        # Get class names from directories
        if class_to_idx is None:
            self.classes = sorted(
                [d.name for d in self.data_dir.iterdir() if d.is_dir()]
            )
            self.class_to_idx = {
                cls_name: idx for idx, cls_name in enumerate(self.classes)
            }
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())

        # Collect all image paths and labels
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        self.samples.append(
                            (str(img_path), self.class_to_idx[class_name])
                        )

        print(
            f"Found {len(self.samples)} images in {split} set across {len(self.classes)} classes"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image with memory optimization
        try:
            image = Image.open(img_path).convert("RGB")
            # Ensure image is not too large to prevent memory issues
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a small black image as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to {img_path}: {e}")
                # Apply ultra-minimal transform as fallback
                fallback_transform = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),  # Smaller size for memory
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                image = fallback_transform(image)

        return image, label


def get_transforms(
    input_size: int = 224, is_training: bool = True
):  # Reduced default size
    """Get data transforms with ultra-stable augmentations (NO ColorJitter)."""

    if is_training:
        # Ultra-stable training transforms - NO ColorJitter to prevent overflow
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (input_size + 16, input_size + 16)
                ),  # Smaller padding
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),  # Very conservative rotation
                # REMOVED ColorJitter completely to prevent overflow errors
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.02, 0.02),  # Very small translation
                    scale=(0.98, 1.02),  # Very small scale change
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                # Reduced RandomErasing
                transforms.RandomErasing(p=0.02, scale=(0.01, 0.05), ratio=(0.5, 2.0)),
            ]
        )
    else:
        # Validation/test transforms (no augmentation)
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


class ConvNextClassifier(nn.Module):
    """ConvNext-L classifier optimized for memory usage."""

    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super(ConvNextClassifier, self).__init__()

        # Load ConvNext-L model with proper feature extraction
        self.backbone = timm.create_model(
            "convnext_large", pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Get feature dimension from the backbone
        feature_dim = self.backbone.num_features
        print(f"ConvNext backbone feature dimension: {feature_dim}")

        # Simplified classification head to reduce memory usage
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),  # Direct mapping to reduce parameters
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)
        return self.classifier(features)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_probs = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with aggressive memory management."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        try:
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )

            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Clear intermediate variables to save memory
            del output, loss

            # Periodic memory cleanup
            if batch_idx % 20 == 0:
                clear_memory()
                if batch_idx % 50 == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                        f"Loss: {running_loss/(batch_idx+1):.4f}, Acc: {100.*correct/total:.2f}%"
                    )

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    f"CUDA out of memory in batch {batch_idx}. Clearing cache and skipping batch."
                )
                clear_memory()
                continue
            else:
                print(f"Runtime error in batch {batch_idx}: {e}")
                continue
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model with memory optimization."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            try:
                data, target = data.to(device, non_blocking=True), target.to(
                    device, non_blocking=True
                )
                output = model(data)
                loss = criterion(output, target)

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Clear variables
                del output, loss

                # Periodic memory cleanup
                if batch_idx % 10 == 0:
                    clear_memory()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"CUDA out of memory in validation batch {batch_idx}. Clearing cache and skipping."
                    )
                    clear_memory()
                    continue
                else:
                    print(f"Runtime error in validation batch {batch_idx}: {e}")
                    continue
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue

    val_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    val_acc = 100.0 * correct / total if total > 0 else 0.0

    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-Stable ConvNext-L for fundus image classification"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root directory of the dataset"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size (reduced for memory)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate (reduced)"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input image size (reduced for memory)",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers (0 for stability)",
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Clear memory at start
    clear_memory()

    # Data transforms
    train_transform = get_transforms(args.input_size, is_training=True)
    val_transform = get_transforms(args.input_size, is_training=False)

    print("Ultra-stable augmentation pipeline:")
    print("- NO ColorJitter (completely removed to prevent overflow)")
    print("- Conservative geometric transforms only")
    print("- Aggressive memory management")
    print("- Reduced input size and batch size")

    # Datasets
    train_dataset = FundusDataset(args.data_root, "train", transform=train_transform)
    val_dataset = FundusDataset(
        args.data_root,
        "val",
        transform=val_transform,
        class_to_idx=train_dataset.class_to_idx,
    )

    # Save class mapping
    class_mapping = {
        "class_to_idx": train_dataset.class_to_idx,
        "idx_to_class": {v: k for k, v in train_dataset.class_to_idx.items()},
    }
    with open(os.path.join(args.results_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=2)

    # Data loaders with memory optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,  # Disabled to save memory
        drop_last=True,  # Drop incomplete batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    # Model
    num_classes = len(train_dataset.classes)
    model = ConvNextClassifier(num_classes=num_classes, dropout=args.dropout)
    model = model.to(device)

    print(f"Model created with {num_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Test model with a dummy input
    try:
        dummy_input = torch.randn(1, 3, args.input_size, args.input_size).to(device)
        with torch.no_grad():
            dummy_output = model(dummy_input)
            print(f"Model output shape: {dummy_output.shape}")
        del dummy_input, dummy_output
        clear_memory()
    except Exception as e:
        print(f"Warning: Model shape verification failed: {e}")

    # Loss function and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # Training loop
    best_val_acc = 0.0
    train_history = []

    print(f"Starting ultra-stable training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        try:
            # Clear memory before each epoch
            clear_memory()

            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch + 1
            )

            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            # Update learning rate
            scheduler.step()

            epoch_time = time.time() - epoch_start

            print(f"Epoch {epoch + 1}/{args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
            print(f"  Time: {epoch_time:.2f}s")
            print("-" * 50)

            # Save training history
            train_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
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
                        "best_val_acc": best_val_acc,
                        "class_to_idx": train_dataset.class_to_idx,
                        "args": vars(args),
                    },
                    os.path.join(args.results_dir, "best_model.pth"),
                )
                print(
                    f"New best model saved with validation accuracy: {best_val_acc:.2f}%"
                )

            # Early stopping check
            if early_stopping(val_loss):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        except Exception as e:
            print(f"Error in epoch {epoch + 1}: {e}")
            clear_memory()
            print("Continuing to next epoch...")
            continue

    # Save final model
    try:
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "final_val_acc": val_acc if "val_acc" in locals() else 0.0,
                "class_to_idx": train_dataset.class_to_idx,
                "args": vars(args),
            },
            os.path.join(args.results_dir, "final_model.pth"),
        )
    except Exception as e:
        print(f"Error saving final model: {e}")

    # Save training history
    try:
        with open(os.path.join(args.results_dir, "training_history.json"), "w") as f:
            json.dump(train_history, f, indent=2)
    except Exception as e:
        print(f"Error saving training history: {e}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
