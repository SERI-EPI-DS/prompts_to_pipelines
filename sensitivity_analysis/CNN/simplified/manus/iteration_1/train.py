#!/usr/bin/env python3
"""
ConvNext-L Fine-tuning Script for Ophthalmology Image Classification
Author: AI Assistant for Ophthalmology Research
Description: State-of-the-art training script for fine-tuning ConvNext-L on fundus photographs
"""

import os
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class FundusDataset:
    """Custom dataset class for fundus photographs with advanced augmentations"""

    def __init__(self, data_dir, split="train", img_size=384):
        self.data_dir = os.path.join(data_dir, split)
        self.img_size = img_size
        self.split = split

        # State-of-the-art augmentations for medical images
        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.RandomAffine(
                        degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def get_dataset(self):
        return datasets.ImageFolder(self.data_dir, transform=self.transform)


class ConvNextClassifier(nn.Module):
    """ConvNext-L classifier with custom head for ophthalmology diagnosis"""

    def __init__(self, num_classes, pretrained=True, dropout_rate=0.3):
        super(ConvNextClassifier, self).__init__()

        # Load ConvNext-L with ImageNet pretrained weights
        self.backbone = timm.create_model(
            "convnext_large", pretrained=pretrained, num_classes=0
        )

        # Get the feature dimension
        self.feature_dim = self.backbone.num_features

        # Custom classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
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


class CosineAnnealingWarmRestarts:
    """Custom cosine annealing with warm restarts scheduler"""

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = (
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + np.cos(np.pi * self.T_cur / self.T_i))
                / 2
            )


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler:
            scheduler.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        pbar.set_postfix(
            {
                "Loss": f"{running_loss/(batch_idx+1):.4f}",
                "Acc": f"{100.*correct/total:.2f}%",
            }
        )

    return running_loss / len(dataloader), 100.0 * correct / total


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            pbar.set_postfix(
                {
                    "Loss": f"{running_loss/(batch_idx+1):.4f}",
                    "Acc": f"{100.*correct/total:.2f}%",
                }
            )

    return (
        running_loss / len(dataloader),
        100.0 * correct / total,
        all_preds,
        all_targets,
    )


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(train_losses, label="Training Loss", color="blue")
    ax1.plot(val_losses, label="Validation Loss", color="red")
    ax1.set_title("Model Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(train_accs, label="Training Accuracy", color="blue")
    ax2.plot(val_accs, label="Validation Accuracy", color="red")
    ax2.set_title("Model Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


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
        description="ConvNext-L Fine-tuning for Ophthalmology"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory (should contain train/val/test folders)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results and model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=384, help="Input image size")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save training configuration
    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets...")
    train_dataset_obj = FundusDataset(args.data_dir, "train", args.img_size)
    val_dataset_obj = FundusDataset(args.data_dir, "val", args.img_size)

    train_dataset = train_dataset_obj.get_dataset()
    val_dataset = val_dataset_obj.get_dataset()

    # Get class names and number of classes
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

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

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize model
    print("Initializing ConvNext-L model...")
    model = ConvNextClassifier(num_classes, pretrained=True, dropout_rate=args.dropout)
    model = model.to(device)

    # Loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("Starting training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )

        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )

        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Save final model
    model_path = os.path.join(args.output_dir, "convnext_fundus_classifier.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": num_classes,
            "config": config,
            "training_history": {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
            },
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    # Generate final validation report
    val_loss, val_acc, val_preds, val_targets = validate_epoch(
        model, val_loader, criterion, device
    )

    # Classification report
    report = classification_report(
        val_targets, val_preds, target_names=class_names, output_dict=True
    )
    report_path = os.path.join(args.output_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Best Model Saved: {model_path}")

    # Save plots
    plot_path = os.path.join(args.output_dir, "training_history.png")
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    save_confusion_matrix(val_targets, val_preds, class_names, cm_path)

    print(f"Training plots saved to {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
