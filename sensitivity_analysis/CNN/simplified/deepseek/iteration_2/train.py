import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Data transforms with augmentation
    train_transform = transforms.Compose(
        [
            transforms.Resize(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )

    # Create datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"), transform=val_transform
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
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model
    model = create_model(
        "convnext_large",
        pretrained=True,
        num_classes=len(train_dataset.classes),
        drop_rate=args.dropout_rate,
    )
    model = model.to(device)

    # Loss function
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Gradient scaler for mixed precision
    scaler = GradScaler()

    # Learning rate scheduler
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        warmup_t=args.warmup_epochs,
        warmup_lr_init=args.warmup_lr,
        lr_min=args.min_lr,
    )

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        all_preds = []
        all_targets = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_targets, all_preds)
        train_f1 = f1_score(all_targets, all_preds, average="weighted")

        # Validate
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(epoch + 1)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}"
        )
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_to_idx": train_dataset.class_to_idx,
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    val_loss = val_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds, average="weighted")
    return val_loss, val_acc, val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ConvNeXt-L Ophthalmology Classifier Training"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for models"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="Minimum learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing"
    )
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument(
        "--warmup_lr", type=float, default=1e-6, help="Warmup learning rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
