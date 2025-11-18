"""
Fine-tuning script for Swin-V2-B on a custom dataset.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
from timm.data import create_transform
from timm.loss import LabelSmoothingCrossEntropy
from torch.cuda.amp import GradScaler, autocast


def main():
    parser = argparse.ArgumentParser(description="Swin-V2-B Fine-tuning")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the root data directory"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to the results directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # --- Data Preparation ---
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    # Get number of classes
    num_classes = len(os.listdir(train_dir))

    # Create transforms
    train_transform = create_transform(
        input_size=256,
        is_training=True,
        auto_augment="rand-m9-mstd0.5",  # RandAugment
        re_prob=0.25,  # Random Erasing
        re_mode="pixel",
    )

    val_transform = create_transform(input_size=256, is_training=False)

    # Create datasets
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- Model Preparation ---
    model = timm.create_model(
        "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        pretrained=True,
        num_classes=num_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Training Setup ---
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # --- Training Loop ---
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")

        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(args.results_dir, "best_model.pth")
            )

        scheduler.step()


if __name__ == "__main__":
    main()
