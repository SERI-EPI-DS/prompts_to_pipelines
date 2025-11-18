#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def main():
    parser = argparse.ArgumentParser(
        description="Train a Swin-V2-B classifier for ophthalmology image classification."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the root data directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save model weights and results.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Define data transforms
    train_transform = create_transform(
        input_size=256,
        is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )

    # Create datasets and dataloaders
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"), transform=val_transform
    )

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

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_dataset.classes}")

    # Create the model
    model = timm.create_model(
        "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        pretrained=True,
        num_classes=num_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.05
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    print(f"Starting training for {args.num_epochs} epochs...")

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_acc = correct / total

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "num_classes": num_classes,
                    "class_names": train_dataset.classes,
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )
            print(f"  New best model saved! Val Acc: {best_val_acc:.4f}")

        print("-" * 50)

    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
