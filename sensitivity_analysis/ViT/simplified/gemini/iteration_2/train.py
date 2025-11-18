# train.py (Corrected)

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm


def train_model(args):
    """Main function to train the model."""

    # --- 1. Setup and Configuration ---
    print("Starting model training setup...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Loading and Preprocessing ---
    # Create a base model to get its data configuration
    temp_model = timm.create_model(args.model_name, pretrained=True)
    data_config = resolve_data_config({}, model=temp_model)

    # Create transforms using timm's factory, which are optimized for the model
    train_transform = create_transform(**data_config, is_training=True)
    val_transform = create_transform(**data_config, is_training=False)

    print("Data Transforms:")
    print(f"  - Training: {train_transform}")
    print(f"  - Validation: {val_transform}")

    # Create datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"), transform=val_transform
    )

    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {', '.join(train_dataset.classes)}")

    # Save class mapping for later use in testing
    class_to_idx_path = os.path.join(args.output_dir, "class_to_idx.json")
    with open(class_to_idx_path, "w") as f:
        json.dump(train_dataset.class_to_idx, f)
    print(f"Saved class mapping to {class_to_idx_path}")

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

    # --- 3. Model Initialization ---
    print(f"Initializing model: {args.model_name}")
    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)
    model.to(device)

    # --- 4. Optimizer, Scheduler, and Loss Function ---
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    # Use a learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    # Automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler()

    # --- 5. Training and Validation Loop ---
    best_val_accuracy = 0.0
    print("\nStarting training...")

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_dataset)
        train_accuracy = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_dataset)
        val_accuracy = 100 * val_correct / val_total

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
        )

        scheduler.step()

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(
                f"✨ New best model saved to {model_path} with accuracy: {best_val_accuracy:.2f}%"
            )

    print("\nTraining finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Swin Transformer on fundus images."
    )

    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory to save results.",
    )

    # FIX: Corrected the default model name to a valid timm model identifier.
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_window12_192.ms_in22k",
        help="Name of the timm model to use.",
    )

    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )

    args = parser.parse_args()
    train_model(args)
