import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
import argparse
import json
from collections import Counter


def get_data_loaders(data_dir, batch_size, img_size):
    """
    Creates training and validation data loaders with appropriate augmentations.
    """
    # State-of-the-art augmentations for training
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Standard transformations for validation (no augmentation)
    val_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes


def train_model(args):
    """
    Main function to train the ConvNext-L model.
    """
    # Setup device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get data loaders and class names
    train_loader, val_loader, class_names = get_data_loaders(
        args.data_dir, args.batch_size, args.img_size
    )
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {', '.join(class_names)}")

    # Save class names to a file for later use in testing
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Load pretrained ConvNext-L model
    model = timm.create_model(
        "convnext_large", pretrained=True, num_classes=num_classes
    )
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Automatic Mixed Precision (AMP) for faster training
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_val_accuracy = 0.0
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct_predictions / total_predictions) * 100
        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%"
        )

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = (val_correct / val_total) * 100
        print(
            f"Epoch {epoch+1}/{args.epochs} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%"
        )

        # Save the best model
        if val_epoch_acc > best_val_accuracy:
            best_val_accuracy = val_epoch_acc
            model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(
                f"New best model saved to {model_path} with accuracy: {best_val_accuracy:.2f}%"
            )

        # Step the scheduler
        scheduler.step()

    print("Finished Training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a ConvNext-L classifier on fundus images."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the train/val datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained model and results.",
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size for model input."
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size. Reduce if you have VRAM issues.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for AdamW.",
    )

    args = parser.parse_args()
    train_model(args)
