import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.transforms import autoaugment, functional
import numpy as np
from tqdm import tqdm
import json


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_dim = max(w, h)
        hp = (max_dim - w) // 2
        vp = (max_dim - h) // 2
        padding = (
            hp,
            vp,
            hp + (0 if (max_dim - w) % 2 == 0 else 1),
            vp + (0 if (max_dim - h) % 2 == 0 else 1),
        )
        return functional.pad(image, padding, 0, "constant")


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define transforms
    train_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((args.img_size, args.img_size)),
            autoaugment.RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_root, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_root, "val"), transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize model
    model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features, len(train_dataset.classes)
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=args.lr * 1e-2
    )

    # Training variables
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * images.size(0)

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"
            ):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, "best_model.pth")
            )
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    print(
        f"Training complete. Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune ConvNeXt-L for ophthalmology"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root directory of dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model weights and logs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loader workers"
    )

    args = parser.parse_args()
    main(args)
