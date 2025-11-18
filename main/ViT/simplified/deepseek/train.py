import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Swin-V2-B on Fundus Images")
    parser.add_argument(
        "--train_dir", type=str, required=True, help="Path to training dataset"
    )
    parser.add_argument(
        "--val_dir", type=str, required=True, help="Path to validation dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for logs and models",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument("--input_size", type=int, default=256, help="Input image size")
    args = parser.parse_args()

    # Configuration
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # Data Augmentation & Transformation
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
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
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )

    # Datasets
    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)

    # Data Loaders
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
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model Setup
    model = timm.create_model(
        "swinv2_base_window8_256",
        pretrained=True,
        num_classes=len(train_dataset.classes),
    )
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training Loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = train_loss / len(train_loader.dataset)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Validation
        val_acc, val_loss = validate(model, val_loader, criterion, device)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, "best_model.pth")
            )

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {time.time()-start_time:.2f}s"
        )

    writer.close()
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


def validate(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(data_loader.dataset)
    accuracy = correct / total
    return accuracy, val_loss


if __name__ == "__main__":
    main()
