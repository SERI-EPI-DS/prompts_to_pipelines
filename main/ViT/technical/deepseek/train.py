import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from timm.data.auto_augment import rand_augment_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.cuda.amp import GradScaler, autocast


def parse_args():
    parser = argparse.ArgumentParser(
        description="Swin-V2-B Fine-tuning for Ophthalmology"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of dataset (contains train/val/test)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save models and results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Input batch size (default: 16)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing coefficient (default: 0.1)",
    )
    parser.add_argument(
        "--img_size", type=int, default=256, help="Input image size (default: 256)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    return parser.parse_args()


def setup_environment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    return device


def create_transforms(args, is_training=True):
    if is_training:
        # RandAugment with ophthalmology-friendly parameters
        rand_aug = rand_augment_transform(
            config_str="rand-m9-mstd0.5", hparams={"translate_const": 117}
        )

        return transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(15),
                rand_aug,
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )


def main():
    args = parse_args()
    device = setup_environment(args)

    # Create datasets
    train_transform = create_transforms(args, is_training=True)
    val_transform = create_transforms(args, is_training=False)

    train_dataset = ImageFolder(
        root=os.path.join(args.data_root, "train"), transform=train_transform
    )
    val_dataset = ImageFolder(
        root=os.path.join(args.data_root, "val"), transform=val_transform
    )

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

    print(
        f"Found {len(train_dataset)} training images in {len(train_dataset.classes)} classes"
    )
    print(f"Found {len(val_dataset)} validation images")

    # Initialize model
    model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, len(train_dataset.classes))
    model = model.to(device)

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training variables
    best_acc = 0.0
    scaler = GradScaler()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

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

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calculate metrics
        epoch_loss = running_loss / len(train_dataset)
        val_loss = val_loss / len(val_dataset)
        val_acc = 100.0 * correct / total

        # Update scheduler
        scheduler.step()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "class_to_idx": train_dataset.class_to_idx,
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Time: {time.time()-start_time:.1f}s | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
