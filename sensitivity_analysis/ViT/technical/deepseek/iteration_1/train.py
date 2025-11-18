import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from timm.data import create_transform
from timm.loss import LabelSmoothingCrossEntropy
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class FundusDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_transforms(img_size=224, is_train=True):
    if is_train:
        return create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def create_model(num_classes, pretrained=True):
    model = timm.create_model(
        "swinv2_base_window8_256", pretrained=pretrained, num_classes=num_classes
    )
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler and isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.*correct_predictions/total_samples:.2f}%",
            }
        )

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct_predictions / total_samples
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{100.*correct_predictions/total_samples:.2f}%",
                }
            )

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct_predictions / total_samples
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Swin-V2-B for Ophthalmology Diagnosis"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root directory of dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model and results",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="Minimum learning rate"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Data transforms
    train_transform = get_transforms(args.img_size, is_train=True)
    val_transform = get_transforms(args.img_size, is_train=False)

    # Datasets
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    train_dataset = FundusDataset(train_dir, transform=train_transform, is_train=True)
    val_dataset = FundusDataset(val_dir, transform=val_transform, is_train=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # DataLoaders
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

    # Model
    num_classes = len(train_dataset.classes)
    model = create_model(num_classes, pretrained=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Learning rate schedulers
    scheduler_cosine = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    scheduler_plateau = ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )

    # Training variables
    best_val_acc = 0.0
    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Number of classes: {num_classes}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler_cosine
        )

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Update plateau scheduler
        scheduler_plateau.step(val_loss)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "classes": train_dataset.classes,
                    "class_to_idx": train_dataset.class_to_idx,
                },
                best_model_path,
            )
            print(f"New best model saved with val_acc: {val_acc:.2f}%")

        # Save latest model
        latest_model_path = os.path.join(args.output_dir, "latest_model.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            },
            latest_model_path,
        )

    # Save training history
    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(args.output_dir, "training_history.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
