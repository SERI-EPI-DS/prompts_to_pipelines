import os
import argparse
import json
import time
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FundusDataset(Dataset):
    """Custom dataset for fundus images with additional augmentations."""

    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(input_size=384, training=True):
    """Get augmentation transforms optimized for fundus images."""

    if training:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1
                ),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return transform


def create_model(num_classes, pretrained=True):
    """Create Swin-V2-B model with custom head for classification."""
    # Load pre-trained Swin Transformer V2-B
    model = models.swin_v2_b(weights="IMAGENET1K_V1" if pretrained else None)

    # Modify the classifier head
    num_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (self.smoothing / n_classes)
        log_prb = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_transform = get_transforms(args.input_size, training=True)
    val_transform = get_transforms(args.input_size, training=False)

    train_dataset = FundusDataset(
        os.path.join(args.data_dir, "train"), transform=train_transform
    )
    val_dataset = FundusDataset(
        os.path.join(args.data_dir, "val"), transform=val_transform
    )

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Create model
    model = create_model(num_classes, pretrained=True)
    model = model.to(device)

    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    print("\nStarting training...")
    print("=" * 50)

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Save metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{args.epochs}] - Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
                "args": args,
            }

            torch.save(checkpoint, os.path.join(args.results_dir, "best_model.pth"))
            print(f"  >> New best model saved! Val Acc: {val_acc:.2f}%")

        print("-" * 50)

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    # Save training history
    history = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
    }

    with open(os.path.join(args.results_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nResults saved to: {args.results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Swin-V2-B for fundus image classification"
    )

    # Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing train/val/test folders",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to save results and model weights",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (default: 0.1)",
    )

    # Model parameters
    parser.add_argument(
        "--input_size", type=int, default=384, help="Input image size (default: 384)"
    )

    # Other parameters
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    main(args)
