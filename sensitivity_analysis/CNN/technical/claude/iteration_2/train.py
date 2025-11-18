import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time
from datetime import datetime
from pathlib import Path
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


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(-1)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class FundusDataset(Dataset):
    """Custom dataset for fundus images with advanced augmentations."""

    def __init__(self, root_dir, transform=None, is_training=True):
        self.dataset = ImageFolder(root_dir)
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Convert to RGB if grayscale
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(is_training=True, img_size=384):
    """Get data transforms optimized for fundus images."""

    # Mean and std for ImageNet pretrained models
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_training:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                # Fixed ColorJitter - removed hue parameter to avoid overflow
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.14)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transform


def create_model(num_classes, pretrained=True):
    """Create ConvNext-L model with custom head."""
    model = models.convnext_large(weights="IMAGENET1K_V1" if pretrained else None)

    # Modify the classifier head
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping to prevent instability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 10 == 0:
            print(f"  Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
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

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save training configuration
    config = vars(args)
    config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(results_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Data paths
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"

    # Get number of classes
    num_classes = len(os.listdir(train_dir))
    print(f"Number of classes: {num_classes}")

    # Create datasets
    train_transform = get_transforms(is_training=True, img_size=args.img_size)
    val_transform = get_transforms(is_training=False, img_size=args.img_size)

    train_dataset = FundusDataset(
        train_dir, transform=train_transform, is_training=True
    )
    val_dataset = FundusDataset(val_dir, transform=val_transform, is_training=False)

    # Create dataloaders with error handling
    def collate_fn(batch):
        """Custom collate function to handle potential errors."""
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    model = create_model(num_classes, pretrained=True)
    model = model.to(device)

    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    # Optimizer - AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Mixed precision training
    scaler = GradScaler()

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0
    patience_counter = 0

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(args.max_epochs):
        print(f"\nEpoch [{epoch+1}/{args.max_epochs}]")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "num_classes": num_classes,
                "class_names": train_dataset.dataset.classes,
            }

            torch.save(checkpoint, results_dir / "best_model.pth")
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "num_classes": num_classes,
                "class_names": train_dataset.dataset.classes,
            }
            torch.save(checkpoint, results_dir / f"checkpoint_epoch_{epoch+1}.pth")

    # Save final model
    final_checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
        "num_classes": num_classes,
        "class_names": train_dataset.dataset.classes,
    }
    torch.save(final_checkpoint, results_dir / "final_model.pth")

    # Save training history
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ConvNext-L for fundus image classification"
    )

    # Data arguments
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

    # Training arguments
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument("--img_size", type=int, default=384, help="Input image size")
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )

    # Other arguments
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    main(args)
