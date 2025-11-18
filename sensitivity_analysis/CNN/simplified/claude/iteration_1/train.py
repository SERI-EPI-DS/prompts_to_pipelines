import os
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class FundusImageDataset(datasets.ImageFolder):
    """Custom dataset for fundus images with additional augmentations"""

    def __init__(self, root, transform=None):
        super().__init__(root, transform)


def get_transforms(image_size=224, is_training=True):
    """Get data transforms appropriate for fundus images"""
    if is_training:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                ),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transform


class ConvNeXtClassifier(nn.Module):
    """ConvNeXt-L with custom classification head"""

    def __init__(self, num_classes, dropout_rate=0.2):
        super().__init__()
        # Load pretrained ConvNeXt-L
        self.base_model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)

        # Get the in_features of the final layer
        in_features = self.base_model.classifier[2].in_features

        # Replace the classifier with a custom head
        self.base_model.classifier[2] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, writer):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix(
            {"loss": running_loss / (batch_idx + 1), "acc": 100.0 * correct / total}
        )

        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("Train/Loss", loss.item(), global_step)

    return running_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device, epoch, writer, phase="Val"):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"{phase} Epoch {epoch}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                label_item = label.item()
                if label_item not in class_correct:
                    class_correct[label_item] = 0
                    class_total[label_item] = 0
                class_total[label_item] += 1
                if label_item == pred.item():
                    class_correct[label_item] += 1

            pbar.set_postfix(
                {"loss": running_loss / (len(pbar) + 1), "acc": 100.0 * correct / total}
            )

    avg_loss = running_loss / len(dataloader)
    avg_acc = 100.0 * correct / total

    # Log to tensorboard
    writer.add_scalar(f"{phase}/Loss", avg_loss, epoch)
    writer.add_scalar(f"{phase}/Accuracy", avg_acc, epoch)

    # Log per-class accuracy
    for class_idx in sorted(class_correct.keys()):
        class_acc = 100.0 * class_correct[class_idx] / class_total[class_idx]
        writer.add_scalar(f"{phase}/Class_{class_idx}_Accuracy", class_acc, epoch)

    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train ConvNeXt-L on Fundus Images")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"convnext_l_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.1f} GB")

    # Data transforms
    train_transform = get_transforms(args.image_size, is_training=True)
    val_transform = get_transforms(args.image_size, is_training=False)

    # Datasets
    train_dataset = FundusImageDataset(
        os.path.join(args.data_dir, "train"), transform=train_transform
    )
    val_dataset = FundusImageDataset(
        os.path.join(args.data_dir, "val"), transform=val_transform
    )

    # Get number of classes and save class mapping
    num_classes = len(train_dataset.classes)
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    with open(os.path.join(output_dir, "class_mapping.json"), "w") as f:
        json.dump(
            {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=4
        )

    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size}")

    # Data loaders
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

    # Model
    model = ConvNeXtClassifier(num_classes=num_classes, dropout_rate=args.dropout)
    model = model.to(device)

    # Enable multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Mixed precision training
    scaler = GradScaler()

    # Tensorboard
    writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, writer
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        # Step scheduler
        scheduler.step()

        # Log learning rate
        writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], epoch)

        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "args": vars(args),
        }

        torch.save(checkpoint, os.path.join(output_dir, "last_checkpoint.pth"))

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(output_dir, "best_checkpoint.pth"))
            print(f"New best validation accuracy: {val_acc:.2f}%")

    writer.close()
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved in: {output_dir}")


if __name__ == "__main__":
    main()
