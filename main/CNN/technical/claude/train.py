import os
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PIL import Image
import random
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class CustomImageFolder(ImageFolder):
    """Custom ImageFolder to return image paths along with images and labels."""

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""

    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def get_transforms(is_training=True, input_size=384):
    """Get image transforms with state-of-the-art augmentations for medical images."""

    # ConvNext normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if is_training:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                # Fixed ColorJitter - removed hue adjustment to avoid overflow
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(5, 5))], p=0.3
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=0, shear=10)], p=0.3
                ),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(int(input_size * 1.14)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transform


def create_model(num_classes, pretrained=True):
    """Create ConvNext-L model with appropriate modifications."""
    # Use the weights parameter correctly
    model = models.convnext_large(
        weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
    )

    # Modify the classifier head
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)

    return model


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    for batch_idx, (inputs, targets, _) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()

        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{running_loss/(batch_idx+1):.4f}",
                "Acc": f"{100.*correct/total:.2f}%",
            }
        )

    return running_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
        for inputs, targets, _ in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {
                    "Loss": f"{running_loss/len(dataloader):.4f}",
                    "Acc": f"{100.*correct/total:.2f}%",
                }
            )

    return running_loss / len(dataloader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Train ConvNext-L for ophthalmology image classification"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to data root folder"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to results folder"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Set deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    # Data paths
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    # Create datasets
    train_transform = get_transforms(is_training=True, input_size=args.input_size)
    val_transform = get_transforms(is_training=False, input_size=args.input_size)

    try:
        train_dataset = CustomImageFolder(train_dir, transform=train_transform)
        val_dataset = CustomImageFolder(val_dir, transform=val_transform)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Save class mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(os.path.join(args.results_dir, "class_mapping.json"), "w") as f:
        json.dump(
            {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2
        )

    # Create data loaders with error handling
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
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

    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(
        classes=num_classes, smoothing=args.label_smoothing
    )

    # Optimizer - using AdamW with different learning rates for pretrained and new layers
    pretrained_params = []
    new_params = []
    for name, param in model.named_parameters():
        if "classifier.2" in name:
            new_params.append(param)
        else:
            pretrained_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": pretrained_params, "lr": args.lr * 0.1},
            {"params": new_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    max_epochs = 50
    patience = 10
    patience_counter = 0

    train_history = {"loss": [], "acc": []}
    val_history = {"loss": [], "acc": []}

    print("\nStarting training...")

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        train_history["loss"].append(train_loss)
        train_history["acc"].append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        val_history["loss"].append(val_loss)
        val_history["acc"].append(val_acc)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Current LR: {current_lr:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "args": args,
            }

            checkpoint_path = os.path.join(args.results_dir, "best_model.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "args": args,
            }
            checkpoint_path = os.path.join(
                args.results_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(checkpoint, checkpoint_path)

    # Save training history
    history = {
        "train": train_history,
        "val": val_history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
    }
    with open(os.path.join(args.results_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(
        f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}"
    )


if __name__ == "__main__":
    main()
