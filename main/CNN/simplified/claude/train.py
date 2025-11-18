import os
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from timm import create_model
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


class FundusDataset(datasets.ImageFolder):
    """Custom dataset for fundus images with additional augmentation options."""

    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        """Override to handle potential errors in image loading."""
        path, target = self.samples[index]
        try:
            # Open and convert image to RGB
            sample = self.loader(path)
            if sample.mode != "RGB":
                sample = sample.convert("RGB")

            if self.transform is not None:
                sample = self.transform(sample)

            return sample, target
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            # Return a black image of the expected size as fallback
            if self.transform is not None:
                # Create a black RGB image
                sample = Image.new("RGB", (384, 384), (0, 0, 0))
                sample = self.transform(sample)
            else:
                sample = torch.zeros(3, 384, 384)
            return sample, target


class SafeColorJitter(transforms.ColorJitter):
    """Safe version of ColorJitter that handles edge cases."""

    def forward(self, img):
        try:
            return super().forward(img)
        except Exception as e:
            # If ColorJitter fails, return the original image
            return img


def get_transforms(input_size=384, training=True):
    """Get data transforms for training or validation/testing."""
    if training:
        # Updated transforms without hue adjustment for medical images
        transform_list = [
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            # Removed hue from ColorJitter as it can cause issues with fundus images
            SafeColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        transform_list = [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    return transforms.Compose(transform_list)


def create_dataloaders(data_dir, batch_size, num_workers, input_size=384):
    """Create train, validation, and test dataloaders."""
    train_transform = get_transforms(input_size, training=True)
    val_transform = get_transforms(input_size, training=False)

    train_dataset = FundusDataset(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = FundusDataset(os.path.join(data_dir, "val"), transform=val_transform)

    # Use persistent_workers to avoid reloading data
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, len(train_dataset.classes)


class ConvNeXtClassifier(nn.Module):
    """ConvNeXt-Large classifier with optional dropout."""

    def __init__(self, num_classes, pretrained=True, dropout_rate=0.2):
        super().__init__()
        self.model = create_model(
            "convnext_large",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )

    def forward(self, x):
        return self.model(x)


def train_epoch(
    model, train_loader, criterion, optimizer, scaler, device, mixup_fn=None
):
    """Train for one epoch with error handling."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        try:
            inputs, targets = inputs.to(device), targets.to(device)

            if mixup_fn is not None:
                inputs, targets = mixup_fn(inputs, targets)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}, skipping...")
                continue

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if mixup_fn is None:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100.0 * correct / total
            else:
                acc = 0.0

            pbar.set_postfix({"loss": running_loss / (batch_idx + 1), "acc": acc})

        except Exception as e:
            print(f"Error in training batch {batch_idx}: {str(e)}")
            continue

    return running_loss / len(train_loader), acc


def validate(model, val_loader, criterion, device):
    """Validate the model with error handling."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs, targets = inputs.to(device), targets.to(device)

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                if not torch.isnan(loss):
                    running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                pbar.set_postfix(
                    {
                        "loss": running_loss / (batch_idx + 1),
                        "acc": 100.0 * correct / total,
                    }
                )

            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue

    return (
        running_loss / len(val_loader),
        100.0 * correct / total,
        all_preds,
        all_targets,
    )


def save_training_plots(history, output_dir):
    """Save training history plots."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if history["train_acc"][0] > 0:  # Only plot if we have accuracy values
        plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt-L for fundus image classification"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save outputs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--mixup_alpha", type=float, default=0.2, help="Mixup alpha (0 to disable)"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing"
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    try:
        train_loader, val_loader, num_classes = create_dataloaders(
            args.data_dir, args.batch_size, args.num_workers, args.input_size
        )
        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error creating dataloaders: {str(e)}")
        return

    # Create model
    model = ConvNeXtClassifier(num_classes, pretrained=True, dropout_rate=args.dropout)
    model = model.to(device)

    # Create loss function and optimizer
    if args.mixup_alpha > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=0.0,
            label_smoothing=args.label_smoothing,
            num_classes=num_classes,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        mixup_fn = None

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, mixup_fn
        )

        # Validate
        val_criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, _, _ = validate(model, val_loader, val_criterion, device)

        # Update scheduler
        scheduler.step()

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "class_names": train_loader.dataset.classes,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "history": history,
                    "class_names": train_loader.dataset.classes,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

    # Save final model
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "class_names": train_loader.dataset.classes,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "final_model.pth"),
    )

    # Save training history
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    save_training_plots(history, args.output_dir)

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
