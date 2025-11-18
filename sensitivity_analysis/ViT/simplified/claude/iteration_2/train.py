import os
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class FundusDataset(ImageFolder):
    """Custom dataset for fundus images with additional augmentations"""

    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets"""
        class_counts = np.bincount([self.targets[i] for i in range(len(self))])
        total_samples = len(self)
        class_weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(class_weights)


def get_transforms(input_size=384, is_training=True):
    """Get augmentation transforms optimized for fundus images"""
    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((input_size + 32, input_size + 32)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, mixup_fn=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)

        if mixup_fn is None:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        progress_bar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total if mixup_fn is None else 0.0

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_targets))

    return epoch_loss, epoch_acc, all_preds, all_targets


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Save confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Swin-V2-B on fundus images")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save outputs"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument(
        "--use_mixup", action="store_true", help="Use mixup augmentation"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing"
    )
    parser.add_argument(
        "--drop_path_rate", type=float, default=0.2, help="Drop path rate"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_window12_192",
        choices=[
            "swinv2_base_window12_192",
            "swinv2_base_window8",
            "swinv2_base_window16_256",
            "swinv2_large_window12_192",
        ],
        help="Swin-V2 model variant",
    )

    args = parser.parse_args()

    # Adjust input size based on model variant
    model_input_sizes = {
        "swinv2_base_window12_192": 192,
        "swinv2_base_window8": 256,
        "swinv2_base_window16_256": 256,
        "swinv2_large_window12_192": 192,
    }

    # Override input size if using a model with specific requirements
    if args.model_name in model_input_sizes and args.input_size == 384:
        original_size = args.input_size
        args.input_size = model_input_sizes[args.model_name]
        print(
            f"Note: Adjusting input size from {original_size} to {args.input_size} for model {args.model_name}"
        )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_transform = get_transforms(args.input_size, is_training=True)
    val_transform = get_transforms(args.input_size, is_training=False)

    train_dataset = FundusDataset(
        os.path.join(args.data_dir, "train"), transform=train_transform
    )
    val_dataset = FundusDataset(
        os.path.join(args.data_dir, "val"), transform=val_transform
    )

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    # Get class weights for imbalanced data
    class_weights = train_dataset.get_class_weights().to(device)

    # Create data loaders
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
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model - using more flexible Swin-V2 variants
    print(f"Creating model: {args.model_name}")
    try:
        if args.model_name == "swinv2_base_window8":
            # This variant is more flexible with input sizes
            model = timm.create_model(
                "swinv2_base_window8_256",
                pretrained=True,
                num_classes=num_classes,
                drop_path_rate=args.drop_path_rate,
            )
        else:
            model = timm.create_model(
                args.model_name,
                pretrained=True,
                num_classes=num_classes,
                drop_path_rate=args.drop_path_rate,
                img_size=args.input_size,
            )
    except Exception as e:
        print(f"Error creating model {args.model_name}: {e}")
        print(
            "Falling back to swinv2_base_window8_256 which supports various input sizes"
        )
        model = timm.create_model(
            "swinv2_base_window8_256",
            pretrained=True,
            num_classes=num_classes,
            drop_path_rate=args.drop_path_rate,
        )

    model = model.to(device)

    # Print model info
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters:,}")

    # Model EMA for better generalization
    model_ema = ModelEmaV2(model, decay=0.9998)

    # Create loss function
    if args.use_mixup:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=0.5,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=args.label_smoothing
        )
        mixup_fn = None

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Create scheduler
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs, lr_min=1e-6, warmup_t=10, warmup_lr_init=1e-6
    )

    # Mixed precision training
    scaler = GradScaler()

    # Training loop
    best_val_acc = 0
    patience_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, mixup_fn
        )
        model_ema.update(model)

        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(
            model_ema.module, val_loader, nn.CrossEntropyLoss(), device
        )

        # Update scheduler
        scheduler.step(epoch)

        # Log results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_ema_state_dict": model_ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_names": class_names,
                    "model_name": args.model_name,
                    "input_size": args.input_size,
                },
                os.path.join(exp_dir, "best_model.pth"),
            )

            # Save confusion matrix
            save_confusion_matrix(
                val_targets,
                val_preds,
                class_names,
                os.path.join(exp_dir, "confusion_matrix.png"),
            )

            # Save classification report
            report = classification_report(
                val_targets, val_preds, target_names=class_names, output_dict=True
            )
            with open(os.path.join(exp_dir, "classification_report.json"), "w") as f:
                json.dump(report, f, indent=4)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "training_curves.png"))

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
