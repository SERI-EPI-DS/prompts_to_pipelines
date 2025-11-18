import os
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import timm
from timm.data import create_transform
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
import wandb
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class FundusDataset(torch.utils.data.Dataset):
    """Custom dataset for fundus images with additional augmentations"""

    def __init__(self, root_dir, transform=None, is_training=False):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(config, is_training=False):
    """Get augmentation transforms optimized for fundus images"""
    if is_training:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(config["image_size"], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((config["image_size"], config["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transform


def create_model(config, num_classes):
    """Create Swin-V2-B model with proper initialization"""
    # You can choose different Swin-V2 variants:
    # 'swinv2_base_window12_192_22k' - expects 192x192 input
    # 'swinv2_base_window16_256' - expects 256x256 input
    # 'swinv2_base_window12to24_192to384_22kft1k' - can handle variable sizes

    if config["image_size"] == 192:
        model_name = "swinv2_base_window12_192_22k"
    elif config["image_size"] == 256:
        model_name = "swinv2_base_window16_256"
    elif config["image_size"] == 384:
        model_name = "swinv2_base_window12to24_192to384_22kft1k"
    else:
        # For other sizes, use the flexible variant and it will interpolate
        print(
            f"Warning: Image size {config['image_size']} is not standard for Swin-V2. Using flexible variant."
        )
        model_name = "swinv2_base_window12to24_192to384_22kft1k"

    print(f"Creating model: {model_name}")

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=config["drop_rate"],
        drop_path_rate=config["drop_path_rate"],
        img_size=config["image_size"],  # Explicitly set image size
    )

    # Initialize the final layer with smaller weights for stable training
    # Handle different possible head structures in timm models
    if hasattr(model, "head"):
        if hasattr(model.head, "fc"):
            # Some models have head.fc
            if model.head.fc is not None:
                nn.init.xavier_uniform_(model.head.fc.weight)
                if model.head.fc.bias is not None:
                    nn.init.zeros_(model.head.fc.bias)
        elif hasattr(model.head, "weight"):
            # Direct linear layer
            nn.init.xavier_uniform_(model.head.weight)
            if model.head.bias is not None:
                nn.init.zeros_(model.head.bias)
        else:
            # For ClassifierHead or other complex heads, find the actual linear layer
            for name, module in model.head.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    break
    elif hasattr(model, "classifier"):
        # Some models use 'classifier' instead of 'head'
        if isinstance(model.classifier, nn.Linear):
            nn.init.xavier_uniform_(model.classifier.weight)
            if model.classifier.bias is not None:
                nn.init.zeros_(model.classifier.bias)

    # Print model structure for debugging
    print(f"Model created with {num_classes} classes")
    print(f"Expected input size: {config['image_size']}x{config['image_size']}")

    return model


def train_epoch(
    model, train_loader, criterion, optimizer, scaler, config, mixup_fn=None
):
    """Train for one epoch with mixed precision training"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.cuda(), labels.cuda()

        # Apply mixup if enabled
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Calculate accuracy for non-mixup case
        if mixup_fn is None:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total if mixup_fn is None else 0

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    return epoch_loss, accuracy, precision, recall, f1, all_labels, all_preds


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
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
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    # Configuration
    config = {
        "image_size": args.image_size,  # Now configurable
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-4,
        "drop_rate": 0.1,
        "drop_path_rate": 0.2,
        "grad_clip": 1.0,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0,
        "label_smoothing": 0.1,
        "warmup_epochs": 5,
        "min_lr": 1e-6,
    }

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"swin_v2_b_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project="fundus-classification",
            name=f"swin_v2_b_{timestamp}",
            config=config,
        )

    # Create datasets
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)

    train_dataset = FundusDataset(
        os.path.join(args.data_dir, "train"),
        transform=train_transform,
        is_training=True,
    )
    val_dataset = FundusDataset(
        os.path.join(args.data_dir, "val"), transform=val_transform, is_training=False
    )

    # Get number of classes and class names
    num_classes = len(train_dataset.dataset.classes)
    class_names = train_dataset.dataset.classes

    print(f"\nDataset Information:")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    model = create_model(config, num_classes).cuda()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create loss function
    if config["mixup_alpha"] > 0 or config["cutmix_alpha"] > 0:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    # Create scheduler with warmup
    num_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = len(train_loader) * config["warmup_epochs"]

    scheduler = CosineAnnealingLR(
        optimizer, T_max=num_steps - warmup_steps, eta_min=config["min_lr"]
    )

    # Create mixup augmentation
    mixup_fn = None
    if config["mixup_alpha"] > 0 or config["cutmix_alpha"] > 0:
        mixup_fn = Mixup(
            mixup_alpha=config["mixup_alpha"],
            cutmix_alpha=config["cutmix_alpha"],
            prob=0.5,
            num_classes=num_classes,
        )

    # Mixed precision training
    scaler = GradScaler()

    # Training loop
    best_val_acc = 0.0
    best_val_f1 = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []

    print("\nStarting training...")
    print(f"Image size: {config['image_size']}x{config['image_size']}")

    for epoch in range(config["num_epochs"]):
        print(f'\nEpoch [{epoch+1}/{config["num_epochs"]}]')

        # Learning rate warmup
        if epoch < config["warmup_epochs"]:
            lr = config["learning_rate"] * (epoch + 1) / config["warmup_epochs"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, config, mixup_fn
        )

        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, y_true, y_pred = validate(
            model, val_loader, nn.CrossEntropyLoss()
        )

        # Step scheduler
        if epoch >= config["warmup_epochs"]:
            scheduler.step()

        # Log metrics
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        print(
            f"Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val F1: {val_f1:.4f}"
        )
        print(f"Learning Rate: {current_lr:.6f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc * 100,
                    "val_precision": val_prec,
                    "val_recall": val_rec,
                    "val_f1": val_f1,
                    "learning_rate": current_lr,
                    "epoch": epoch + 1,
                }
            )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "config": config,
                    "class_names": class_names,
                    "image_size": config["image_size"],
                },
                os.path.join(output_dir, "best_model.pth"),
            )

            # Plot confusion matrix for best model
            plot_confusion_matrix(
                y_true,
                y_pred,
                class_names,
                os.path.join(output_dir, "confusion_matrix_best.png"),
            )

            print(f"New best model saved! F1: {val_f1:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "image_size": config["image_size"],
                },
                os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot([acc * 100 for acc in val_accuracies], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()

    # Save final metrics
    final_metrics = {
        "best_val_accuracy": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "num_epochs": config["num_epochs"],
        "image_size": config["image_size"],
        "class_names": class_names,
        "training_time": time.strftime("%H:%M:%S", time.gmtime(time.time())),
    }

    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Best validation F1 score: {best_val_f1:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Swin-V2-B for fundus image classification"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory containing train/val/test folders",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=192,
        help="Input image size (192, 256, or 384 recommended for Swin-V2)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training will be slow on CPU.")
    else:
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")

    main(args)
