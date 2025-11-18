import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class FundusDataset(torch.utils.data.Dataset):
    """Custom dataset for fundus images with additional augmentations"""

    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_transforms(input_size=384, is_training=True):
    """Get data transforms with medical image-specific augmentations"""
    if is_training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(int(input_size * 1.1)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class ConvNextClassifier:
    def __init__(
        self,
        num_classes,
        model_name="convnext_large_in22k",
        input_size=384,
        device="cuda",
    ):
        self.device = device
        self.num_classes = num_classes
        self.input_size = input_size

        # Load pretrained ConvNeXt-L model
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=0.3,
            drop_path_rate=0.2,
        ).to(device)

        # Initialize loss functions
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mixup_fn = Mixup(
            mixup_alpha=0.2,
            cutmix_alpha=1.0,
            prob=0.5,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.1,
            num_classes=num_classes,
        )
        self.mixup_criterion = SoftTargetCrossEntropy()

    def train_epoch(self, train_loader, optimizer, scaler, epoch, use_mixup=True):
        """Train for one epoch with mixed precision and mixup"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Apply mixup/cutmix
            if use_mixup and self.mixup_fn is not None:
                inputs, targets = self.mixup_fn(inputs, targets)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = self.model(inputs)
                if use_mixup and self.mixup_fn is not None:
                    loss = self.mixup_criterion(outputs, targets)
                else:
                    loss = self.criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Calculate accuracy (only for non-mixup batches)
            if not use_mixup or self.mixup_fn is None:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({"loss": running_loss / (batch_idx + 1)})

        accuracy = 100.0 * correct / total if total > 0 else 0
        return running_loss / len(train_loader), accuracy

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / len(val_loader)

        return avg_loss, accuracy, all_preds, all_targets


def train_model(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"convnext_l_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(project="fundus-classification", config=args)

    # Create datasets
    train_dataset = FundusDataset(
        os.path.join(args.data_dir, "train"),
        transform=get_transforms(args.input_size, is_training=True),
    )
    val_dataset = FundusDataset(
        os.path.join(args.data_dir, "val"),
        transform=get_transforms(args.input_size, is_training=False),
    )

    # Create data loaders
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

    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")

    # Save class mapping
    class_mapping = {
        idx: class_name for class_name, idx in train_dataset.class_to_idx.items()
    }
    with open(os.path.join(output_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=4)

    # Initialize model
    classifier = ConvNextClassifier(
        num_classes=num_classes,
        model_name=args.model_name,
        input_size=args.input_size,
        device=device,
    )

    # Set up optimizer with different learning rates for different parts
    backbone_params = []
    head_params = []
    for name, param in classifier.model.named_parameters():
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.001
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    best_val_acc = 0
    best_val_loss = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc = classifier.train_epoch(
            train_loader, optimizer, scaler, epoch, use_mixup=args.use_mixup
        )

        # Validate
        val_loss, val_acc, val_preds, val_targets = classifier.validate(val_loader)

        # Update scheduler
        scheduler.step()

        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": classifier.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "class_mapping": class_mapping,
                    "args": args,
                },
                os.path.join(output_dir, "best_model.pth"),
            )

            # Save classification report
            report = classification_report(
                val_targets,
                val_preds,
                target_names=train_dataset.classes,
                output_dict=True,
            )
            with open(
                os.path.join(output_dir, "best_classification_report.json"), "w"
            ) as f:
                json.dump(report, f, indent=4)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

        # Save checkpoint every N epochs
        if epoch % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": classifier.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth"),
            )

    # Plot training history
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
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt-L for Fundus Image Classification"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory containing train/val/test folders",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Directory to save outputs"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="convnext_large_in22k",
        help="Model name from timm library",
    )
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )
    parser.add_argument(
        "--save_interval", type=int, default=5, help="Save checkpoint every N epochs"
    )

    # Augmentation arguments
    parser.add_argument(
        "--use_mixup", action="store_true", help="Use mixup/cutmix augmentation"
    )

    # Other arguments
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )

    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
