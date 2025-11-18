import os
import argparse
import time
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from timm.loss import LabelSmoothingCrossEntropy

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Swin-V2 for ophthalmology classification"
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
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--img_size", type=int, default=256, help="Input image size"
    )  # Changed to 256 for Swin V2
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_window8_256",
        choices=[
            "swinv2_base_window8_256",
            "swinv2_small_window8_256",
            "swinv2_tiny_window8_256",
        ],
        help="Swin Transformer V2 model variant",
    )
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_transforms(img_size, is_training=False):
    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
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


def get_model_parameters(model, model_name):
    """Get parameter groups for different learning rates based on model architecture"""
    # Get all parameters except the head
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": head_params, "lr_mult": 1.0},
        {"params": backbone_params, "lr_mult": 0.1},
    ]

    return param_groups


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
        if scheduler and not isinstance(scheduler, CosineAnnealingLR):
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
    all_preds = []
    all_labels = []

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

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{100.*correct_predictions/total_samples:.2f}%",
                }
            )

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct_predictions / total_samples
    balanced_acc = 100.0 * balanced_accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, balanced_acc


def main():
    args = get_args()
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    train_transform = create_transforms(args.img_size, is_training=True)
    val_transform = create_transforms(args.img_size, is_training=False)

    train_dataset = ImageFolder(
        root=os.path.join(args.data_root, "train"), transform=train_transform
    )
    val_dataset = ImageFolder(
        root=os.path.join(args.data_root, "val"), transform=val_transform
    )

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

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")

    # Create model - Swin V2 models expect 256x256 input
    try:
        model = timm.create_model(
            args.model_name,
            pretrained=True,
            num_classes=len(train_dataset.classes),
        )
        print(f"Successfully loaded {args.model_name}")
    except Exception as e:
        print(f"Error loading {args.model_name}: {e}")
        print("Falling back to swinv2_tiny_window8_256")
        model = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=True,
            num_classes=len(train_dataset.classes),
        )

    model = model.to(device)

    # Print model info
    print(f"Model: {args.model_name}")
    print(f"Input size: {args.img_size}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    if args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Get parameter groups and create optimizer
    param_groups = get_model_parameters(model, args.model_name)

    # Apply learning rate multipliers
    optimizer_params = []
    base_lr = args.lr
    for group in param_groups:
        lr = base_lr * group["lr_mult"]
        optimizer_params.append({"params": group["params"], "lr": lr})
        print(f"Parameter group LR: {lr:.2e} (multiplier: {group['lr_mult']})")

    optimizer = AdamW(optimizer_params, lr=base_lr, weight_decay=args.weight_decay)

    # Schedulers
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    # Create separate schedulers for warmup and cosine annealing
    schedulers = []
    if args.warmup_epochs > 0:
        warmup_scheduler = OneCycleLR(
            optimizer,
            max_lr=[pg["lr"] for pg in optimizer_params],
            total_steps=warmup_steps,
            pct_start=0.1,
        )
        schedulers.append(("warmup", warmup_scheduler))

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    schedulers.append(("cosine", cosine_scheduler))

    # Training variables
    best_val_acc = 0.0
    best_val_balanced_acc = 0.0
    best_model_weights = None
    train_history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_balanced_acc": [],
    }

    print("Starting training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler=(
                warmup_scheduler if epoch < args.warmup_epochs else cosine_scheduler
            ),
        )

        # Validate
        val_loss, val_acc, val_balanced_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        # Update history
        train_history["epoch"].append(epoch + 1)
        train_history["train_loss"].append(train_loss)
        train_history["train_acc"].append(train_acc)
        train_history["val_loss"].append(val_loss)
        train_history["val_acc"].append(val_acc)
        train_history["val_balanced_acc"].append(val_balanced_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Balanced Acc: {val_balanced_acc:.2f}%"
        )

        # Save best model based on balanced accuracy
        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_model_weights,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_balanced_acc": best_val_balanced_acc,
                    "val_acc": best_val_acc,
                    "classes": train_dataset.classes,
                    "class_to_idx": train_dataset.class_to_idx,
                    "model_name": args.model_name,
                    "img_size": args.img_size,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )
            print(
                f"New best model saved with balanced accuracy: {best_val_balanced_acc:.2f}%"
            )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_balanced_acc": val_balanced_acc,
                    "classes": train_dataset.classes,
                    "class_to_idx": train_dataset.class_to_idx,
                    "model_name": args.model_name,
                    "img_size": args.img_size,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

    # Calculate total training time
    end_time = time.time()
    total_time = end_time - start_time
    print(
        f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )

    # Save final model and training history
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_balanced_acc": val_balanced_acc,
            "classes": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx,
            "model_name": args.model_name,
            "img_size": args.img_size,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "final_model.pth"),
    )

    # Save training history
    history_df = pd.DataFrame(train_history)
    history_df.to_csv(
        os.path.join(args.output_dir, "training_history.csv"), index=False
    )

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_history["epoch"], train_history["train_loss"], label="Train Loss")
    plt.plot(train_history["epoch"], train_history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(train_history["epoch"], train_history["train_acc"], label="Train Acc")
    plt.plot(train_history["epoch"], train_history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(
        train_history["epoch"],
        train_history["val_balanced_acc"],
        label="Val Balanced Acc",
        color="red",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy (%)")
    plt.legend()
    plt.title("Validation Balanced Accuracy")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, "training_history.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nTraining completed!")
    print(f"Best validation balanced accuracy: {best_val_balanced_acc:.2f}%")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
