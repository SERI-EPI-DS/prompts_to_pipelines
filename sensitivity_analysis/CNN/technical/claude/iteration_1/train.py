import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class RandomAffine(object):
    """Custom random affine transformation for medical images"""

    def __init__(self, degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        translate_x = random.uniform(-self.translate[0], self.translate[0]) * img.width
        translate_y = random.uniform(-self.translate[1], self.translate[1]) * img.height
        scale = random.uniform(self.scale[0], self.scale[1])

        return transforms.functional.affine(
            img, angle, (translate_x, translate_y), scale, shear=0
        )


def get_transforms(input_size=384, train=True):
    """Get image transformations for training/validation"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if train:
        # Create transform list with safer augmentations for fundus images
        transform_list = [
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]

        # Add custom affine transformation
        transform_list.append(
            RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
        )

        # Add ColorJitter without hue (which causes the overflow)
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.15,  # Slightly reduced for medical images
                contrast=0.15,  # Slightly reduced for medical images
                saturation=0.15,  # Slightly reduced for medical images
                hue=0,  # Removed to prevent overflow
            )
        )

        # Add remaining transforms
        transform_list.extend(
            [
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # Add random erasing after tensor conversion
        composed = transforms.Compose(transform_list)

        # Create a final transform with RandomErasing
        return transforms.Compose(
            [
                composed,
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(int(input_size * 1.1)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]
        )


def create_model(num_classes, dropout=0.2):
    """Create ConvNext-L model with custom classifier"""
    model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)

    # Replace classifier with custom head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=dropout), nn.Linear(in_features, num_classes)
    )

    return model


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        try:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Gradient clipping to prevent overflow
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue

    epoch_loss = running_loss / len(dataloader.dataset) if total > 0 else 0
    epoch_acc = 100.0 * correct / total if total > 0 else 0

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def main(args):
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage:")
        print(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
        print(f"Reserved: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")

    # Data preparation
    train_transform = get_transforms(args.input_size, train=True)
    val_transform = get_transforms(args.input_size, train=False)

    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"), transform=val_transform
    )

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Save class mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(os.path.join(args.output_dir, "class_mapping.json"), "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f)

    # Data loaders with error handling
    # Set num_workers to 0 if persistent errors occur
    num_workers = args.num_workers if not args.disable_workers else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Model setup
    model = create_model(num_classes, dropout=args.dropout)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    # Different learning rates for pretrained and new layers
    pretrained_params = []
    new_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
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
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Mixed precision training
    scaler = GradScaler()

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)

        try:
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device
            )

            # Validate
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

            # Update scheduler
            scheduler.step()

            # Log metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0

                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_to_idx": class_to_idx,
                    "args": args,
                }
                torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pth"))
                print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "class_to_idx": class_to_idx,
                    "args": args,
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
                )

            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        except KeyboardInterrupt:
            print("Training interrupted by user")
            break
        except Exception as e:
            print(f"Error during epoch {epoch+1}: {str(e)}")
            print("Attempting to continue training...")

            # If persistent errors, try disabling workers
            if "DataLoader" in str(e) and num_workers > 0:
                print("Recreating data loaders with num_workers=0...")
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )
            continue

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    # Save final model
    final_checkpoint = {
        "epoch": len(train_losses),
        "model_state_dict": model.state_dict(),
        "final_train_acc": train_accs[-1] if train_accs else 0,
        "final_val_acc": val_accs[-1] if val_accs else 0,
        "best_val_acc": best_val_acc,
        "class_to_idx": class_to_idx,
        "args": args,
    }
    torch.save(final_checkpoint, os.path.join(args.output_dir, "final_model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune ConvNext-L for fundus image classification"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--disable_workers",
        action="store_true",
        help="Disable multiprocessing in data loading",
    )

    args = parser.parse_args()
    main(args)
