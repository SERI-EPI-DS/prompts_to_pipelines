import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.cuda.amp import GradScaler, autocast
import timm
from timm.data import create_transform
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import wandb
from datetime import datetime
import warnings
from PIL import ImageEnhance, Image
import random

warnings.filterwarnings("ignore")

# Available Swin-V2 models with their expected input sizes
SWIN_MODELS = {
    "swinv2_tiny_window8_256": 256,
    "swinv2_small_window8_256": 256,
    "swinv2_base_window8_256": 256,
    "swinv2_base_window12_192": 192,
    "swinv2_base_window16_256": 256,
    "swinv2_base_window24_384": 384,
    "swinv2_large_window12_192": 192,
    "swinv2_large_window16_256": 256,
    "swinv2_large_window24_384": 384,
}


class SafeColorJitter:
    """Custom ColorJitter that handles edge cases better"""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        # Apply transformations with try-except to handle edge cases
        try:
            # Brightness
            if self.brightness > 0:
                brightness_factor = 1 + random.uniform(
                    -self.brightness, self.brightness
                )
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness_factor)

            # Contrast
            if self.contrast > 0:
                contrast_factor = 1 + random.uniform(-self.contrast, self.contrast)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast_factor)

            # Saturation
            if self.saturation > 0:
                saturation_factor = 1 + random.uniform(
                    -self.saturation, self.saturation
                )
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(saturation_factor)

            # Hue - using a safer approach
            if self.hue > 0:
                # Convert to HSV, adjust hue, convert back
                img_array = np.array(img)
                # Skip hue adjustment if image is grayscale
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    import cv2

                    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
                    hue_shift = (
                        random.uniform(-self.hue, self.hue) * 180
                    )  # OpenCV uses 0-180 for hue
                    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                    img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                    img = Image.fromarray(img_array)
        except Exception as e:
            # If any transformation fails, return original image
            print(
                f"Warning: Color jitter failed with error: {e}. Using original image."
            )
            pass

        return img


class FundusDataset(datasets.ImageFolder):
    """Custom dataset for fundus images with additional preprocessing"""

    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.samples_per_class = self._calculate_samples_per_class()

    def _calculate_samples_per_class(self):
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def __getitem__(self, index):
        """Override to handle potential errors in image loading"""
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"Error loading image at index {index}: {e}")
            # Return a black image with the correct label as fallback
            path, target = self.samples[index]
            img = Image.new("RGB", (256, 256), color="black")
            if self.transform is not None:
                img = self.transform(img)
            return img, target


def create_train_transform(input_size=256):
    """Create augmentation pipeline for training with safer transforms"""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            SafeColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),  # Reduced hue value
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_val_transform(input_size=256):
    """Create transform pipeline for validation/testing"""
    return transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.143)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_epoch(
    model, train_loader, criterion, optimizer, scaler, device, mixup_fn=None
):
    """Train for one epoch with error handling"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batches_processed = 0

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

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batches_processed += 1

            if mixup_fn is None:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100.0 * correct / total
            else:
                acc = 0.0

            pbar.set_postfix({"loss": loss.item(), "acc": acc})

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    avg_loss = (
        running_loss / batches_processed if batches_processed > 0 else float("inf")
    )
    return avg_loss, acc


def validate(model, val_loader, criterion, device):
    """Validate the model with error handling"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    batches_processed = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs, targets = inputs.to(device), targets.to(device)

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                running_loss += loss.item()
                batches_processed += 1
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = (
        running_loss / batches_processed if batches_processed > 0 else float("inf")
    )

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train Swin-V2-B for Fundus Image Classification"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for models and logs",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_window16_256",
        choices=list(SWIN_MODELS.keys()),
        help="Swin-V2 model variant",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=None,
        help="Input image size (auto-detected if not specified)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--mixup_alpha", type=float, default=0.2, help="Mixup alpha parameter"
    )
    parser.add_argument(
        "--cutmix_alpha", type=float, default=1.0, help="CutMix alpha parameter"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="fundus-classification",
        help="W&B project name",
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_true",
        help="Disable data augmentation for debugging",
    )

    args = parser.parse_args()

    # Auto-detect input size based on model if not specified
    if args.input_size is None:
        args.input_size = SWIN_MODELS[args.model_name]
        print(
            f"Auto-detected input size: {args.input_size} for model {args.model_name}"
        )
    else:
        # Verify that the specified input size is compatible
        expected_size = SWIN_MODELS[args.model_name]
        if args.input_size != expected_size:
            print(
                f"Warning: Model {args.model_name} expects input size {expected_size}, but {args.input_size} was specified."
            )
            print(f"Using {expected_size} instead.")
            args.input_size = expected_size

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save configuration
    config = vars(args)
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project=args.project_name, config=args)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create transforms
    if args.no_augmentation:
        # Minimal transforms for debugging
        train_transform = transforms.Compose(
            [
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        train_transform = create_train_transform(args.input_size)

    val_transform = create_val_transform(args.input_size)

    # Load datasets with error handling
    try:
        train_dataset = FundusDataset(
            os.path.join(args.data_dir, "train"), transform=train_transform
        )
        val_dataset = FundusDataset(
            os.path.join(args.data_dir, "val"), transform=val_transform
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Calculate class weights for imbalanced datasets
    train_targets = [s[1] for s in train_dataset.samples]
    try:
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(train_targets), y=train_targets
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    except Exception as e:
        print(f"Warning: Could not compute class weights: {e}. Using uniform weights.")
        class_weights = None

    # Create data loaders with reduced workers if necessary
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(args.num_workers, 2),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Initialize model
    num_classes = len(train_dataset.classes)
    print(
        f"Initializing {args.model_name} with {num_classes} classes and input size {args.input_size}"
    )

    try:
        model = timm.create_model(
            args.model_name, pretrained=True, num_classes=num_classes
        )
        model = model.to(device)
    except Exception as e:
        print(f"Error creating model {args.model_name}: {e}")
        print("Available Swin-V2 models:")
        for model_name, input_size in SWIN_MODELS.items():
            print(f"  - {model_name} (input size: {input_size})")
        return

    # Save class names
    class_names = train_dataset.classes
    with open(os.path.join(experiment_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Initialize loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Initialize mixed precision training
    scaler = GradScaler()

    # Initialize mixup/cutmix (disable if no augmentation)
    if args.no_augmentation or args.mixup_alpha == 0:
        mixup_fn = None
    else:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=0.5,
            switch_prob=0.5,
            mode="batch",
            num_classes=num_classes,
        )

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=15)

    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []

    print(f"\nStarting training:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Input size: {args.input_size}x{args.input_size}")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            SoftTargetCrossEntropy() if mixup_fn is not None else criterion,
            optimizer,
            scaler,
            device,
            mixup_fn,
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Log metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "class_names": class_names,
                    "model_name": args.model_name,
                    "input_size": args.input_size,
                },
                os.path.join(experiment_dir, "best_model.pth"),
            )
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "model_name": args.model_name,
                    "input_size": args.input_size,
                },
                os.path.join(experiment_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_acc,
        "model_name": args.model_name,
        "input_size": args.input_size,
        "args": vars(args),
    }

    with open(os.path.join(experiment_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
