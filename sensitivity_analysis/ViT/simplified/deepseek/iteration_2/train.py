# train_swin_ophthalmology_fixed.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, SwinForImageClassification
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class OphthalmologyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(
            [
                d
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                ):
                    self.samples.append(
                        (
                            os.path.join(class_dir, img_file),
                            self.class_to_idx[class_name],
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label, img_path

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            dummy_image = torch.randn(3, 224, 224)
            return dummy_image, label, img_path


def create_transforms(args, is_training=True):
    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
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
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def setup_model(num_classes, args):
    """Initialize Swin Transformer V2 model"""
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-base-patch4-window7-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # Freeze early layers if specified
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:  # Only freeze backbone, not classifier
                param.requires_grad = False

    return model


class CustomMixup:
    """Custom Mixup implementation that works with any number of classes"""

    def __init__(self, alpha=0.2, num_classes=1000):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index]

        # Convert labels to one-hot if needed
        if len(y.shape) == 1:
            y_onehot = torch.zeros(batch_size, self.num_classes).to(y.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
            y = y_onehot

        y_a, y_b = y, y[index]
        mixed_y = lam * y_a + (1 - lam) * y_b

        return mixed_x, mixed_y


def train_epoch(model, train_loader, optimizer, criterion, device, args, mixup_fn=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Apply mixup if specified
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)
            use_mixup_loss = True
        else:
            use_mixup_loss = False

        optimizer.zero_grad()

        outputs = model(images)

        # Use appropriate loss function based on whether mixup was applied
        if use_mixup_loss:
            loss = criterion(outputs.logits, labels)
        else:
            loss = criterion(outputs.logits, labels)

        loss.backward()

        # Gradient clipping
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy (for non-mixup batches)
        if not use_mixup_loss:
            _, preds = torch.max(outputs.logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        if batch_idx % args.print_freq == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)

    if not use_mixup_loss:
        epoch_acc = accuracy_score(all_labels, all_preds)
    else:
        epoch_acc = 0.0

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs.logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")

    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


def main(args):
    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Load datasets
    train_dataset = OphthalmologyDataset(args.train_dir)
    val_dataset = OphthalmologyDataset(args.val_dir)

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")

    # Save class mapping
    class_mapping = {i: cls_name for i, cls_name in enumerate(train_dataset.classes)}
    with open(os.path.join(args.output_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=4)

    # Setup model
    model = setup_model(num_classes, args)
    model = model.to(device)

    # Setup transforms
    train_transform = create_transforms(args, is_training=True)
    val_transform = create_transforms(args, is_training=False)

    # Update datasets with transforms
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform

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

    # Setup mixup - use custom implementation to avoid dimension issues
    mixup_fn = None
    if args.mixup > 0:
        print(f"Using mixup with alpha={args.mixup} and num_classes={num_classes}")
        mixup_fn = CustomMixup(alpha=args.mixup, num_classes=num_classes)

    # Loss function - use SoftTargetCrossEntropy for mixup, regular CrossEntropy otherwise
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args, mixup_fn
        )

        # Validate
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "class_mapping": class_mapping,
                    "args": args,
                },
                best_model_path,
            )
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_mapping": class_mapping,
            "args": args,
        },
        final_model_path,
    )

    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, "training_history.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Swin Transformer V2 for Ophthalmology"
    )

    # Data parameters
    parser.add_argument(
        "--train_dir", type=str, required=True, help="Path to training data directory"
    )
    parser.add_argument(
        "--val_dir", type=str, required=True, help="Path to validation data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")

    # Advanced options
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.0,
        help="Mixup alpha parameter (set to 0 to disable)",
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze backbone layers"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm"
    )

    # System parameters
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--print_freq", type=int, default=50, help="Print frequency during training"
    )

    args = parser.parse_args()

    # Create timestamped output directory if not specified
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"swin_ophthalmology_{timestamp}"

    main(args)
