import os
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import timm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class SafeColorJitter(object):
    """A safer implementation of ColorJitter that handles edge cases"""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        try:
            # Use torchvision's ColorJitter but with error handling
            transform = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=min(self.hue, 0.05),  # Limit hue to prevent overflow
            )
            return transform(img)
        except:
            # If ColorJitter fails, return the original image
            return img


class FundusDataset(Dataset):
    """Custom dataset for fundus images with advanced augmentations"""

    def __init__(self, root_dir, transform=None, is_training=True):
        self.dataset = ImageFolder(root_dir)
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            image, label = self.dataset[idx]

            # Ensure image is in RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            # If there's an error loading this image, return a black image
            print(f"Error loading image at index {idx}: {str(e)}")
            if self.transform:
                # Create a black RGB image
                image = Image.new("RGB", (384, 384), color="black")
                image = self.transform(image)
            else:
                image = torch.zeros(3, 384, 384)
            return image, 0  # Return class 0 as default


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing"""

    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def get_transforms(input_size=384, is_training=True):
    """Get data augmentation transforms optimized for fundus images"""

    if is_training:
        transform_list = [
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            SafeColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02
            ),  # Reduced hue
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        # Add RandomErasing after ToTensor
        transform = transforms.Compose(transform_list)

        # Create a composed transform with RandomErasing
        return transforms.Compose(
            [
                transform,
                transforms.RandomErasing(
                    p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value="random"
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


def create_model(num_classes, input_size=384, pretrained=True):
    """Create Swin model with proper initialization based on input size"""

    # Model configurations based on input size
    if input_size == 192:
        model_configs = [
            ("swinv2_base_window12_192", 192),
            ("swinv2_base_window12_192_22k", 192),
        ]
    elif input_size == 224:
        model_configs = [
            ("swin_base_patch4_window7_224", 224),
            ("swin_base_patch4_window7_224_in22k", 224),
        ]
    elif input_size == 256:
        model_configs = [
            ("swinv2_base_window16_256", 256),
            ("swinv2_base_window8_256", 256),
        ]
    else:  # 384 or other sizes
        model_configs = [
            ("swin_large_patch4_window12_384", 384),
            ("swin_large_patch4_window12_384_in22k", 384),
            ("swin_base_patch4_window12_384", 384),
            ("swin_base_patch4_window12_384_in22k", 384),
        ]

    # Add fallback options
    model_configs.extend(
        [
            ("swin_base_patch4_window7_224", 224),
            ("swinv2_base_window12_192_22k", 192),
        ]
    )

    model = None
    actual_input_size = input_size

    for model_name, expected_size in model_configs:
        try:
            print(f"Trying to load model: {model_name}")
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=0.2,
                drop_path_rate=0.2,
            )
            actual_input_size = expected_size
            print(
                f"Successfully loaded model: {model_name} (input size: {expected_size})"
            )
            break
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
            continue

    if model is None:
        # Final fallback - use a ResNet model
        print("Warning: No Swin model available, using ResNet50 instead")
        model = timm.create_model(
            "resnet50", pretrained=pretrained, num_classes=num_classes, drop_rate=0.2
        )
        actual_input_size = input_size  # ResNet can handle various input sizes

    # Initialize classifier layer properly
    if hasattr(model, "head"):
        if hasattr(model.head, "fc"):
            # If the head has an fc layer
            if hasattr(model.head.fc, "weight"):
                nn.init.xavier_uniform_(model.head.fc.weight)
                if model.head.fc.bias is not None:
                    nn.init.constant_(model.head.fc.bias, 0)
        elif hasattr(model.head, "weight"):
            # If the head directly has weight
            nn.init.xavier_uniform_(model.head.weight)
            if model.head.bias is not None:
                nn.init.constant_(model.head.bias, 0)
        else:
            # For other structures, iterate through modules
            for m in model.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    elif hasattr(model, "fc"):
        # For models with fc layer (like ResNet)
        if hasattr(model.fc, "weight"):
            nn.init.xavier_uniform_(model.fc.weight)
            if model.fc.bias is not None:
                nn.init.constant_(model.fc.bias, 0)

    return model, actual_input_size


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    device,
    use_mixup=True,
    mixup_alpha=0.2,
):
    """Train for one epoch with mixed precision and optional mixup"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        try:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Apply mixup
            if use_mixup and np.random.random() > 0.5:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, alpha=mixup_alpha
                )

                with autocast():
                    outputs = model(images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)

            # For accuracy calculation with mixup
            if use_mixup and "labels_a" in locals():
                correct += (
                    lam * predicted.eq(labels_a).sum().item()
                    + (1 - lam) * predicted.eq(labels_b).sum().item()
                )
            else:
                correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
            )

        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue

    return running_loss / max(len(dataloader), 1), correct / max(total, 1)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                images, labels = images.to(device), labels.to(device)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
                )

            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue

    return running_loss / max(len(dataloader), 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train Swin-V2-B for fundus image classification"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to data root folder"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to results folder"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--input_size",
        type=int,
        default=192,
        help="Input image size (192, 224, 256, or 384)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument(
        "--use_mixup", action="store_true", help="Use mixup augmentation"
    )
    parser.add_argument(
        "--mixup_alpha", type=float, default=0.2, help="Mixup alpha parameter"
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create temporary dataset to get number of classes
    temp_dataset = ImageFolder(os.path.join(args.data_root, "train"))
    num_classes = len(temp_dataset.classes)
    class_names = temp_dataset.classes
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

    # Create model and get actual input size
    model, actual_input_size = create_model(
        num_classes, input_size=args.input_size, pretrained=True
    )
    model = model.to(device)

    if actual_input_size != args.input_size:
        print(
            f"Warning: Model requires input size {actual_input_size}, adjusting from {args.input_size}"
        )
        args.input_size = actual_input_size

    # Create datasets with correct input size
    try:
        train_dataset = FundusDataset(
            os.path.join(args.data_root, "train"),
            transform=get_transforms(args.input_size, is_training=True),
            is_training=True,
        )

        val_dataset = FundusDataset(
            os.path.join(args.data_root, "val"),
            transform=get_transforms(args.input_size, is_training=False),
            is_training=False,
        )
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return

    # Create dataloaders with fewer workers to avoid issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(args.num_workers, 2),  # Limit workers to avoid overflow issues
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,  # Disable persistent workers to avoid memory issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        pin_memory=True,
        persistent_workers=False,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup training components
    criterion = LabelSmoothingCrossEntropy(num_classes, smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\nStarting training...")
    print(f"Using mixup: {args.use_mixup}")
    print(f"Input size: {args.input_size}x{args.input_size}")

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_mixup=args.use_mixup,
            mixup_alpha=args.mixup_alpha,
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log results
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "class_names": class_names,
                "input_size": args.input_size,
                "args": vars(args),
            }

            torch.save(checkpoint, os.path.join(args.results_dir, "best_model.pth"))
            print(f"Saved best model with validation accuracy: {best_val_acc:.4f}")

    print(
        f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}"
    )

    # Save training history
    with open(os.path.join(args.results_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Save final model
    final_checkpoint = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "best_val_acc": best_val_acc,
        "class_names": class_names,
        "input_size": args.input_size,
        "args": vars(args),
    }
    torch.save(final_checkpoint, os.path.join(args.results_dir, "final_model.pth"))

    # Save configuration
    config = {
        "model": "swinv2_base_window12_192",
        "num_classes": num_classes,
        "class_names": class_names,
        "input_size": args.input_size,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "training_args": vars(args),
    }

    with open(os.path.join(args.results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()
