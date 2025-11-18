import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import argparse
import os
import json
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class LabelSmoothingCrossEntropy(nn.Module):
    """Robust label smoothing cross entropy loss implementation"""

    def __init__(self, smoothing=0.1, reduction="mean"):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction

    def forward(self, x, target):
        # Ensure x is 2D: [batch_size, num_classes]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten to 2D
            # If still too many dimensions, take mean over spatial dimensions
            if x.dim() > 2:
                x = x.mean(dim=[2, 3]) if x.dim() == 4 else x.mean(dim=2)

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SwinClassifier(nn.Module):
    """Wrapper class for Swin Transformer with proper classification head"""

    def __init__(
        self, model_name, num_classes, pretrained=True, dropout=0.1, img_size=224
    ):
        super(SwinClassifier, self).__init__()

        # Create base model without classifier
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # No classifier
            drop_rate=dropout,
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone(dummy_input)
            if isinstance(features, (list, tuple)):
                feature_dim = features[0].shape[1]
            else:
                # Handle 4D output by global averaging
                if features.dim() == 4:
                    feature_dim = features.shape[1]
                else:
                    feature_dim = features.shape[-1]

        # Global average pooling for 4D features
        self.global_pool = (
            nn.AdaptiveAvgPool2d((1, 1)) if features.dim() == 4 else nn.Identity()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        print(
            f"Model: {model_name}, Feature dim: {feature_dim}, Num classes: {num_classes}"
        )

    def forward(self, x):
        features = self.backbone(x)

        # Handle different feature shapes
        if features.dim() == 4:  # [batch, channels, height, width]
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)
        elif features.dim() == 3:  # [batch, seq_len, features]
            features = features.mean(dim=1)  # Average over sequence

        return self.classifier(features)


def get_swin_model_names():
    """Get available Swin Transformer model names in current timm version"""
    all_models = timm.list_models("*swin*")
    print("Available Swin models in your timm version:")
    for model in all_models:
        print(f"  - {model}")
    return all_models


def setup_data_transforms(img_size=224):
    """Setup data transforms with augmentation for training"""
    # Training transforms with augmentation
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def create_data_loaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    """Create data loaders for train, validation, and test sets"""
    train_transform, val_transform = setup_data_transforms(img_size)

    # Create datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes


def setup_model(num_classes, pretrained=True, dropout=0.1, img_size=224):
    """Setup Swin Transformer model with proper classification head"""
    # Try different Swin model names
    swin_model_names = [
        "swin_base_patch4_window7_224",  # Most common Swin-B base model
        "swin_small_patch4_window7_224",
        "swin_tiny_patch4_window7_224",
        "swin_v2_base_patch4_window8_256",
        "swin_v2_small_patch4_window8_256",
        "swin_v2_tiny_patch4_window8_256",
    ]

    model = None
    used_model_name = None

    for model_name in swin_model_names:
        try:
            print(f"Trying model: {model_name}")
            model = SwinClassifier(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                img_size=img_size,
            )
            used_model_name = model_name
            print(f"Successfully loaded model: {model_name}")
            break
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

    if model is None:
        # If no Swin model works, show available models and use a fallback
        print("Could not load any Swin model. Available models:")
        get_swin_model_names()
        raise RuntimeError(
            "No suitable Swin model found. Please check your timm version."
        )

    return model, used_model_name


def train_epoch(
    model, train_loader, criterion, optimizer, device, scaler, accumulation_steps=1
):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Ensure labels are long type and have correct shape
        labels = labels.long()

        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)

            # Debug: Print shapes to verify dimensions
            if i == 0:  # Only print first batch
                print(f"Outputs shape: {outputs.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Labels dtype: {labels.dtype}")

            loss = criterion(outputs, labels) / accumulation_steps

        # Backward pass with gradient accumulation
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # Statistics
        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Update progress bar
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            {
                "Loss": f"{running_loss/(i+1):.4f}",
                "Acc": f"{100.*correct_predictions/total_samples:.2f}%",
                "LR": f"{current_lr:.2e}",
            }
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct_predictions / total_samples

    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()  # Ensure labels are long type

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct_predictions / total_samples

    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Swin Transformer for Ophthalmology"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument(
        "--accumulation_steps", type=int, default=2, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Specific model name to use (optional)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--no_label_smoothing", action="store_true", help="Disable label smoothing"
    )
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save command line arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Setup TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        args.data_dir, args.batch_size, args.img_size
    )

    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Save class names
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Setup model
    if args.model_name:
        # Use specific model name if provided
        try:
            model = SwinClassifier(
                model_name=args.model_name,
                num_classes=len(class_names),
                pretrained=True,
                dropout=args.dropout,
                img_size=args.img_size,
            )
            used_model_name = args.model_name
            print(f"Using user-specified model: {used_model_name}")
        except Exception as e:
            print(f"Failed to load user-specified model {args.model_name}: {e}")
            print("Falling back to automatic model selection...")
            model, used_model_name = setup_model(
                len(class_names), img_size=args.img_size, dropout=args.dropout
            )
    else:
        model, used_model_name = setup_model(
            len(class_names), img_size=args.img_size, dropout=args.dropout
        )

    model = model.to(device)

    # Print model summary
    print(f"Model: {used_model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Save model name
    with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
        json.dump({"model_name": used_model_name}, f)

    # Loss function - choose between label smoothing and standard cross entropy
    if args.no_label_smoothing:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
    else:
        try:
            criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
            print("Using LabelSmoothingCrossEntropy with smoothing=0.1")
        except Exception as e:
            print(
                f"LabelSmoothingCrossEntropy failed: {e}. Falling back to CrossEntropyLoss"
            )
            criterion = nn.CrossEntropyLoss()

    # Optimizer with differential learning rates
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Mixed precision scaler (only for CUDA)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Training variables
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
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
            scaler,
            args.accumulation_steps,
        )

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "class_names": class_names,
                    "model_name": used_model_name,
                    "history": history,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )

            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "class_names": class_names,
                    "model_name": used_model_name,
                    "history": history,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

    # Calculate training time
    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)

    # Save final model and history
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "model_name": used_model_name,
            "history": history,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "final_model.pth"),
    )

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    writer.close()

    print(f"\nTraining completed in {hours}h {minutes}m {seconds}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
