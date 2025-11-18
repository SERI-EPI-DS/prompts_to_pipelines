import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.models import convnext_large
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler
import os
import argparse
import numpy as np
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Argument parser
parser = argparse.ArgumentParser(description="ConvNext-L Training")
parser.add_argument(
    "--train_dir", type=str, required=True, help="Path to training dataset"
)
parser.add_argument(
    "--val_dir", type=str, required=True, help="Path to validation dataset"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./output",
    help="Output directory for models and logs",
)
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

# Setup directories and device
os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)

# Data transformations with augmentation
# Using torchvision's RandAugment implementation for compatibility
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandAugment(
            num_ops=2, magnitude=9
        ),  # Using torchvision's implementation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create datasets
train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

# Initialize model
model = convnext_large(pretrained=True, num_classes=len(train_dataset.classes))
model = model.to(device)

# Loss function with label smoothing
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
num_steps = args.epochs * len(train_loader)
lr_scheduler = CosineLRScheduler(
    optimizer,
    t_initial=num_steps,
    warmup_t=500,
    warmup_lr_init=1e-6,
    warmup_prefix=True,
    lr_min=1e-6,
)

# Training loop
best_val_acc = 0.0
for epoch in range(args.epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update learning rate scheduler
        global_step = epoch * len(train_loader) + i
        lr_scheduler.step_update(global_step)

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)

    # Validation phase
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_dataset)
    val_acc = 100 * correct / total
    print(
        f"Epoch [{epoch+1}/{args.epochs}] | "
        f"Train Loss: {epoch_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint = {
            "model": model.state_dict(),
            "classes": train_dataset.classes,
            "val_acc": val_acc,
            "epoch": epoch + 1,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pth"))
        print(f"New best model saved with accuracy: {val_acc:.2f}%")

    # Save last model
    torch.save(
        {
            "model": model.state_dict(),
            "classes": train_dataset.classes,
            "epoch": epoch + 1,
            "val_acc": val_acc,
        },
        os.path.join(args.output_dir, "last_model.pth"),
    )

print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
