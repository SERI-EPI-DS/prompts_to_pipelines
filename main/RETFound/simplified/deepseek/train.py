import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Configuration
parser = argparse.ArgumentParser(description="Fine-tune RETFound for classification")
parser.add_argument(
    "--data_dir", type=str, required=True, help="Root dataset directory"
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Output directory for models/logs"
)
parser.add_argument("--batch_size", type=int, default=32, help="Input batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers")
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Data transformations
train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ]
)

# Dataset preparation
train_dataset = datasets.ImageFolder(
    root=os.path.join(args.data_dir, "train"), transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(args.data_dir, "val"), transform=val_transform
)

# Data loaders
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

# Initialize model - Using ViT-Large to match RETFound
model = timm.create_model(
    "vit_large_patch16_224", pretrained=False, num_classes=len(train_dataset.classes)
)

# Load RETFound weights
retfound_path = "../../../RETFound/RETFound_CFP_weights.pth"  # Update this path
state_dict = torch.load(retfound_path, map_location="cpu")["model"]

# Remove decoder weights (not needed for classification)
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder")}

# Adapt RETFound weights to timm model
msg = model.load_state_dict(state_dict, strict=False)
print(f"Loaded weights with message: {msg}")

# Replace classification head
model.reset_classifier(len(train_dataset.classes))

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6
)

# Training loop
best_acc = 0.0
for epoch in range(args.epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * images.size(0)

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Calculate metrics
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = 100.0 * correct / total

    print(
        f"Epoch {epoch+1}/{args.epochs}: "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": train_dataset.class_to_idx,
            },
            os.path.join(args.output_dir, "best_model.pth"),
        )

print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
