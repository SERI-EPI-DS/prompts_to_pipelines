import os
import argparse
import time
import copy
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms
from timm.data.auto_augment import rand_augment_transform
from timm.loss import LabelSmoothingCrossEntropy

# Argument parser
parser = argparse.ArgumentParser(description="ConvNext-L Fine-tuning")
parser.add_argument(
    "--data_root", type=str, default="../data", help="Root directory for dataset"
)
parser.add_argument(
    "--results_dir", type=str, default="../results", help="Directory to save results"
)
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size for training"
)
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
parser.add_argument(
    "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Setup directories
train_dir = os.path.join(args.data_root, "train")
val_dir = os.path.join(args.data_root, "val")
os.makedirs(args.results_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Image transforms
def get_transforms(img_size=224):
    # RandAugment parameters
    ra_config = {
        "translate_const": 100,
        "img_mean": tuple([min(255, round(255 * x)) for x in [0.485, 0.456, 0.406]]),
    }

    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform("rand-m9-mstd0.5", ra_config),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


# Create datasets
train_transform, val_transform = get_transforms(224)
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True
)

# Model initialization
model = torchvision.models.convnext_large(
    weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
)
num_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, len(train_dataset.classes))
model = model.to(device)

# Loss function with label smoothing
criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)


# Training function
def train_model():
    best_acc = 0.0
    best_model = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        val_loss, val_acc = validate()
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, os.path.join(args.results_dir, "best_model.pth"))

    return best_model


# Validation function
def validate():
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = running_corrects.double() / len(val_dataset)
    return epoch_loss, epoch_acc


# Execute training
if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")

    start_time = time.time()
    best_model = train_model()
    training_time = time.time() - start_time

    print(f"Training completed in {training_time//60:.0f}m {training_time%60:.0f}s")
    print(f'Best model saved to {os.path.join(args.results_dir, "best_model.pth")}')
