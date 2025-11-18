import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import time
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        n_classes = output.size(-1)
        log_preds = nn.functional.log_softmax(output, dim=-1)
        loss = -log_preds.sum(dim=-1)
        nll = nn.functional.nll_loss(log_preds, target, reduction="none")
        smooth_loss = loss / n_classes
        eps_i = self.epsilon / n_classes
        loss = (1.0 - self.epsilon) * nll + eps_i * smooth_loss.sum(dim=-1)
        return loss.mean()


class FundusDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.is_training = is_training

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_transforms(is_training=True, input_size=384):
    """
    State-of-the-art augmentation pipeline for fundus images
    """
    if is_training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1
                ),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
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


def create_model(num_classes, pretrained=True):
    """
    Create Swin-V2-B model with custom head for fundus image classification
    """
    # Load pretrained Swin Transformer V2 Base
    model = models.swin_v2_b(weights="IMAGENET1K_V1" if pretrained else None)

    # Get the number of features in the last layer
    num_features = model.head.in_features

    # Replace the head with a more sophisticated classifier
    model.head = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )

    return model


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"}
        )

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
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

            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"}
            )

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(
        description="Train Swin-V2-B for Fundus Image Classification"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to data root folder"
    )
    parser.add_argument(
        "--results_folder", type=str, required=True, help="Path to results folder"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create results directory
    os.makedirs(args.results_folder, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    train_transform = get_transforms(is_training=True, input_size=args.input_size)
    val_transform = get_transforms(is_training=False, input_size=args.input_size)

    train_dataset = FundusDataset(
        train_dir, transform=train_transform, is_training=True
    )
    val_dataset = FundusDataset(val_dir, transform=val_transform, is_training=False)

    # Get number of classes
    num_classes = len(train_dataset.dataset.classes)
    class_names = train_dataset.dataset.classes
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

    # Save class names
    with open(os.path.join(args.results_folder, "class_names.json"), "w") as f:
        json.dump(class_names, f)

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

    # Create model
    model = create_model(num_classes, pretrained=True)
    model = model.to(device)

    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(epsilon=args.label_smoothing)

    # Optimizer - AdamW with different learning rates for backbone and head
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},  # Lower lr for backbone
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # Mixed precision training
    scaler = GradScaler()

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Training loop
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(50):  # Maximum 50 epochs as specified
        print(f"\nEpoch {epoch+1}/50")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(args.results_folder, "best_model.pth")
            )
            print(f"Best model saved with validation accuracy: {val_acc:.2f}%")

        # Early stopping
        early_stopping(
            val_loss, model, os.path.join(args.results_folder, "checkpoint.pth")
        )

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.results_folder, "final_model.pth"))

    # Save training history
    with open(os.path.join(args.results_folder, "training_history.json"), "w") as f:
        json.dump(history, f)

    # Save training config
    config = vars(args)
    config["num_classes"] = num_classes
    config["best_val_acc"] = best_val_acc
    config["training_time"] = time.time() - start_time
    config["final_epoch"] = epoch + 1

    with open(os.path.join(args.results_folder, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
