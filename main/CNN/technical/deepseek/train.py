import os
import argparse
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from collections import defaultdict


# Label smoothing cross-entropy loss
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingCE, self).__init__()
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def main():
    parser = argparse.ArgumentParser(description="ConvNeXt-L Fine-tuning")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of dataset (should contain train/val folders)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model weights and logs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (default: 0.1)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Epochs to wait before early stopping (default: 7)",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation and normalization
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_root, "train"), train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_root, "val"), val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")

    # Initialize ConvNeXt-L model
    model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)

    # Modify classifier head
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, len(train_dataset.classes))

    # Parallelize if multiple GPUs available (but will use single GPU as per requirements)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Loss function with label smoothing
    criterion = LabelSmoothingCE(smoothing=args.smoothing)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # Mixed precision scaler
    scaler = amp.GradScaler()

    # Training variables
    best_acc = 0.0
    best_epoch = 0
    history = defaultdict(list)

    # Training loop
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc.cpu().item())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad(), amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc.cpu().item())

        # Update learning rate
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_model_wts,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_acc,
                    "class_to_idx": train_dataset.class_to_idx,
                },
                model_path,
            )
            print(f"Model saved at {model_path} with val acc: {best_acc:.4f}")

        # Early stopping check
        if epoch - best_epoch > args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_dataset.class_to_idx,
            "classes": train_dataset.classes,
        },
        final_model_path,
    )
    print(f"Training complete. Best model saved to {final_model_path}")


if __name__ == "__main__":
    main()
