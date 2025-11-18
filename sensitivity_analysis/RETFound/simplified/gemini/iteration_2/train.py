import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from timm.data import create_transform
from tqdm import tqdm
import os
import argparse
import json

# Suppress timm warnings about hub
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="timm.models._hub")


def train_model(data_dir, output_dir, model_path, epochs=50, batch_size=32, lr=1e-4):
    """
    Fine-tunes the RETFound model for classification.

    Args:
        data_dir (str): Path to the root data directory (containing train/val folders).
        output_dir (str): Path to the directory where results and models will be saved.
        model_path (str): Path to the pre-trained RETFound model checkpoint.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        lr (float): Learning rate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading and Augmentation ---
    # State-of-the-art augmentations for retinal images
    train_transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {train_dataset.classes}")

    # Save class names for use in the test script
    class_map = {"class_to_idx": train_dataset.class_to_idx}
    with open(os.path.join(output_dir, "class_map.json"), "w") as f:
        json.dump(class_map, f)

    # --- Model Preparation ---
    model = timm.create_model(
        "vit_large_patch16_224", pretrained=False, num_classes=num_classes
    )

    # Load RETFound pre-trained weights
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the head to train the classifier
    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        print(
            "Warning: Model does not have a 'head' attribute. The classifier layer might not be trainable."
        )

    model.to(device)
    print(f"Model loaded. Classifier head is trainable.")

    # --- Training Setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.05
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0

    # --- Training and Validation Loop ---
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        corrects = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)
                val_pbar.set_postfix(
                    acc=f"{(corrects.double() / len(val_loader.dataset)):.4f}"
                )

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{epochs} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"New best model saved with accuracy: {best_val_acc:.4f}")

        scheduler.step()

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RETFound-based classifier.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root data directory (should contain 'train' and 'val' subfolders).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained model and results.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="retfound_mae_v1/checkpoint.pth",
        help="Path to the RETFound pre-trained model checkpoint.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")

    args = parser.parse_args()
    train_model(
        args.data_dir,
        args.output_dir,
        args.model_path,
        args.epochs,
        args.batch_size,
        args.lr,
    )
