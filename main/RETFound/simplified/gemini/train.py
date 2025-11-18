import sys
import os

# --- NEW: Self-Aware Path Configuration ---
# This code block makes the script runnable from anywhere by ensuring
# the project root is in Python's search path.
try:
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the path of the parent directory (the project root)
    project_root = os.path.dirname(script_dir)
    # Add the project root to the Python path
    sys.path.insert(0, project_root)
    print(f"Project root added to path: {project_root}")
except Exception as e:
    print(f"Error adjusting Python path: {e}")
    # Fallback for interactive environments
    project_root = os.path.abspath(".")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Fallback: Added current directory to path: {project_root}")
# --- END NEW BLOCK ---

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.models.vision_transformer import VisionTransformer
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

# This import should now work reliably
from models_vit import vit_large_patch16


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="RETFound Fine-tuning for Image Classification"
    )
    parser.add_argument(
        "--data_path", required=True, type=str, help="Path to the root of your dataset"
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        type=str,
        help="Path to save logs and checkpoints",
    )
    parser.add_argument(
        "--model_name",
        default="retfound_large_patch16",
        type=str,
        help="Name of the model to use",
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", default=50, type=int, help="Number of training epochs"
    )
    parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay")
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of data loading workers"
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed for reproducibility"
    )
    return parser


def main(args):
    # --- Setup ---
    print("Starting RETFound Fine-tuning...")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Preparation ---
    print(f"Loading data from: {args.data_path}")
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), transform=transform_train
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "val"), transform=transform_val
    )

    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {train_dataset.classes}")

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

    # --- Model Loading ---
    print(f"Loading model: {args.model_name}")
    model = vit_large_patch16(
        num_classes=num_classes, drop_path_rate=0.1, global_pool=True
    )

    # Load pre-trained RETFound weights
    checkpoint = torch.hub.load_state_dict_from_url(
        url="https://huggingface.co/rmaphoh/RETFound_MAE/resolve/main/retfound_large_patch16.pth",
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)

    # --- Training Setup ---
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_scaler = torch.cuda.amp.GradScaler()
    criterion = SoftTargetCrossEntropy()
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        label_smoothing=0.1,
        num_classes=num_classes,
    )

    # --- Training Loop ---
    print("Starting training...")
    best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            samples, targets = mixup_fn(images, labels)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)

            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{args.epochs}], Training Loss: {running_loss/len(train_loader):.4f}"
        )

        # --- Validation Loop ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch+1}/{args.epochs}], Validation Accuracy: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )
            print("Saved new best model.")

    print("Training finished.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
