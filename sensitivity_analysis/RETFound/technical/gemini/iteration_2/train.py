import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import sys
import argparse
from tqdm import tqdm
import time
from functools import partial

# Add the RETFound repository to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RETFound"))
)
# --- CORRECTED IMPORT: Use the base class for robustness ---
from models_vit import VisionTransformer


def get_args_parser():
    parser = argparse.ArgumentParser(description="RETFound Fine-tuning Training")
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to the root data directory (containing train, val, test folders)",
    )
    parser.add_argument(
        "--results_path",
        required=True,
        type=str,
        help="Path to the directory where results (model weights) will be saved",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="../RETFound/RETFound_CFP_weights.pth",
        help="Path to the pre-trained RETFound weights",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing for CrossEntropyLoss",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    return parser


def main(args):
    # --- 1. Setup and Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.results_path, exist_ok=True)

    # --- 2. Data Preparation and Augmentation ---
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
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

    # --- 3. Model Preparation ---
    # --- CORRECTED MODEL INSTANTIATION: Directly build the ViT-Large model ---
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        drop_path_rate=0.1,  # Adjust dropout as needed
    )

    checkpoint = torch.load(args.weights_path, map_location="cpu")
    print("Loading pre-trained weights from:", args.weights_path)

    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from loaded checkpoint due to size mismatch.")
            del checkpoint_model[k]

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"State dict loading message: {msg}")

    model.to(device)

    # --- 4. Training Setup ---
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_epoch = 0

    # --- 5. Training and Validation Loop ---
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_pbar.set_postfix({"loss": loss.item()})

        train_loss = running_loss / len(train_dataset)
        train_acc = correct_train / total_train

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_pbar.set_postfix({"loss": loss.item()})

        val_loss = running_val_loss / len(val_dataset)
        val_acc = correct_val / total_val

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_path = os.path.join(args.results_path, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(
                f"New best model saved to {save_path} with validation accuracy: {best_val_acc:.4f}"
            )

        lr_scheduler.step()

    end_time = time.time()
    total_time = end_time - start_time
    print("\n--- Training Finished ---")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
