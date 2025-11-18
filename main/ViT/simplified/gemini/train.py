import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Swin-V2-B Fine-Tuning for Ophthalmic Diagnosis", add_help=False
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Path to the root directory of your dataset (containing train, val, test folders)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Path to save the trained model and results",
    )
    # --- CORRECTED MODEL NAME ---
    parser.add_argument(
        "--model_name",
        default="swinv2_base_window12_192.ms_in22k",
        type=str,
        help="Name of the Swin-V2 model from timm",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--epochs", default=25, type=int, help="Number of training epochs"
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        default=1e-5,
        type=float,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of data loading workers"
    )
    return parser


def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Using model: {args.model_name}")

    # --- Data Loading and Transformations ---
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    # Get model-specific transforms
    # This line no longer causes the error with the corrected model name
    model_for_transforms = timm.create_model(args.model_name, pretrained=True)
    data_config = timm.data.resolve_model_data_config(model_for_transforms)
    data_transform = timm.data.create_transform(**data_config, is_training=True)
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    train_dataset = ImageFolder(train_dir, transform=data_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

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

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f"Found {num_classes} classes: {class_names}")

    # --- Model Definition ---
    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)
    model.to(device)

    # --- Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # --- Training Loop ---
    best_val_accuracy = 0.0
    training_stats = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # --- Validation Loop ---
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="weighted", zero_division=0
        )

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {epoch_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val F1: {f1:.4f}"
        )

        training_stats.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "val_accuracy": val_accuracy,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
            }
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, "best_model.pth")
            )
            print("Saved new best model.")

    # --- Save Training Log ---
    df_stats = pd.DataFrame(training_stats)
    df_stats.to_csv(os.path.join(args.output_dir, "training_log.csv"), index=False)
    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Find the best model at: {os.path.join(args.output_dir, 'best_model.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Swin-V2-B Fine-Tuning", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
