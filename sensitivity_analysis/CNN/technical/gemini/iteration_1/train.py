import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(args):
    """
    Main function to orchestrate the fine-tuning of the ConvNext-L model.
    """
    # --- 1. Device Configuration ---
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. This script requires a GPU.")
        return
    device = torch.device("cuda")
    logging.info(f"Using device: {device}")

    # --- 2. Data Preparation and Augmentation ---
    # Using advanced augmentation (TrivialAugmentWide) for training,
    # and standard resizing/normalization for validation.
    # Image size is 384x384, optimal for larger ConvNext models.
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    # --- 3. Dataset and DataLoader Setup ---
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        logging.error(
            "Training or validation directory not found. Please check the data_dir path."
        )
        return

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, data_transforms["train"]),
        "val": datasets.ImageFolder(val_dir, data_transforms["val"]),
    }

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        ),
    }

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)
    logging.info(f"Detected {num_classes} classes: {class_names}")

    # --- 4. Model Loading and Modification ---
    # Load a pre-trained ConvNext-L model.
    model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)

    # Replace the final classifier layer to match the number of classes in your dataset.
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # --- 5. Loss, Optimizer, and Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # Cosine annealing scheduler for better convergence.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    # Gradient scaler for mixed-precision training.
    scaler = GradScaler()

    # --- 6. Training and Validation Loop ---
    best_val_acc = 0.0
    os.makedirs(args.results_dir, exist_ok=True)

    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        progress_bar = tqdm(dataloaders["train"], desc="Training")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Use automatic mixed precision for faster training.
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        epoch_loss = running_loss / len(image_datasets["train"])
        epoch_acc = running_corrects.double() / len(image_datasets["train"])
        logging.info(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        progress_bar_val = tqdm(dataloaders["val"], desc="Validation")
        with torch.no_grad():
            for inputs, labels in progress_bar_val:
                inputs, labels = inputs.to(device), labels.to(device)

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(image_datasets["val"])
        val_epoch_acc = val_running_corrects.double() / len(image_datasets["val"])
        logging.info(f"Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # --- 7. Model Checkpointing ---
        # Save the model if it achieves a new best validation accuracy.
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_model_path = os.path.join(args.results_dir, "best_model_weights.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(
                f"🎉 New best model saved to {best_model_path} with accuracy: {best_val_acc:.4f}"
            )

    # Save the final model at the end of training.
    final_model_path = os.path.join(args.results_dir, "final_model_weights.pth")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to fine-tune a ConvNext-L model for ophthalmic diagnosis."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root data directory (containing train, val, test folders).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where model weights and results will be saved.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation. Default is 16 for ~24GB VRAM.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate for the AdamW optimizer."
    )

    args = parser.parse_args()
    main(args)
