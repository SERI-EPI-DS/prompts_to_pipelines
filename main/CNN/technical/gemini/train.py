# /main/project/code/train.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time


def main(args):
    """
    Main function to execute the training and validation process.
    """
    print("PyTorch Version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print(
        "Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Data Preparation and Augmentation ---
    # Define transformations for training and validation sets
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Create datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }

    # Create dataloaders
    dataloaders = {
        x: DataLoader(
            image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {', '.join(class_names)}")
    print(
        f"Training set size: {dataset_sizes['train']}, Validation set size: {dataset_sizes['val']}"
    )

    # --- 2. Model Loading and Modification ---
    # Load pretrained ConvNext-L model
    model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)

    # Modify the final classifier layer for our number of classes
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    # --- 3. Training Setup ---
    # Loss function with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 1)

    # Automatic Mixed Precision Scaler
    scaler = GradScaler()

    # --- 4. Training Loop ---
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            progress_bar = tqdm(
                dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}"
            )
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass with Automatic Mixed Precision
                with torch.set_grad_enabled(phase == "train"):
                    with autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_postfix(
                    loss=loss.item(),
                    acc=torch.sum(preds == labels.data).item() / inputs.size(0),
                )

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model if it's the best so far
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        # Update the learning rate
        if phase == "train":
            scheduler.step()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # --- 5. Save the best model weights ---
    os.makedirs(args.results_dir, exist_ok=True)
    model_save_path = os.path.join(args.results_dir, "best_model_weights.pth")
    torch.save(best_model_wts, model_save_path)
    print(f"Best model weights saved to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a ConvNext-L classifier on fundus images."
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
        help="Path to the directory where results (model weights) will be saved.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation.",
    )

    args = parser.parse_args()
    main(args)
