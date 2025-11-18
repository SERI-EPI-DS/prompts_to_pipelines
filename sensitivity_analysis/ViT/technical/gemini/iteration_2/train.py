# /main/project/code/train.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import time
import copy


def main(args):
    """
    Main function to run the training script.
    """
    print("Initializing training...")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Dataset and Dataloader Preparation ---
    print("Preparing datasets and dataloaders...")

    # Define transforms for training and validation
    # Using augmentations suitable for medical imaging
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    num_classes = len(class_names)

    print(f"Found {num_classes} classes: {', '.join(class_names)}")
    print(f"Training set size: {dataset_sizes['train']}")
    print(f"Validation set size: {dataset_sizes['val']}")

    # --- 2. Model Initialization ---
    print("Loading Swin-V2-B model...")
    model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)

    # Modify the final classification head for our number of classes
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # --- 3. Training Setup ---
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer (AdamW is recommended for Transformers)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.05
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
    )

    # Automatic Mixed Precision (AMP) for memory savings and speed
    scaler = torch.cuda.amp.GradScaler()

    # --- 4. Training Loop ---
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Use tqdm for a progress bar
            progress_bar = tqdm(
                dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}"
            )

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass with mixed precision
                with torch.set_grad_enabled(phase == "train"):
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best validation accuracy: {best_acc:.4f}. Saving model...")
                if not os.path.exists(args.results_dir):
                    os.makedirs(args.results_dir)
                torch.save(
                    model.state_dict(),
                    os.path.join(args.results_dir, "best_model_weights.pth"),
                )

        # Step the scheduler after validation
        if epoch >= args.warmup_epochs:
            scheduler.step()
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print()

    print(f"Best val Acc: {best_acc:4f}")
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Swin-V2-B Fine-Tuning for Ophthalmology Images"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root data directory (containing train/val/test folders)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where results (model weights) will be saved",
    )

    # Hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate for AdamW",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs before scheduler starts",
    )

    args = parser.parse_args()
    main(args)
