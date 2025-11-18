# /main/project/code/train.py (Final Version with Adjustable Image Size)

import argparse
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def train_model(
    data_dir: str,
    results_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    accumulation_steps: int,
    image_size: int,
):
    """
    Fine-tunes a Swin-V2-B model with robust memory-saving techniques.
    """
    # 1. Setup and Configuration
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    effective_batch_size = batch_size * accumulation_steps
    print(f"Image Resolution: {image_size}x{image_size}")
    print(f"Actual Batch Size: {batch_size}")
    print(f"Gradient Accumulation Steps: {accumulation_steps}")
    print(f"Effective Batch Size: {effective_batch_size}")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    best_model_path = os.path.join(results_dir, "best_model.pth")

    # 2. Data Preparation
    # -------------------
    # --- UPDATED: Transformations now use the image_size argument ---
    image_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(
                    int(image_size / 0.875)
                ),  # Standard practice to resize slightly larger than crop
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, image_transforms["train"]),
        "val": datasets.ImageFolder(val_dir, image_transforms["val"]),
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
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
    print(
        f"Training set size: {dataset_sizes['train']}, Validation set size: {dataset_sizes['val']}"
    )

    # 3. Model Loading and Modification
    # ---------------------------------
    model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)

    # Keeping gradient checkpointing as it's still highly beneficial
    model.gradient_checkpointing = True

    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    # 4. Loss Function, Optimizer, and Scheduler
    # ------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()

    # 5. Training and Validation Loop (No changes here)
    # ... (The rest of the script is identical to the previous version) ...
    since = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)

        model.train()
        # ... (Training loop remains the same) ...
        # --- Training Phase ---
        running_loss = 0.0
        running_corrects = 0
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(
            tqdm(dataloaders["train"], desc="Training Phase")
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0) * accumulation_steps
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_loss = running_loss / dataset_sizes["train"]
        epoch_acc = running_corrects.double() / dataset_sizes["train"]
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Validation Phase ---
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloaders["val"], desc="Validation Phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes["val"]
        epoch_acc = running_corrects.double() / dataset_sizes["val"]
        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_acc:.4f}")

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Swin-V2-B with robust memory-saving techniques."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the root data directory."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory to save results.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Actual batch size to fit in memory."
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3, help="Weight decay (L2 penalty)."
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor."
    )
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--image_size", type=int, default=192, help="Input image resolution."
    )

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        accumulation_steps=args.accumulation_steps,
        image_size=args.image_size,
    )
