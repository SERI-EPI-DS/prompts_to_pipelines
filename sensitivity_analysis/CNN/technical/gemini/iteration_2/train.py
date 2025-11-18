# /project/code/train.py

import argparse
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.cuda.amp import GradScaler, autocast

# For reproducibility
torch.manual_seed(42)


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a ConvNext-L classifier for ophthalmic diagnosis."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the dataset (e.g., /main/data)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save model weights and results (e.g., /project/results)",
    )
    return parser.parse_args()


def main():
    """Main function to run the training and validation pipeline."""
    args = get_args()

    # --- 1. Setup and Hyperparameters ---
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not found. Training on CPU, which will be very slow.")
    else:
        print(f"Using device: {device}")

    # Hyperparameters - chosen for a good balance of performance and stability
    # on a 24GB GPU. May need tuning for different image sizes or datasets.
    IMG_SIZE = 224
    BATCH_SIZE = 32  # Adjust based on VRAM and image size
    LEARNING_RATE = 3e-5
    MAX_EPOCHS = 50
    LABEL_SMOOTHING = 0.1
    NUM_WORKERS = 4  # Adjust based on your system's capabilities

    # --- 2. Data Preparation and Augmentation ---
    print("Preparing data...")
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.TrivialAugmentWide(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, data_transforms["train"]),
        "val": datasets.ImageFolder(val_dir, data_transforms["val"]),
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
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

    # Save class names for use in testing script
    class_map_path = os.path.join(args.results_dir, "class_map.json")
    with open(class_map_path, "w") as f:
        json.dump(class_names, f)
    print(f"Class mapping saved to {class_map_path}")

    # --- 3. Model Setup ---
    print("Setting up model...")
    # Load pretrained ConvNext Large model
    model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head for fine-tuning
    # ConvNext's classifier is a Sequential layer with a LayerNorm and a Linear layer
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)

    # Unfreeze the parameters of the final classifier layer
    for param in model.classifier[2].parameters():
        param.requires_grad = True

    model = model.to(device)

    # --- 4. Training Configuration ---
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-7
    )

    # Automatic Mixed Precision (AMP) scaler
    scaler = GradScaler()

    # --- 5. Training Loop ---
    print("Starting training...")
    since = time.time()

    best_model_weights_path = os.path.join(args.results_dir, "best_model.pth")
    best_acc = 0.0

    for epoch in range(MAX_EPOCHS):
        print(f"Epoch {epoch+1}/{MAX_EPOCHS}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    # Use autocast for mixed precision
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                    # Backward pass + optimize only if in training phase
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                scheduler.step()  # Step the scheduler each epoch

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save the best model based on validation accuracy
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_weights_path)
                print(f"New best model saved with validation accuracy: {best_acc:.4f}")

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")
    print(f"Best model weights saved to {best_model_weights_path}")


if __name__ == "__main__":
    main()
