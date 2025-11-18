# /main/project/code/train.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import copy


def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Swin-V2-B classifier for ophthalmology."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the dataset (e.g., ../../data)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save model weights and results (e.g., ../results)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor."
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Size to which images are resized."
    )
    return parser.parse_args()


def main():
    """Main function to run the training and validation process."""
    args = get_args()
    print("Starting training script with the following arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")

    # --- 1. Setup and Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)

    # --- 2. Data Preparation and Augmentation ---
    # State-of-the-art augmentations for fine-tuning.
    # Normalization values are standard for models pre-trained on ImageNet.
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(
                    args.image_size + 32
                ),  # Resize larger then center crop
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    print("Initializing datasets and dataloaders...")
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

    # --- 3. Model Loading and Modification ---
    print("Loading pre-trained Swin-V2-B model...")
    model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)

    # Modify the final classification head for our number of classes
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # --- 4. Loss, Optimizer, and Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine Annealing with Warmup is a very effective schedule for fine-tuning
    # We'll do a simple linear warmup for the first epoch.
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: min(1.0, (epoch + 1) / 1)
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - 1, eta_min=1e-6
    )

    # Mixed precision training setup
    scaler = GradScaler()

    # --- 5. Training Loop ---
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start_time = time.time()

    print("\n--- Starting Training ---")

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

            # Iterate over data.
            for inputs, labels in tqdm(
                dataloaders[phase], desc=f"{phase.capitalize()} Phase"
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)

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

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Adjust learning rate
            if phase == "train":
                if epoch == 0:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model if it's the best one so far
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model_save_path = os.path.join(
                    args.results_dir, "best_model_weights.pth"
                )
                torch.save(model.state_dict(), model_save_path)
                print(
                    f"New best model saved to {model_save_path} with validation accuracy: {best_acc:.4f}"
                )

    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # Load best model weights for final use (optional, but good practice)
    model.load_state_dict(best_model_wts)


if __name__ == "__main__":
    main()
