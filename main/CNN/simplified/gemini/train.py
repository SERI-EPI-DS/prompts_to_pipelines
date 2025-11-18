import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, models, transforms
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
import os
import argparse
import numpy as np
from tqdm import tqdm


def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a ConvNext-L classifier for ophthalmic diagnosis."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root data directory (containing train/val/test folders).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where results and models will be saved.",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate."
    )
    return parser.parse_args()


def main():
    """Main function to train the classifier."""
    args = get_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define data transformations
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
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
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Load datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained ConvNext-L model
    model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)

    # Replace the classifier head
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.1, verbose=True
    )

    # Training loop
    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # Initialize metrics for the epoch
            accuracy_metric = MulticlassAccuracy(device=device)

            progress_bar = tqdm(
                dataloaders[phase],
                desc=f"{phase.capitalize()} Epoch {epoch+1}/{args.epochs}",
            )

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                accuracy_metric.update(outputs, labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = accuracy_metric.compute().item()

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.output_dir, "best_model.pth"),
                    )
                    print("Best model saved.")

    print(f"Best val Acc: {best_acc:4f}")


if __name__ == "__main__":
    main()
