# train.py (Updated)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
import argparse
import json
from tqdm import tqdm


def train_model(data_dir, output_dir, epochs=25, batch_size=32, learning_rate=1e-4):
    """
    Fine-tunes a Swin-V2-B model on a given dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Data Preparation and Augmentation ---
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
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
                transforms.Resize(288),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
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

    with open(os.path.join(output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # --- 2. Model Initialization ---
    # **FIX:** Updated model name to one available in your timm version.
    model_name = "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"

    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"Error creating model: {e}")
        print(f"The model name '{model_name}' was not found.")
        print("Please choose a model from the list of available Swin Transformers:")
        available_models = timm.list_models("*swin*", pretrained=True)
        for name in available_models:
            print(name)
        return

    model.to(device)

    # --- 3. Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # --- 4. Training Loop ---
    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}\n" + "-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(
                dataloaders[phase],
                desc=f"{phase.capitalize()} Epoch {epoch+1}/{epochs}",
            )

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{torch.sum(preds == labels.data).item() / inputs.size(0):.4f}",
                )

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                model_path = os.path.join(output_dir, "best_model.pth")
                torch.save(model.state_dict(), model_path)
                print(
                    f"New best model saved to {model_path} with accuracy: {best_val_acc:.4f}"
                )

        if phase == "train":
            scheduler.step()

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Swin-V2-B classifier.")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model and results.",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )

    args = parser.parse_args()

    train_model(
        args.data_dir, args.output_dir, args.epochs, args.batch_size, args.learning_rate
    )
