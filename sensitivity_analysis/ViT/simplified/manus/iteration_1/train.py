"""
Fine-tuning a Swin Transformer V2 model for ophthalmic fundus image classification.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from polyloss import PolyLoss


def main():
    parser = argparse.ArgumentParser(
        description="Train a Swin-V2-B classifier for fundus image classification."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save model checkpoints and logs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        help="Name of the timm model.",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Input image size for the model."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable automatic mixed-precision training.",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Data transformations
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Create datasets and dataloaders
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }
    num_classes = len(image_datasets["train"].classes)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)
    model.to(device)

    # Loss function, optimizer, and scheduler
    criterion = PolyLoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Training loop
    best_acc = 0.0
    best_model_path = None

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

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    with torch.cuda.amp.autocast(enabled=args.use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with validation accuracy: {best_acc:.4f}")

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Save model checkpoint
        # torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))

    print("Training complete.")


if __name__ == "__main__":
    main()
