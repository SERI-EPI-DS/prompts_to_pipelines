#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning a Swin-V2-B classifier for fundus image diagnosis.
Fixed to handle proper image sizing for different Swin-V2 models.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import time
import copy


class PolyLoss(nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions.
    """

    def __init__(self, epsilon=2.0):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        poly_loss = ce + self.epsilon * (1 - pt)
        return poly_loss.mean()


def get_model_input_size(model_name):
    """
    Get the expected input size for different Swin-V2 models.
    """
    size_mapping = {
        "swinv2_tiny_window8_256.ms_in1k": 256,
        "swinv2_small_window8_256.ms_in1k": 256,
        "swinv2_base_window8_256.ms_in1k": 256,
        "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k": 256,
        "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k": 256,
        "swinv2_cr_tiny_ns_224.sw_in1k": 224,
        "swinv2_cr_small_ns_224.sw_in1k": 224,
        "swinv2_cr_base_ns_224.sw_in1k": 224,
    }

    # Default to 256 if model not in mapping, but try to infer from name
    if model_name in size_mapping:
        return size_mapping[model_name]
    elif "224" in model_name:
        return 224
    elif "256" in model_name:
        return 256
    elif "384" in model_name:
        return 384
    else:
        return 256  # Default for most Swin-V2 models


def create_data_transforms(input_size):
    """
    Create data transforms based on the model's expected input size.
    """
    # Calculate resize dimension (typically 8/7 of input size for better cropping)
    resize_dim = int(input_size * 8 / 7)

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
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
                transforms.Resize(resize_dim),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    return data_transforms


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device,
    num_epochs=25,
    output_dir=".",
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
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
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(
                    model.state_dict(), os.path.join(output_dir, "best_model.pth")
                )

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    model.load_state_dict(best_model_wts)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a Swin-V2-B classifier for fundus image diagnosis."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for results.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
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
        "--epsilon", type=float, default=2.0, help="Epsilon for PolyLoss."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        help="Swin-V2 model name from timm.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=None,
        help="Input image size. If not specified, will be inferred from model name.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine input size
    if args.input_size is None:
        input_size = get_model_input_size(args.model_name)
    else:
        input_size = args.input_size

    print(f"Using input size: {input_size}x{input_size}")

    # Create data transforms based on input size
    data_transforms = create_data_transforms(input_size)

    print("Loading datasets...")
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

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Training samples: {len(image_datasets['train'])}")
    print(f"Validation samples: {len(image_datasets['val'])}")

    # Create model using timm
    print(f"Creating model: {args.model_name}")
    try:
        # Try to create model with specific input size
        model = timm.create_model(
            args.model_name,
            pretrained=True,
            num_classes=num_classes,
            img_size=input_size,
        )
    except:
        # Fallback to default creation
        model = timm.create_model(
            args.model_name, pretrained=True, num_classes=num_classes
        )

    model = model.to(device)

    # Print model info
    print(f"Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Print expected input size from model
    if hasattr(model, "default_cfg") and "input_size" in model.default_cfg:
        expected_size = model.default_cfg["input_size"]
        print(f"Model expects input size: {expected_size}")

    criterion = PolyLoss(epsilon=args.epsilon)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Starting training...")
    trained_model = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        device,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
    )

    # Save final model
    torch.save(
        trained_model.state_dict(), os.path.join(args.output_dir, "final_model.pth")
    )

    # Save model configuration for testing
    config = {
        "model_name": args.model_name,
        "input_size": input_size,
        "num_classes": num_classes,
        "class_names": class_names,
    }
    torch.save(config, os.path.join(args.output_dir, "model_config.pth"))

    print(f"Training completed. Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
