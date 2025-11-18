#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test a fine-tuned Swin-V2-B classifier for fundus image diagnosis.
Fixed to handle proper image sizing for different Swin-V2 models.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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


def create_test_transform(input_size):
    """
    Create test data transform based on the model's expected input size.
    """
    # Calculate resize dimension (typically 8/7 of input size for better cropping)
    resize_dim = int(input_size * 8 / 7)

    return transforms.Compose(
        [
            transforms.Resize(resize_dim),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def test_model(model, dataloader, device, output_dir, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Classification Report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Save classification report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(
        f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}"
    )

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(
                np.array(all_preds)[class_mask] == np.array(all_labels)[class_mask]
            )
            print(f"{class_name}: {class_acc:.4f}")

    return accuracy, report


def main():
    parser = argparse.ArgumentParser(description="Test a Swin-V2-B classifier.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the test dataset directory.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for results.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Swin-V2 model name from timm. If not provided, will try to load from config.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=None,
        help="Input image size. If not specified, will be inferred from model or config.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to model config file (model_config.pth). If not provided, will look in same dir as model.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try to load configuration from training
    config_path = args.config_path
    if config_path is None:
        # Try to find config in same directory as model
        model_dir = os.path.dirname(args.model_path)
        config_path = os.path.join(model_dir, "model_config.pth")

    config = None
    if os.path.exists(config_path):
        try:
            config = torch.load(config_path, map_location="cpu")
            print(f"Loaded configuration from {config_path}")
        except:
            print(f"Could not load configuration from {config_path}")

    # Determine model name and input size
    if args.model_name is not None:
        model_name = args.model_name
    elif config is not None and "model_name" in config:
        model_name = config["model_name"]
    else:
        model_name = "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"  # Default
        print(f"Using default model name: {model_name}")

    if args.input_size is not None:
        input_size = args.input_size
    elif config is not None and "input_size" in config:
        input_size = config["input_size"]
    else:
        input_size = get_model_input_size(model_name)

    print(f"Using model: {model_name}")
    print(f"Using input size: {input_size}x{input_size}")

    # Create data transform
    data_transform = create_test_transform(input_size)

    print("Loading test dataset...")
    image_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), data_transform
    )
    dataloader = DataLoader(
        image_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Get class names from config or dataset
    if config is not None and "class_names" in config:
        class_names = config["class_names"]
        num_classes = config["num_classes"]
        print("Using class names from training configuration")
    else:
        class_names = image_dataset.classes
        num_classes = len(class_names)
        print("Using class names from test dataset")

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Test samples: {len(image_dataset)}")

    # Create model using timm
    print(f"Creating model: {model_name}")
    try:
        # Try to create model with specific input size
        model = timm.create_model(
            model_name, pretrained=False, num_classes=num_classes, img_size=input_size
        )
    except:
        # Fallback to default creation
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    # Load trained weights
    print(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    print("Starting evaluation...")
    accuracy, report = test_model(
        model, dataloader, device, args.output_dir, class_names
    )

    print(f"\nEvaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
