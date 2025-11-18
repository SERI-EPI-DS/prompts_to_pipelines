# test.py (Corrected)

import argparse
import os
import json
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def test_model(args):
    """Main function to test the model."""

    # --- 1. Setup and Configuration ---
    print("Starting model evaluation...")
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Class Mapping and Data ---
    class_to_idx_path = os.path.join(
        os.path.dirname(args.model_path), "class_to_idx.json"
    )
    try:
        with open(class_to_idx_path, "r") as f:
            class_to_idx = json.load(f)
        class_names = list(class_to_idx.keys())
        num_classes = len(class_names)
        print(f"Loaded {num_classes} classes: {', '.join(class_names)}")
    except FileNotFoundError:
        print(
            f"Error: 'class_to_idx.json' not found in {os.path.dirname(args.model_path)}"
        )
        return

    # --- 3. Model and Data Transforms ---
    print(f"Loading model architecture: {args.model_name}")
    model = timm.create_model(args.model_name, num_classes=num_classes)
    data_config = resolve_data_config({}, model=model)
    test_transform = create_transform(**data_config, is_training=False)

    print("Data Transforms:")
    print(f"  - Testing: {test_transform}")

    # Load test dataset
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Load the trained model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model weights loaded successfully.")

    # --- 4. Run Inference ---
    all_preds = []
    all_labels = []

    print("\nRunning inference on the test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 5. Generate and Save Reports ---
    # Classification Report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print("\nClassification Report:")
    print(report)
    with open(os.path.join(args.results_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print(
        f"Classification report saved to {args.results_dir}/classification_report.txt"
    )

    # Overall Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")

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
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(args.results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    print("\nEvaluation finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained Swin Transformer.")

    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.pth file).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results.",
    )

    # FIX: Corrected the default model name to match the training script.
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_window12_192.ms_in22k",
        help="Name of the timm model architecture.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for testing."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )

    args = parser.parse_args()
    test_model(args)
