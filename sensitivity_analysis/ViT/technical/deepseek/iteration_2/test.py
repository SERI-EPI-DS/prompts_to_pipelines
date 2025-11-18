import os
import argparse
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import timm
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import json


def get_args():
    parser = argparse.ArgumentParser(
        description="Test Swin-V2 model on ophthalmology dataset"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root directory of test dataset"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model weights"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Output CSV file for results"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for testing"
    )
    parser.add_argument(
        "--img_size", type=int, default=256, help="Input image size"
    )  # Changed to 256
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    return parser.parse_args()


def create_test_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(model_path, device):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location="cpu")

    # Get model configuration from checkpoint
    model_name = checkpoint.get("model_name", "swinv2_tiny_window8_256")
    num_classes = (
        len(checkpoint["classes"])
        if "classes" in checkpoint
        else checkpoint.get("num_classes", 2)
    )
    img_size = checkpoint.get("img_size", 256)

    print(f"Loading model: {model_name}")
    print(f"Number of classes: {num_classes}")
    print(f"Trained image size: {img_size}")

    try:
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
        )
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print("Falling back to swinv2_tiny_window8_256")
        model = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=False,
            num_classes=num_classes,
        )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, checkpoint


def test_model(model, test_loader, device, class_names):
    """Run model on test set and return predictions"""
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

            # Get filenames for current batch
            start_idx = batch_idx * test_loader.batch_size
            batch_filenames = []
            for i in range(len(images)):
                file_idx = start_idx + i
                if file_idx < len(test_loader.dataset):
                    filepath, _ = test_loader.dataset.samples[file_idx]
                    batch_filenames.append(os.path.basename(filepath))

            all_filenames.extend(batch_filenames)

    return all_filenames, all_probabilities, all_predictions, all_labels


def save_results(
    output_csv, filenames, probabilities, predictions, true_labels, class_names
):
    """Save results to CSV file"""
    # Create results dataframe
    results_data = {
        "filename": filenames,
        "true_label": [class_names[label] for label in true_labels],
        "predicted_label": [class_names[pred] for pred in predictions],
        "correct": [true_labels[i] == predictions[i] for i in range(len(true_labels))],
    }

    # Add probability scores for each class
    for i, class_name in enumerate(class_names):
        results_data[f"score_{class_name}"] = [prob[i] for prob in probabilities]

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_csv, index=False)

    return results_df


def calculate_metrics(true_labels, predictions, class_names, output_dir):
    """Calculate and display evaluation metrics"""
    accuracy = accuracy_score(true_labels, predictions)
    balanced_accuracy = balanced_accuracy_score(true_labels, predictions)

    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    report = classification_report(
        true_labels, predictions, target_names=class_names, digits=4, output_dict=True
    )
    print(
        classification_report(
            true_labels, predictions, target_names=class_names, digits=4
        )
    )

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Save metrics to file
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return accuracy, balanced_accuracy, cm


def main():
    args = get_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory for results
    output_dir = os.path.dirname(args.output_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Load model and checkpoint
    model, checkpoint = load_model(args.model_path, device)

    # Get class information from checkpoint
    if "classes" in checkpoint:
        class_names = checkpoint["classes"]
        class_to_idx = checkpoint["class_to_idx"]
        print(f"Loaded classes from checkpoint: {class_names}")
    else:
        # Fallback: infer from test dataset
        test_dataset = ImageFolder(
            root=args.data_root, transform=create_test_transform(args.img_size)
        )
        class_names = test_dataset.classes
        class_to_idx = test_dataset.class_to_idx
        print("Warning: Using classes from test dataset instead of checkpoint")

    # Create test dataset and dataloader
    test_transform = create_test_transform(args.img_size)
    test_dataset = ImageFolder(root=args.data_root, transform=test_transform)

    # Update dataset classes to match checkpoint if there's a mismatch
    if hasattr(test_dataset, "classes") and class_names != test_dataset.classes:
        print("Warning: Class mismatch between model and test dataset!")
        print(f"Model classes: {class_names}")
        print(f"Test dataset classes: {test_dataset.classes}")
        # We'll use the model's class names for output, but need to handle mapping
        # For now, we'll proceed but this might cause issues

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    # Run testing
    filenames, probabilities, predictions, true_labels = test_model(
        model, test_loader, device, class_names
    )

    # Save results
    results_df = save_results(
        args.output_csv, filenames, probabilities, predictions, true_labels, class_names
    )

    # Calculate and display metrics
    accuracy, balanced_accuracy, cm = calculate_metrics(
        true_labels, predictions, class_names, output_dir
    )

    # Print some example results
    print(f"\nFirst 10 predictions:")
    print(results_df[["filename", "true_label", "predicted_label", "correct"]].head(10))

    print(f"\nOverall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Balanced accuracy: {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%)")
    print(f"Results saved to: {args.output_csv}")
    print(f"Metrics saved to: {output_dir}")


if __name__ == "__main__":
    args = get_args()
    main()
