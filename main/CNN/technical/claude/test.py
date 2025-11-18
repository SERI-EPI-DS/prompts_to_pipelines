import os
import argparse
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class CustomImageFolder(ImageFolder):
    """Custom ImageFolder to return image paths along with images and labels."""

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


def create_model(num_classes):
    """Create ConvNext-L model."""
    model = models.convnext_large(weights=None)
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)
    return model


def get_test_transforms(input_size=384):
    """Get test transforms."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform


def test_model(model, dataloader, device, num_classes):
    """Test the model and collect predictions."""
    model.eval()

    all_predictions = []
    all_scores = []
    all_targets = []
    all_paths = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        for inputs, targets, paths in pbar:
            inputs = inputs.to(device)

            # Get predictions
            outputs = model(inputs)
            scores = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # Store results
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_paths.extend(paths)

    return all_predictions, all_scores, all_targets, all_paths


def calculate_metrics(predictions, targets, num_classes):
    """Calculate classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average="weighted", zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(targets, predictions)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "support_per_class": support_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Test ConvNext-L for ophthalmology image classification"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to data root folder"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to results folder"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class mapping
    class_mapping_path = os.path.join(args.results_dir, "class_mapping.json")
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)
    idx_to_class = {int(k): v for k, v in class_mapping["idx_to_class"].items()}
    num_classes = len(idx_to_class)

    # Data path
    test_dir = os.path.join(args.data_root, "test")

    # Create dataset
    test_transform = get_test_transforms(input_size=args.input_size)
    test_dataset = CustomImageFolder(test_dir, transform=test_transform)

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model and load weights
    model = create_model(num_classes)

    # Load checkpoint
    checkpoint_path = os.path.join(args.results_dir, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(
        f'Loaded model from epoch {checkpoint["epoch"]} with validation accuracy: {checkpoint["val_acc"]:.2f}%'
    )

    # Test the model
    predictions, scores, targets, paths = test_model(
        model, test_loader, device, num_classes
    )

    # Calculate metrics
    metrics = calculate_metrics(predictions, targets, num_classes)

    # Save results to CSV
    csv_path = os.path.join(args.results_dir, "test_results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        # Define field names
        fieldnames = ["filename", "true_class", "predicted_class"] + [
            f"score_class_{i}" for i in range(num_classes)
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write each test sample
        for i in range(len(paths)):
            row = {
                "filename": os.path.basename(paths[i]),
                "true_class": idx_to_class[targets[i]],
                "predicted_class": idx_to_class[predictions[i]],
            }

            # Add scores for each class
            for j in range(num_classes):
                row[f"score_class_{j}"] = f"{scores[i][j]:.6f}"

            writer.writerow(row)

    # Save metrics
    metrics_path = os.path.join(args.results_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"\nTest Results:")
    print(f'Accuracy: {metrics["accuracy"]*100:.2f}%')
    print(f'Precision: {metrics["precision"]*100:.2f}%')
    print(f'Recall: {metrics["recall"]*100:.2f}%')
    print(f'F1-Score: {metrics["f1"]*100:.2f}%')

    print(f"\nPer-class results:")
    for i in range(num_classes):
        print(f"Class {idx_to_class[i]}:")
        print(f'  Precision: {metrics["precision_per_class"][i]*100:.2f}%')
        print(f'  Recall: {metrics["recall_per_class"][i]*100:.2f}%')
        print(f'  F1-Score: {metrics["f1_per_class"][i]*100:.2f}%')
        print(f'  Support: {metrics["support_per_class"][i]}')

    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
