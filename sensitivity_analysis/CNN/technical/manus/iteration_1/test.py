#!/usr/bin/env python3
"""
Testing script for ConvNext-L classifier on ophthalmology fundus images.
Evaluates the trained model on test data and saves detailed results.
"""

import argparse
import os
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import convnext_large
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class TestFundusDataset(Dataset):
    """Dataset for test images with filename tracking."""

    def __init__(
        self,
        data_dir: str,
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.samples = []

        # Collect all test images
        for class_name, class_idx in class_to_idx.items():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in [
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".tiff",
                        ".bmp",
                    ]:
                        self.samples.append((str(img_path), class_idx, img_path.name))

        print(
            f"Found {len(self.samples)} test images across {len(class_to_idx)} classes"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label, filename = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label, filename


def get_test_transforms(input_size: int = 224) -> transforms.Compose:
    """Get test transforms (same as validation transforms)."""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_model(num_classes: int) -> nn.Module:
    """Create ConvNext-L model with custom classifier head."""
    model = convnext_large(weights=None)

    # Replace classifier head (must match training configuration)
    # The corrected classifier structure: AdaptiveAvgPool2d -> Flatten -> LayerNorm -> Dropout -> Linear
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # Global average pooling
        nn.Flatten(1),  # Flatten to [batch_size, features]
        nn.LayerNorm(in_features),  # LayerNorm after flattening
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )

    return model


def load_model(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, Dict[str, int]]:
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    # Create model
    model = create_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(
        f"Validation accuracy during training: {checkpoint.get('val_acc', 'N/A'):.2f}%"
    )
    print(f"Training epoch: {checkpoint.get('epoch', 'N/A')}")

    return model, class_to_idx


def test_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Test the model and collect detailed results."""

    model.eval()
    all_results = []
    all_predictions = []
    all_labels = []
    all_probabilities = []

    print("Starting model evaluation...")
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (data, target, filenames) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            _, predicted = output.max(1)

            # Convert to numpy for easier handling
            probs_np = probabilities.cpu().numpy()
            pred_np = predicted.cpu().numpy()
            target_np = target.cpu().numpy()

            # Store results for each image in the batch
            for i in range(len(filenames)):
                result = {
                    "filename": filenames[i],
                    "true_label": class_names[target_np[i]],
                    "predicted_label": class_names[pred_np[i]],
                    "correct": pred_np[i] == target_np[i],
                }

                # Add probability scores for each class
                for j, class_name in enumerate(class_names):
                    result[f"prob_{class_name}"] = probs_np[i, j]

                all_results.append(result)

            # Collect for overall metrics
            all_predictions.extend(pred_np)
            all_labels.extend(target_np)
            all_probabilities.extend(probs_np)

            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

    test_time = time.time() - start_time
    print(f"Testing completed in {test_time:.1f}s")

    return all_results, np.array(all_predictions), np.array(all_labels)


def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Calculate comprehensive evaluation metrics."""

    # Basic accuracy
    accuracy = (predictions == labels).mean() * 100

    # Per-class metrics
    report = classification_report(
        labels, predictions, target_names=class_names, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    # AUC scores (if binary or can compute)
    auc_scores = {}
    if len(class_names) == 2:
        # Binary classification
        try:
            auc_scores["binary_auc"] = roc_auc_score(labels, probabilities[:, 1])
        except:
            auc_scores["binary_auc"] = "N/A"
    else:
        # Multi-class AUC (one-vs-rest)
        try:
            auc_scores["macro_auc"] = roc_auc_score(
                labels, probabilities, multi_class="ovr", average="macro"
            )
            auc_scores["weighted_auc"] = roc_auc_score(
                labels, probabilities, multi_class="ovr", average="weighted"
            )
        except:
            auc_scores["macro_auc"] = "N/A"
            auc_scores["weighted_auc"] = "N/A"

    metrics = {
        "overall_accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "auc_scores": auc_scores,
        "num_samples": len(labels),
        "num_classes": len(class_names),
    }

    return metrics


def save_results_csv(
    results: List[Dict], output_path: str, class_names: List[str]
) -> None:
    """Save detailed results to CSV file."""

    # Define CSV columns
    columns = ["filename", "true_label", "predicted_label", "correct"]

    # Add probability columns for each class
    prob_columns = [f"prob_{class_name}" for class_name in class_names]
    columns.extend(prob_columns)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()

        for result in results:
            # Format probabilities to reasonable precision
            formatted_result = result.copy()
            for prob_col in prob_columns:
                if prob_col in formatted_result:
                    formatted_result[prob_col] = f"{formatted_result[prob_col]:.6f}"

            writer.writerow(formatted_result)

    print(f"Detailed results saved to {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], save_path: str
) -> None:
    """Plot and save confusion matrix."""

    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names))))

    # Normalize confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Count"},
    )

    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to {save_path}")


def save_metrics_report(metrics: Dict, class_names: List[str], save_path: str) -> None:
    """Save comprehensive metrics report."""

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("OPHTHALMOLOGY CLASSIFIER TEST RESULTS")
    report_lines.append("=" * 60)
    report_lines.append("")

    # Overall metrics
    report_lines.append(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    report_lines.append(f"Number of Test Samples: {metrics['num_samples']}")
    report_lines.append(f"Number of Classes: {metrics['num_classes']}")
    report_lines.append("")

    # AUC scores
    report_lines.append("AUC Scores:")
    for auc_name, auc_value in metrics["auc_scores"].items():
        if isinstance(auc_value, (int, float)):
            report_lines.append(f"  {auc_name}: {auc_value:.4f}")
        else:
            report_lines.append(f"  {auc_name}: {auc_value}")
    report_lines.append("")

    # Per-class metrics
    report_lines.append("Per-Class Metrics:")
    report_lines.append("-" * 40)

    class_report = metrics["classification_report"]
    for class_name in class_names:
        if class_name in class_report:
            class_metrics = class_report[class_name]
            report_lines.append(f"{class_name}:")
            report_lines.append(f"  Precision: {class_metrics['precision']:.4f}")
            report_lines.append(f"  Recall: {class_metrics['recall']:.4f}")
            report_lines.append(f"  F1-Score: {class_metrics['f1-score']:.4f}")
            report_lines.append(f"  Support: {class_metrics['support']}")
            report_lines.append("")

    # Macro and weighted averages
    if "macro avg" in class_report:
        macro_avg = class_report["macro avg"]
        report_lines.append("Macro Average:")
        report_lines.append(f"  Precision: {macro_avg['precision']:.4f}")
        report_lines.append(f"  Recall: {macro_avg['recall']:.4f}")
        report_lines.append(f"  F1-Score: {macro_avg['f1-score']:.4f}")
        report_lines.append("")

    if "weighted avg" in class_report:
        weighted_avg = class_report["weighted avg"]
        report_lines.append("Weighted Average:")
        report_lines.append(f"  Precision: {weighted_avg['precision']:.4f}")
        report_lines.append(f"  Recall: {weighted_avg['recall']:.4f}")
        report_lines.append(f"  F1-Score: {weighted_avg['f1-score']:.4f}")
        report_lines.append("")

    # Confusion matrix
    report_lines.append("Confusion Matrix:")
    report_lines.append("-" * 20)
    cm = np.array(metrics["confusion_matrix"])

    # Header
    header = "True\\Pred".ljust(12)
    for class_name in class_names:
        header += class_name[:8].ljust(10)
    report_lines.append(header)

    # Matrix rows
    for i, class_name in enumerate(class_names):
        row = class_name[:10].ljust(12)
        for j in range(len(class_names)):
            row += str(cm[i, j]).ljust(10)
        report_lines.append(row)

    # Save report
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Metrics report saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test ConvNext-L classifier on ophthalmology images"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing test folder",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory to save test results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for testing (default: 32)",
    )
    parser.add_argument(
        "--input_size", type=int, default=224, help="Input image size (default: 224)"
    )

    args = parser.parse_args()

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Load model
    model, class_to_idx = load_model(args.model_path, device)
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]

    print(f"Classes: {class_names}")

    # Create test dataset
    test_transform = get_test_transforms(args.input_size)
    test_dataset = TestFundusDataset(
        os.path.join(args.data_root, "test"), class_to_idx, transform=test_transform
    )

    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Test model
    results, predictions, labels = test_model(model, test_loader, device, class_names)

    # Calculate metrics
    probabilities = np.array(
        [
            [result[f"prob_{class_name}"] for class_name in class_names]
            for result in results
        ]
    )
    metrics = calculate_metrics(predictions, labels, probabilities, class_names)

    # Print summary
    print(f"\nTest Results Summary:")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    print(f"Number of test samples: {metrics['num_samples']}")

    # Save results
    csv_path = results_dir / "test_results.csv"
    save_results_csv(results, str(csv_path), class_names)

    # Save confusion matrix plot
    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]), class_names, str(cm_path)
    )

    # Save metrics report
    report_path = results_dir / "test_metrics_report.txt"
    save_metrics_report(metrics, class_names, str(report_path))

    # Save metrics as JSON
    json_path = results_dir / "test_metrics.json"
    # Convert numpy arrays to lists for JSON serialization
    json_metrics = metrics.copy()
    json_metrics["confusion_matrix"] = np.array(metrics["confusion_matrix"]).tolist()

    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=2, default=str)

    print(f"\nAll results saved to {results_dir}")
    print(f"- Detailed results: {csv_path}")
    print(f"- Confusion matrix: {cm_path}")
    print(f"- Metrics report: {report_path}")
    print(f"- Metrics JSON: {json_path}")


if __name__ == "__main__":
    main()
