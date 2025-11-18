"""
Testing/evaluation script for fine-tuned RETFound models.
Provides comprehensive evaluation metrics and visualization options.
Updated version with fixed JSON serialization and PyTorch 2.6+ compatibility.
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Add RETFound directory to path
sys.path.append("./RETFound_MAE")

try:
    import models_vit
    from util.pos_embed import interpolate_pos_embed
except ImportError:
    print(
        "Error: Could not import RETFound modules. Make sure RETFound_MAE is in the correct path."
    )
    print("You may need to adjust the sys.path.append line above.")
    sys.exit(1)


class RetinalTestDataset(datasets.ImageFolder):
    """Extended ImageFolder for test dataset with path tracking"""

    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path


def build_test_dataset(args):
    """Build test dataset with appropriate preprocessing"""

    transform = transforms.Compose(
        [
            transforms.Resize(int(args.input_size * 1.15)),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = RetinalTestDataset(args.test_data_path, transform=transform)

    return dataset


def load_model(args):
    """Load the fine-tuned model"""

    # Create model architecture
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=0.0,  # No dropout during testing
        global_pool=args.global_pool,
    )

    # Load checkpoint with PyTorch 2.6+ compatibility
    try:
        # Try loading with weights_only=True first
        checkpoint = torch.load(
            args.checkpoint_path, map_location="cpu", weights_only=True
        )
    except:
        # Fall back to weights_only=False if needed
        print("Note: Loading checkpoint with weights_only=False for compatibility")
        checkpoint = torch.load(
            args.checkpoint_path, map_location="cpu", weights_only=False
        )

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Load weights
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model from {args.checkpoint_path}")
    print(f"Load message: {msg}")

    return model


@torch.no_grad()
def get_predictions(model, data_loader, device):
    """Get all predictions, labels, and probabilities"""

    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_paths = []

    for samples, targets, paths in tqdm(data_loader, desc="Evaluating"):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        outputs = model(samples)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_paths.extend(paths)

    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_probabilities),
        all_paths,
    )


def to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(to_json_serializable(item) for item in obj)
    else:
        return obj


def calculate_metrics(predictions, labels, probabilities, class_names):
    """Calculate comprehensive evaluation metrics"""

    # Basic metrics
    accuracy = (predictions == labels).mean() * 100

    # Per-class metrics
    report_dict = classification_report(
        labels, predictions, target_names=class_names, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    # ROC AUC scores (one-vs-rest)
    auc_scores = {}
    if len(class_names) > 2:
        # Multi-class: calculate AUC for each class
        for i, class_name in enumerate(class_names):
            y_true_binary = (labels == i).astype(int)
            y_score = probabilities[:, i]
            try:
                auc = roc_auc_score(y_true_binary, y_score)
                auc_scores[class_name] = float(auc)  # Ensure it's a Python float
            except:
                auc_scores[class_name] = None

        # Macro and weighted average AUC
        valid_aucs = [auc for auc in auc_scores.values() if auc is not None]
        if valid_aucs:
            auc_scores["macro_avg"] = float(np.mean(valid_aucs))
    else:
        # Binary classification
        auc_scores["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))

    return {
        "accuracy": float(accuracy),
        "classification_report": to_json_serializable(report_dict),
        "confusion_matrix": cm.tolist(),  # Convert to list for JSON
        "auc_scores": auc_scores,
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""

    plt.figure(figsize=(10, 8))

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
        cbar_kws={"label": "Normalized Frequency"},
    )

    plt.title("Normalized Confusion Matrix", fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Also save raw counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )

    plt.title("Confusion Matrix (Raw Counts)", fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_raw.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(labels, probabilities, class_names, save_path):
    """Plot ROC curves for each class"""

    plt.figure(figsize=(10, 8))

    if len(class_names) > 2:
        # Multi-class
        for i, class_name in enumerate(class_names):
            y_true_binary = (labels == i).astype(int)
            y_score = probabilities[:, i]

            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            try:
                auc = roc_auc_score(y_true_binary, y_score)
                plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc:.3f})")
            except:
                pass
    else:
        # Binary classification
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        auc = roc_auc_score(labels, probabilities[:, 1])
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_predictions(predictions, labels, probabilities, paths, class_names, save_path):
    """Save detailed predictions to CSV"""

    # Create dataframe
    data = {
        "image_path": paths,
        "true_label": labels,
        "true_class": [class_names[l] for l in labels],
        "predicted_label": predictions,
        "predicted_class": [class_names[p] for p in predictions],
        "correct": predictions == labels,
        "confidence": probabilities.max(axis=1),
    }

    # Add probability for each class
    for i, class_name in enumerate(class_names):
        data[f"prob_{class_name}"] = probabilities[:, i]

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)

    return df


def analyze_errors(df, output_dir):
    """Analyze and save error cases"""

    # Get misclassified samples
    errors = df[df["correct"] == False].copy()

    if len(errors) > 0:
        # Sort by confidence (most confident errors first)
        errors = errors.sort_values("confidence", ascending=False)

        # Save error analysis
        errors.to_csv(os.path.join(output_dir, "error_analysis.csv"), index=False)

        # Find most common confusion
        confusion_counts = errors.groupby(["true_class", "predicted_class"]).size()
        if len(confusion_counts) > 0:
            most_common_confusion = confusion_counts.idxmax()
            confusion_str = f"{most_common_confusion[0]} -> {most_common_confusion[1]}"
        else:
            confusion_str = "N/A"

        # Error statistics
        error_stats = {
            "total_errors": int(len(errors)),
            "error_rate": float(len(errors) / len(df) * 100),
            "avg_confidence_on_errors": float(errors["confidence"].mean()),
            "most_common_confusion": confusion_str,
            "confusion_counts": {
                f"{true_cls} -> {pred_cls}": int(count)
                for (true_cls, pred_cls), count in confusion_counts.items()
            },
        }

        # Convert to JSON-serializable format
        error_stats = to_json_serializable(error_stats)

        with open(os.path.join(output_dir, "error_statistics.json"), "w") as f:
            json.dump(error_stats, f, indent=4)

        print(f"\nError Analysis:")
        print(f"Total errors: {error_stats['total_errors']}")
        print(f"Error rate: {error_stats['error_rate']:.2f}%")
        print(
            f"Average confidence on errors: {error_stats['avg_confidence_on_errors']:.3f}"
        )
        print(f"Most common confusion: {error_stats['most_common_confusion']}")
    else:
        print("\nNo errors found! Perfect classification.")
        error_stats = {
            "total_errors": 0,
            "error_rate": 0.0,
            "message": "No errors - perfect classification",
        }

        with open(os.path.join(output_dir, "error_statistics.json"), "w") as f:
            json.dump(error_stats, f, indent=4)

    return errors


def plot_class_distribution(df, class_names, output_dir):
    """Plot distribution of predictions vs true labels"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # True label distribution
    true_counts = df["true_class"].value_counts()
    ax1.bar(range(len(true_counts)), true_counts.values)
    ax1.set_xticks(range(len(true_counts)))
    ax1.set_xticklabels(true_counts.index, rotation=45, ha="right")
    ax1.set_title("True Label Distribution", fontsize=14)
    ax1.set_ylabel("Count")

    # Predicted label distribution
    pred_counts = df["predicted_class"].value_counts()
    ax2.bar(range(len(pred_counts)), pred_counts.values)
    ax2.set_xticks(range(len(pred_counts)))
    ax2.set_xticklabels(pred_counts.index, rotation=45, ha="right")
    ax2.set_title("Predicted Label Distribution", fontsize=14)
    ax2.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "class_distributions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main(args):
    """Main evaluation function"""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        cudnn.benchmark = True

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build test dataset
    dataset_test = build_test_dataset(args)

    # Get class names
    class_names = dataset_test.classes
    args.nb_classes = len(class_names)

    print(f"Found {len(dataset_test)} test images across {args.nb_classes} classes")
    print(f"Classes: {class_names}")

    # Data loader
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Load model
    model = load_model(args)
    model.to(device)

    # Get predictions
    print("\nRunning inference...")
    predictions, labels, probabilities, paths = get_predictions(
        model, data_loader_test, device
    )

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(predictions, labels, probabilities, class_names)

    # Print results
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    print(f"{'='*60}\n")

    print("Classification Report:")
    print("-" * 60)
    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    print(report_df.round(3))

    if metrics["auc_scores"]:
        print(f"\n{'='*60}")
        print("AUC Scores:")
        print("-" * 60)
        for class_name, auc in metrics["auc_scores"].items():
            if auc is not None:
                print(f"{class_name}: {auc:.3f}")

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")

    # Save metrics
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix plot
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        class_names,
        os.path.join(args.output_dir, "confusion_matrix.png"),
    )

    # Save ROC curves
    if args.nb_classes > 1:  # Only plot ROC for classification tasks
        plot_roc_curves(
            labels,
            probabilities,
            class_names,
            os.path.join(args.output_dir, "roc_curves.png"),
        )

    # Save detailed predictions
    df = save_predictions(
        predictions,
        labels,
        probabilities,
        paths,
        class_names,
        os.path.join(args.output_dir, "test_predictions.csv"),
    )

    # Plot class distributions
    plot_class_distribution(df, class_names, args.output_dir)

    # Analyze errors
    errors = analyze_errors(df, args.output_dir)

    # Save summary report
    summary = {
        "dataset_info": {
            "total_samples": len(dataset_test),
            "num_classes": args.nb_classes,
            "class_names": class_names,
        },
        "model_info": {
            "architecture": args.model,
            "checkpoint": args.checkpoint_path,
            "input_size": args.input_size,
        },
        "overall_performance": {
            "accuracy": metrics["accuracy"],
            "macro_avg_precision": metrics["classification_report"]
            .get("macro avg", {})
            .get("precision", 0)
            * 100,
            "macro_avg_recall": metrics["classification_report"]
            .get("macro avg", {})
            .get("recall", 0)
            * 100,
            "macro_avg_f1": metrics["classification_report"]
            .get("macro avg", {})
            .get("f1-score", 0)
            * 100,
        },
    }

    # Convert to JSON-serializable format
    summary = to_json_serializable(summary)

    with open(os.path.join(args.output_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nResults saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RETFound Testing Script")

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        help="Model architecture (must match training)",
    )
    parser.add_argument("--input_size", default=224, type=int, help="Image input size")
    parser.add_argument(
        "--global_pool", action="store_true", default=True, help="Use global pooling"
    )

    # Dataset parameters
    parser.add_argument(
        "--test_data_path", required=True, type=str, help="Path to test dataset folder"
    )
    parser.add_argument(
        "--checkpoint_path", required=True, type=str, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", required=True, type=str, help="Path to save results"
    )

    # Other parameters
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--num_workers", default=8, type=int, help="Number of data loading workers"
    )

    args = parser.parse_args()

    # Run evaluation
    main(args)
