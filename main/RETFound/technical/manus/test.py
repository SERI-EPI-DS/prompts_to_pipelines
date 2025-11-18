#!/usr/bin/env python3
"""
RETFound Fine-tuning Testing Script

This script conducts testing of the final weights of the RETFound classifier on a held-out test
portion of the dataset. It loads the trained model, performs inference on test images, and saves
the results in a structured CSV file containing file names, predicted classification scores for
each class, and final classifications.

Requirements:
- Python 3.11.0
- PyTorch 2.3.1
- TorchVision 0.18.1
- PyTorch-CUDA 12.1
- Single RTX3090 with 24GB VRAM

Author: Manus AI
Date: 2025-06-25
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
)

# Import RETFound modules
sys.path.append("RETFound")
try:
    import models_vit as models
    from util.pos_embed import interpolate_pos_embed
except ImportError as e:
    print(f"Error importing RETFound modules: {e}")
    print(
        "Please ensure the RETFound repository is available in the current directory."
    )
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class RETFoundTestDataset(Dataset):
    """
    Custom dataset class for RETFound testing.
    Handles image loading and preprocessing for inference.
    """

    def __init__(self, data_path: str, split: str = "test", transform=None):
        """
        Initialize the test dataset.

        Args:
            data_path: Path to the data directory
            split: Dataset split (default: 'test')
            transform: Image transformations
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform

        # Use ImageFolder for automatic class detection
        self.dataset = ImageFolder(root=self.data_path / split, transform=transform)

        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.samples = self.dataset.samples

        # Store file paths for CSV output
        self.file_paths = [sample[0] for sample in self.samples]
        self.file_names = [Path(path).name for path in self.file_paths]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        file_path = self.file_paths[idx]
        file_name = self.file_names[idx]
        return image, label, file_path, file_name


def get_args_parser():
    """
    Create argument parser for testing script.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser("RETFound Fine-tuning Testing", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="RETFound_mae",
        type=str,
        metavar="MODEL",
        help="Name of model to test (default: RETFound_mae)",
    )
    parser.add_argument(
        "--input_size", default=224, type=int, help="Images input size (default: 224)"
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument("--global_pool", action="store_true", default=True)
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    # Dataset parameters
    parser.add_argument("--data_path", default="./data", type=str, help="Dataset path")
    parser.add_argument(
        "--nb_classes",
        default=None,
        type=int,
        help="Number of the classification types (will be auto-detected if not specified)",
    )
    parser.add_argument(
        "--output_dir", default="./results", help="Path where to save results"
    )
    parser.add_argument("--device", default="cuda", help="Device to use for testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Testing parameters
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size for testing"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--test_split",
        default="test",
        type=str,
        help="Dataset split to test on (default: test)",
    )

    # Output parameters
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=True,
        help="Save detailed predictions to CSV",
    )
    parser.add_argument(
        "--save_metrics",
        action="store_true",
        default=True,
        help="Save evaluation metrics",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=True,
        help="Save visualization plots",
    )
    parser.add_argument(
        "--output_csv",
        default="test_results.csv",
        type=str,
        help="Output CSV filename for predictions",
    )

    return parser


def build_test_transforms(args):
    """
    Build image transformations for testing.

    Args:
        args: Command line arguments

    Returns:
        torchvision.transforms.Compose: Image transformations
    """
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]  # ImageNet std

    # Test transforms without augmentation
    transform_list = [
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(transform_list)


def create_test_dataset(args):
    """
    Create test dataset.

    Args:
        args: Command line arguments

    Returns:
        RETFoundTestDataset: Test dataset
    """
    # Build transforms
    test_transform = build_test_transforms(args)

    # Create dataset
    test_dataset = RETFoundTestDataset(
        args.data_path, args.test_split, transform=test_transform
    )

    # Auto-detect number of classes if not specified
    if args.nb_classes is None:
        args.nb_classes = len(test_dataset.classes)
        print(f"Auto-detected {args.nb_classes} classes: {test_dataset.classes}")

    return test_dataset


def load_model(args, device):
    """
    Load the trained model from checkpoint.

    Args:
        args: Command line arguments
        device: Device to load model on

    Returns:
        torch.nn.Module: Loaded model
    """
    print(f"Creating model: {args.model}")

    # Create model
    model = models.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Load model state dict
    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    else:
        model_state_dict = checkpoint

    # Load state dict
    msg = model.load_state_dict(model_state_dict, strict=True)
    print(f"Loaded checkpoint: {msg}")

    model.to(device)
    model.eval()

    return model


def test_model(model, data_loader, device, args):
    """
    Test the model and collect predictions.

    Args:
        model: The model to test
        data_loader: Test data loader
        device: Device to use
        args: Command line arguments

    Returns:
        Dict with test results
    """
    model.eval()

    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_file_names = []
    all_file_paths = []

    print("Running inference on test set...")
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, targets, file_paths, file_names) in enumerate(
            data_loader
        ):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)

            # Get predictions
            _, predictions = torch.max(outputs, 1)

            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_file_names.extend(file_names)
            all_file_paths.extend(file_paths)

            if batch_idx % 10 == 0:
                print(
                    f"Processed {batch_idx * args.batch_size}/{len(data_loader.dataset)} images"
                )

    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")

    return {
        "predictions": np.array(all_predictions),
        "probabilities": np.array(all_probabilities),
        "targets": np.array(all_targets),
        "file_names": all_file_names,
        "file_paths": all_file_paths,
        "inference_time": inference_time,
    }


def calculate_metrics(targets, predictions, probabilities, class_names):
    """
    Calculate comprehensive evaluation metrics.

    Args:
        targets: True labels
        predictions: Predicted labels
        probabilities: Prediction probabilities
        class_names: List of class names

    Returns:
        Dict with calculated metrics
    """
    # Basic metrics
    accuracy = accuracy_score(targets, predictions)
    precision_macro = precision_score(
        targets, predictions, average="macro", zero_division=0
    )
    precision_weighted = precision_score(
        targets, predictions, average="weighted", zero_division=0
    )
    recall_macro = recall_score(targets, predictions, average="macro", zero_division=0)
    recall_weighted = recall_score(
        targets, predictions, average="weighted", zero_division=0
    )
    f1_macro = f1_score(targets, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(targets, predictions, average="weighted", zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(
        targets, predictions, average=None, zero_division=0
    )
    recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(targets, predictions)

    # Classification report
    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # AUC scores (for multi-class)
    try:
        if len(class_names) == 2:
            # Binary classification
            auc_score = roc_auc_score(targets, probabilities[:, 1])
        else:
            # Multi-class classification
            auc_score = roc_auc_score(
                targets, probabilities, multi_class="ovr", average="weighted"
            )
    except ValueError:
        auc_score = None

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "classification_report": report,
        "auc_score": auc_score,
    }

    return metrics


def save_predictions_csv(results, class_names, output_path):
    """
    Save predictions to CSV file.

    Args:
        results: Test results dictionary
        class_names: List of class names
        output_path: Path to save CSV file
    """
    print(f"Saving predictions to {output_path}")

    # Prepare data for CSV
    csv_data = []

    for i in range(len(results["file_names"])):
        row = {
            "filename": results["file_names"][i],
            "filepath": results["file_paths"][i],
            "true_label": class_names[results["targets"][i]],
            "predicted_label": class_names[results["predictions"][i]],
            "correct": results["targets"][i] == results["predictions"][i],
        }

        # Add probability scores for each class
        for j, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = results["probabilities"][i][j]

        csv_data.append(row)

    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(csv_data)} predictions to {output_path}")


def save_metrics_json(metrics, class_names, output_path):
    """
    Save metrics to JSON file.

    Args:
        metrics: Metrics dictionary
        class_names: List of class names
        output_path: Path to save JSON file
    """
    print(f"Saving metrics to {output_path}")

    # Prepare metrics for JSON serialization
    json_metrics = {
        "overall_metrics": {
            "accuracy": float(metrics["accuracy"]),
            "precision_macro": float(metrics["precision_macro"]),
            "precision_weighted": float(metrics["precision_weighted"]),
            "recall_macro": float(metrics["recall_macro"]),
            "recall_weighted": float(metrics["recall_weighted"]),
            "f1_macro": float(metrics["f1_macro"]),
            "f1_weighted": float(metrics["f1_weighted"]),
            "auc_score": (
                float(metrics["auc_score"])
                if metrics["auc_score"] is not None
                else None
            ),
        },
        "per_class_metrics": {},
    }

    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        json_metrics["per_class_metrics"][class_name] = {
            "precision": float(metrics["precision_per_class"][i]),
            "recall": float(metrics["recall_per_class"][i]),
            "f1_score": float(metrics["f1_per_class"][i]),
        }

    # Add confusion matrix
    json_metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()

    # Add classification report
    json_metrics["classification_report"] = metrics["classification_report"]

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(json_metrics, f, indent=2)

    print(f"Saved metrics to {output_path}")


def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save plot
    """
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
    )

    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix plot to {output_path}")


def plot_class_metrics(metrics, class_names, output_path):
    """
    Plot per-class metrics.

    Args:
        metrics: Metrics dictionary
        class_names: List of class names
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(class_names))

    # Precision
    axes[0].bar(x, metrics["precision_per_class"])
    axes[0].set_title("Precision per Class")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Precision")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45)
    axes[0].set_ylim(0, 1)

    # Recall
    axes[1].bar(x, metrics["recall_per_class"])
    axes[1].set_title("Recall per Class")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Recall")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45)
    axes[1].set_ylim(0, 1)

    # F1-score
    axes[2].bar(x, metrics["f1_per_class"])
    axes[2].set_title("F1-Score per Class")
    axes[2].set_xlabel("Class")
    axes[2].set_ylabel("F1-Score")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved class metrics plot to {output_path}")


def print_summary(metrics, class_names, inference_time, num_samples):
    """
    Print test summary.

    Args:
        metrics: Metrics dictionary
        class_names: List of class names
        inference_time: Total inference time
        num_samples: Number of test samples
    """
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    print(f"Number of test samples: {num_samples}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Average time per image: {inference_time/num_samples:.4f} seconds")

    print("\nOVERALL METRICS:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    print(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")

    if metrics["auc_score"] is not None:
        print(f"AUC score: {metrics['auc_score']:.4f}")

    print("\nPER-CLASS METRICS:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(
            f"{class_name:<15} {metrics['precision_per_class'][i]:<10.4f} "
            f"{metrics['recall_per_class'][i]:<10.4f} {metrics['f1_per_class'][i]:<10.4f}"
        )

    print("\nCONFUSION MATRIX:")
    print(metrics["confusion_matrix"])

    print("=" * 60)


def main(args):
    """
    Main testing function.

    Args:
        args: Command line arguments
    """
    print(f"Job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"Arguments: {args}")

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create test dataset
    print("Creating test dataset...")
    test_dataset = create_test_dataset(args)

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Load model
    model = load_model(args, device)

    # Count parameters
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of params (M): {n_parameters / 1.e6:.2f}")

    # Run testing
    print("Starting testing...")
    results = test_model(model, test_loader, device, args)

    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(
        results["targets"],
        results["predictions"],
        results["probabilities"],
        test_dataset.classes,
    )

    # Save results
    if args.save_predictions:
        csv_path = output_dir / args.output_csv
        save_predictions_csv(results, test_dataset.classes, csv_path)

    if args.save_metrics:
        metrics_path = output_dir / "test_metrics.json"
        save_metrics_json(metrics, test_dataset.classes, metrics_path)

    if args.save_plots:
        # Confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            metrics["confusion_matrix"], test_dataset.classes, cm_path
        )

        # Class metrics
        class_metrics_path = output_dir / "class_metrics.png"
        plot_class_metrics(metrics, test_dataset.classes, class_metrics_path)

    # Print summary
    print_summary(
        metrics, test_dataset.classes, results["inference_time"], len(test_dataset)
    )

    print(f"\nTesting completed. Results saved to {output_dir}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    # Validate arguments
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    if not os.path.exists(args.data_path):
        print(f"Error: Data path not found: {args.data_path}")
        sys.exit(1)

    main(args)
