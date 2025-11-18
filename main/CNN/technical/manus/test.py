#!/usr/bin/env python3
"""
ConvNext-L Testing Script for Ophthalmology Fundus Image Classification

This script loads a trained ConvNext-L model and evaluates it on a test dataset,
generating predictions and saving results to a structured CSV file.

Author: Manus AI
Date: 2025-06-19
"""

import argparse
import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import torchvision.models as models

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
from PIL import Image

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "testing.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


class TestFundusDataset(Dataset):
    """Custom dataset class for test fundus photographs."""

    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize the test dataset.

        Args:
            data_dir: Path to the test data directory
            transform: Optional transform to be applied on images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        self._load_samples()

    def _load_samples(self) -> None:
        """Load all test image samples and create class mappings."""
        if not self.data_dir.exists():
            raise ValueError(f"Test data directory {self.data_dir} does not exist")

        # Get all class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        class_dirs.sort()

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.data_dir}")

        # Create class mappings
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Load all samples
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            # Supported image extensions
            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in extensions:
                    self.samples.append((str(img_path), class_idx, img_path.name))

        if not self.samples:
            raise ValueError(f"No valid images found in {self.data_dir}")

        print(
            f"Found {len(self.samples)} test images across {len(self.classes)} classes"
        )
        for i, cls_name in enumerate(self.classes):
            count = sum(1 for _, label, _ in self.samples if label == i)
            print(f"  {cls_name}: {count} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get a sample from the dataset."""
        img_path, label, filename = self.samples[idx]

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label, filename

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                black_image = Image.new("RGB", (224, 224), (0, 0, 0))
                image = self.transform(black_image)
            else:
                image = torch.zeros(3, 224, 224)
            return image, label, filename


def get_test_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Get test transforms (same as validation transforms).

    Args:
        input_size: Input image size

    Returns:
        Test transform pipeline
    """
    # ImageNet statistics for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Test transforms without augmentation
    test_transform = transforms.Compose(
        [
            transforms.Resize((int(input_size * 1.15), int(input_size * 1.15))),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return test_transform


def create_model(num_classes: int) -> nn.Module:
    """
    Create ConvNext-L model architecture (without pretrained weights).

    Args:
        num_classes: Number of output classes

    Returns:
        ConvNext-L model
    """
    # Load ConvNext-L model without pretrained weights
    model = models.convnext_large(weights=None)

    # Replace classifier
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model


def load_trained_model(
    model_path: str, num_classes: int, device: torch.device
) -> nn.Module:
    """
    Load trained model from checkpoint.

    Args:
        model_path: Path to the trained model checkpoint
        num_classes: Number of classes
        device: Device to load model on

    Returns:
        Loaded model
    """
    # Create model architecture
    model = create_model(num_classes)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Load model state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def test_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    logger: logging.Logger,
) -> Tuple[List[str], List[int], List[int], np.ndarray]:
    """
    Test the model and generate predictions.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run inference on
        class_names: List of class names
        logger: Logger instance

    Returns:
        Tuple of (filenames, true_labels, predictions, prediction_scores)
    """
    model.eval()

    all_filenames = []
    all_true_labels = []
    all_predictions = []
    all_prediction_scores = []

    logger.info("Starting model testing...")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")

        for batch_idx, (inputs, targets, filenames) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)

            # Get predictions
            _, predicted = outputs.max(1)

            # Store results
            all_filenames.extend(filenames)
            all_true_labels.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_prediction_scores.extend(probabilities.cpu().numpy())

            # Update progress
            accuracy = (predicted == targets).float().mean().item() * 100
            pbar.set_postfix({"Batch Acc": f"{accuracy:.2f}%"})

    # Convert to numpy array
    all_prediction_scores = np.array(all_prediction_scores)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_true_labels, all_predictions) * 100
    logger.info(f"Overall test accuracy: {overall_accuracy:.2f}%")

    return all_filenames, all_true_labels, all_predictions, all_prediction_scores


def save_results_csv(
    filenames: List[str],
    true_labels: List[int],
    predictions: List[int],
    prediction_scores: np.ndarray,
    class_names: List[str],
    save_path: Path,
) -> None:
    """
    Save test results to a structured CSV file.

    Args:
        filenames: List of image filenames
        true_labels: List of true class labels
        predictions: List of predicted class labels
        prediction_scores: Array of prediction scores for each class
        class_names: List of class names
        save_path: Path to save the CSV file
    """
    # Create results dictionary
    results_data = {
        "filename": filenames,
        "true_label": [class_names[label] for label in true_labels],
        "predicted_label": [class_names[pred] for pred in predictions],
        "true_label_idx": true_labels,
        "predicted_label_idx": predictions,
        "correct": [true == pred for true, pred in zip(true_labels, predictions)],
    }

    # Add prediction scores for each class
    for i, class_name in enumerate(class_names):
        results_data[f"score_{class_name}"] = prediction_scores[:, i]

    # Add confidence (max probability)
    results_data["confidence"] = np.max(prediction_scores, axis=1)

    # Create DataFrame
    results_df = pd.DataFrame(results_data)

    # Sort by filename for consistency
    results_df = results_df.sort_values("filename").reset_index(drop=True)

    # Save to CSV
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")


def generate_test_report(
    true_labels: List[int],
    predictions: List[int],
    class_names: List[str],
    save_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Generate comprehensive test report with metrics and visualizations.

    Args:
        true_labels: List of true class labels
        predictions: List of predicted class labels
        class_names: List of class names
        save_dir: Directory to save report files
        logger: Logger instance
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)

    # Generate classification report
    report = classification_report(
        true_labels, predictions, target_names=class_names, output_dict=True
    )

    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(save_dir / "test_classification_report.csv")

    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Test Confusion Matrix (Accuracy: {accuracy:.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_dir / "test_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Calculate per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # Create summary statistics
    summary_stats = {
        "overall_accuracy": accuracy,
        "num_samples": len(true_labels),
        "num_classes": len(class_names),
        "per_class_accuracy": dict(zip(class_names, per_class_accuracy)),
    }

    # Save summary
    import json

    with open(save_dir / "test_summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2, default=float)

    # Log summary
    logger.info(f"Test Summary:")
    logger.info(f"  Overall Accuracy: {accuracy:.4f}")
    logger.info(f"  Number of samples: {len(true_labels)}")
    logger.info(f"  Number of classes: {len(class_names)}")
    logger.info(f"  Per-class accuracy:")
    for class_name, acc in zip(class_names, per_class_accuracy):
        logger.info(f"    {class_name}: {acc:.4f}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="ConvNext-L Testing for Ophthalmology")

    # Required arguments
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to data root directory"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to results directory"
    )

    # Optional arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for testing (default: 32)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="test_results.csv",
        help="Output CSV filename (default: test_results.csv)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Setup paths
    data_root = Path(args.data_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(results_dir / "logs")
    logger.info(f"Starting testing with arguments: {args}")

    # Check test data directory
    test_dir = data_root / "test"
    if not test_dir.exists():
        raise ValueError(f"Test directory {test_dir} does not exist")

    # Check model file
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise ValueError(f"Model file {model_path} does not exist")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(
            f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Get test transforms
    test_transform = get_test_transforms()

    # Create test dataset
    logger.info("Loading test dataset...")
    test_dataset = TestFundusDataset(test_dir, transform=test_transform)

    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")
    logger.info(f"Total test samples: {len(test_dataset)}")

    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle for consistent results
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Test batches: {len(test_loader)}")

    # Load trained model
    logger.info(f"Loading trained model from: {model_path}")
    model = load_trained_model(str(model_path), num_classes, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Run testing
    start_time = time.time()
    filenames, true_labels, predictions, prediction_scores = test_model(
        model, test_loader, device, class_names, logger
    )
    test_time = time.time() - start_time

    logger.info(f"Testing completed in {test_time:.2f} seconds")
    logger.info(f"Average time per image: {test_time/len(test_dataset)*1000:.2f} ms")

    # Save results to CSV
    csv_path = results_dir / args.output_filename
    save_results_csv(
        filenames, true_labels, predictions, prediction_scores, class_names, csv_path
    )

    # Generate comprehensive test report
    generate_test_report(
        true_labels, predictions, class_names, results_dir / "test_report", logger
    )

    # Save testing summary
    test_summary = {
        "model_path": str(model_path),
        "test_data_path": str(test_dir),
        "num_test_samples": len(test_dataset),
        "num_classes": num_classes,
        "class_names": class_names,
        "overall_accuracy": accuracy_score(true_labels, predictions),
        "test_time_seconds": test_time,
        "avg_time_per_image_ms": test_time / len(test_dataset) * 1000,
        "batch_size": args.batch_size,
        "device": str(device),
    }

    import json

    with open(results_dir / "testing_summary.json", "w") as f:
        json.dump(test_summary, f, indent=2, default=float)

    logger.info("Testing summary saved to testing_summary.json")
    logger.info("Testing completed successfully!")

    # Print final results
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\n{'='*50}")
    print(f"TESTING COMPLETED")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total Test Samples: {len(test_dataset)}")
    print(f"Results saved to: {csv_path}")
    print(f"Detailed report saved to: {results_dir / 'test_report'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
