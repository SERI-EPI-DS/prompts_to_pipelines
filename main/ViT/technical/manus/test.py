#!/usr/bin/env python3
"""
Swin-V2-B Testing Script for Ophthalmology Classification
Author: AI Assistant
Description: Tests a trained Swin-V2-B model on fundus photographs and saves results to CSV
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import swin_v2_b

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestDataset(Dataset):
    """Dataset for testing with filename tracking"""

    def __init__(self, root_dir: str, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx or {}
        self.samples = []
        self.filenames = []
        self.classes = []

        self._load_samples()

    def _load_samples(self):
        """Load all test samples and track filenames"""
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Test directory {self.root_dir} does not exist")

        # Get all class directories
        class_dirs = [
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]
        class_dirs.sort()

        # If no class mapping provided, create one
        if not self.class_to_idx:
            self.class_to_idx = {
                cls_name: idx for idx, cls_name in enumerate(class_dirs)
            }

        self.classes = list(self.class_to_idx.keys())

        # Load all samples
        for class_name in class_dirs:
            if class_name not in self.class_to_idx:
                logger.warning(
                    f"Class {class_name} not found in class mapping, skipping..."
                )
                continue

            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                ):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
                    # Store relative filename for CSV output
                    relative_filename = os.path.join(class_name, img_name)
                    self.filenames.append(relative_filename)

        logger.info(
            f"Loaded {len(self.samples)} test samples from {len(self.classes)} classes"
        )
        for cls_name, cls_idx in self.class_to_idx.items():
            count = sum(1 for _, idx in self.samples if idx == cls_idx)
            if count > 0:
                logger.info(f"  {cls_name}: {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        filename = self.filenames[idx]

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label, filename
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a dummy image and label
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label, filename


class SwinV2Classifier(nn.Module):
    """Swin-V2-B model for medical image classification"""

    def __init__(
        self, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.2
    ):
        super(SwinV2Classifier, self).__init__()

        # Load pretrained Swin-V2-B
        if pretrained:
            self.backbone = swin_v2_b(weights=None)  # We'll load our own weights
        else:
            self.backbone = swin_v2_b(weights=None)

        # Get the number of features from the classifier
        num_features = self.backbone.head.in_features

        # Replace the classifier head
        self.backbone.head = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(num_features, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)


def get_test_transforms(input_size: int = 224) -> transforms.Compose:
    """Get test transforms"""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(model_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load trained model from checkpoint"""
    logger.info(f"Loading model from {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")

    checkpoint = torch.load(model_path, map_location=device)

    # Extract model information
    class_mapping = checkpoint["class_mapping"]
    num_classes = len(class_mapping["classes"])

    # Create model
    model = SwinV2Classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_mapping['classes']}")

    return model, class_mapping


def test_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device, class_mapping: Dict
) -> Tuple[List, List, List, List]:
    """Test the model and return predictions"""
    model.eval()

    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_filenames = []

    logger.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, (inputs, labels, filenames) in enumerate(dataloader):
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_filenames.extend(filenames)

            if batch_idx % 50 == 0:
                logger.info(f"Processed batch {batch_idx}/{len(dataloader)}")

    logger.info(f"Inference completed on {len(all_filenames)} samples")

    return all_predictions, all_probabilities, all_labels, all_filenames


def save_results_csv(
    predictions: List,
    probabilities: List,
    labels: List,
    filenames: List,
    class_mapping: Dict,
    save_path: str,
):
    """Save test results to CSV file"""

    # Create DataFrame
    results_data = {
        "filename": filenames,
        "true_label": [class_mapping["idx_to_class"][label] for label in labels],
        "predicted_label": [
            class_mapping["idx_to_class"][pred] for pred in predictions
        ],
        "true_label_idx": labels,
        "predicted_label_idx": predictions,
    }

    # Add probability scores for each class
    class_names = class_mapping["classes"]
    for i, class_name in enumerate(class_names):
        results_data[f"prob_{class_name}"] = [prob[i] for prob in probabilities]

    # Add max probability (confidence)
    results_data["confidence"] = [max(prob) for prob in probabilities]

    # Create DataFrame
    df = pd.DataFrame(results_data)

    # Save to CSV
    df.to_csv(save_path, index=False)
    logger.info(f"Results saved to {save_path}")

    return df


def generate_evaluation_report(
    predictions: List, labels: List, class_mapping: Dict, save_dir: str
):
    """Generate comprehensive evaluation report"""

    class_names = class_mapping["classes"]

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    logger.info(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Generate classification report
    report = classification_report(
        labels, predictions, target_names=class_names, output_dict=True, zero_division=0
    )

    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(save_dir, "classification_report.csv"))

    # Generate and save confusion matrix
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Save detailed metrics
    metrics = {
        "overall_accuracy": accuracy,
        "num_samples": len(labels),
        "num_classes": len(class_names),
        "class_names": class_names,
        "per_class_metrics": {},
    }

    for i, class_name in enumerate(class_names):
        if class_name in report:
            metrics["per_class_metrics"][class_name] = {
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1_score": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
            }

    with open(os.path.join(save_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation report saved to {save_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Test Swin-V2-B for ophthalmology classification"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to the data root directory"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        required=True,
        help="Path to the project root directory",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory name within project root",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="best_model.pth",
        help="Name of the model file to load",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--output_csv", type=str, default="test_results.csv", help="Output CSV filename"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup directories
    data_root = Path(args.data_root)
    project_root = Path(args.project_root)
    results_dir = project_root / args.results_dir

    if not results_dir.exists():
        raise ValueError(
            f"Results directory {results_dir} does not exist. Please run training first."
        )

    test_dir = data_root / "test"
    if not test_dir.exists():
        raise ValueError(f"Test directory {test_dir} does not exist")

    # Load model
    model_path = results_dir / args.model_name
    model, class_mapping = load_model(str(model_path), device)

    # Get transforms
    test_transform = get_test_transforms(args.input_size)

    # Create test dataset
    logger.info("Loading test dataset...")
    test_dataset = TestDataset(
        str(test_dir),
        transform=test_transform,
        class_to_idx=class_mapping["class_to_idx"],
    )

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Run inference
    start_time = time.time()
    predictions, probabilities, labels, filenames = test_model(
        model, test_loader, device, class_mapping
    )
    inference_time = time.time() - start_time

    logger.info(f"Inference completed in {inference_time:.2f} seconds")
    logger.info(f"Average time per image: {inference_time/len(filenames)*1000:.2f} ms")

    # Save results to CSV
    csv_path = results_dir / args.output_csv
    results_df = save_results_csv(
        predictions, probabilities, labels, filenames, class_mapping, str(csv_path)
    )

    # Generate evaluation report
    metrics = generate_evaluation_report(
        predictions, labels, class_mapping, str(results_dir)
    )

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total test samples: {len(filenames)}")
    logger.info(
        f"Overall accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)"
    )
    logger.info(f"Number of classes: {metrics['num_classes']}")

    logger.info("\nPer-class performance:")
    for class_name, class_metrics in metrics["per_class_metrics"].items():
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {class_metrics['precision']:.4f}")
        logger.info(f"    Recall: {class_metrics['recall']:.4f}")
        logger.info(f"    F1-score: {class_metrics['f1_score']:.4f}")
        logger.info(f"    Support: {class_metrics['support']}")

    logger.info(f"\nResults saved to:")
    logger.info(f"  CSV file: {csv_path}")
    logger.info(f"  Evaluation report: {results_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
