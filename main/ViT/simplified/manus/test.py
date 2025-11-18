#!/usr/bin/env python3
"""
Swin-V2-B Testing Script for Ophthalmology Image Classification (Updated)

This script provides comprehensive testing and evaluation capabilities for the trained
Swin Transformer V2-Base model, including:

- Model evaluation on test datasets
- Grad-CAM visualization for interpretability
- Confusion matrix and classification reports
- Per-class performance analysis
- ROC curves and AUC scores
- Sample prediction visualization

Key Features:
- Support for multiple model checkpoints
- Grad-CAM heatmap generation for model interpretability
- Comprehensive metrics calculation and visualization
- Export results in multiple formats (JSON, CSV, images)
- Configurable test data paths
- Batch processing for efficiency
- Fixed JSON serialization for NumPy types

Based on research from:
- "Multi-Fundus Diseases Classification Using Retinal OCT Images with Swin Transformer V2"
- Grad-CAM implementation for transformer models

Author: AI Research Assistant
Date: 2025
Version: 1.1 (Fixed JSON serialization)
"""

import os
import sys
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
from PIL import Image

# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("testing.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.

    Args:
        obj: Object that may contain NumPy types

    Returns:
        Object with NumPy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


class MedicalImageDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for medical images with enhanced preprocessing.
    Supports the standard folder structure: dataset > train/test/val > class_folders > images
    """

    def __init__(
        self,
        data_path: str,
        transform: Optional[transforms.Compose] = None,
        return_paths: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the dataset folder
            transform: Torchvision transforms to apply
            return_paths: Whether to return image paths along with data
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.return_paths = return_paths
        self.dataset = ImageFolder(root=data_path, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        logger.info(f"Loaded test dataset from {data_path}")
        logger.info(f"Found {len(self.dataset)} images in {len(self.classes)} classes")
        logger.info(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.return_paths:
            image, label = self.dataset[idx]
            path = self.dataset.samples[idx][0]
            return image, label, path
        else:
            return self.dataset[idx]


def get_test_transforms(image_size: int = 256) -> transforms.Compose:
    """
    Get data transforms for testing (no augmentation).

    Args:
        image_size: Target image size (default: 256)

    Returns:
        Composed transforms for testing
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(
    model_path: str,
    num_classes: int,
    model_name: str = "swinv2_base_window12to16_192to256_22kft1k",
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Load trained Swin-V2-B model from checkpoint.

    Args:
        model_path: Path to the model checkpoint
        num_classes: Number of output classes
        model_name: Swin-V2 model variant
        device: Device to load the model on

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")

    # Create model architecture
    model = timm.create_model(
        model_name,
        pretrained=False,  # We're loading our own weights
        num_classes=num_classes,
        drop_rate=0.0,  # No dropout during inference
        drop_path_rate=0.0,
    )

    # Load checkpoint
    if model_path.endswith(".pth"):
        if "checkpoint" in model_path:
            # Load from training checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
            )
        else:
            # Load from saved model state dict
            model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise ValueError(f"Unsupported model file format: {model_path}")

    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully on {device}")
    return model


class SwinGradCAM:
    """
    Grad-CAM implementation for Swin Transformer models.
    """

    def __init__(self, model: nn.Module, target_layer: str = "layers.3.blocks.1.norm2"):
        """
        Initialize Grad-CAM for Swin Transformer.

        Args:
            model: Trained Swin Transformer model
            target_layer: Target layer for Grad-CAM (default: last layer before classification)
        """
        self.model = model
        self.target_layer = target_layer

        # Find the target layer
        target_layers = []
        for name, module in model.named_modules():
            if name == target_layer:
                target_layers.append(module)
                break

        if not target_layers:
            # Fallback to the last normalization layer
            for name, module in model.named_modules():
                if "norm" in name and isinstance(module, nn.LayerNorm):
                    target_layers = [module]

        self.cam = GradCAM(model=model, target_layers=target_layers)

    def generate_cam(
        self, input_tensor: torch.Tensor, target_class: int = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for CAM generation

        Returns:
            Grad-CAM heatmap as numpy array
        """
        targets = (
            [ClassifierOutputTarget(target_class)] if target_class is not None else None
        )
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device to run evaluation on
        class_names: List of class names

    Returns:
        Dictionary containing evaluation results
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []
    correct_predictions = 0
    total_samples = 0

    logger.info("Evaluating model...")

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=range(len(class_names))
    )

    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(all_labels, all_predictions, average="weighted")
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Classification report
    report = classification_report(
        all_labels, all_predictions, target_names=class_names, output_dict=True
    )

    # ROC AUC (for multi-class)
    try:
        if len(class_names) == 2:
            # Binary classification
            roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probabilities])
        else:
            # Multi-class classification
            roc_auc = roc_auc_score(
                all_labels, all_probabilities, multi_class="ovr", average="weighted"
            )
    except ValueError:
        roc_auc = None

    results = {
        "accuracy": float(accuracy),
        "precision_per_class": [float(p) for p in precision],
        "recall_per_class": [float(r) for r in recall],
        "f1_per_class": [float(f) for f in f1],
        "support_per_class": [int(s) for s in support],
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "classification_report": convert_numpy_types(report),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "predictions": [int(p) for p in all_predictions],
        "true_labels": [int(l) for l in all_labels],
        "probabilities": [[float(p) for p in prob] for prob in all_probabilities],
        "class_names": class_names,
    }

    logger.info(f"Evaluation completed. Overall accuracy: {accuracy:.4f}")

    return results


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], save_path: str, normalize: bool = True
):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
        fmt = ".2f"
    else:
        cm_norm = cm
        title = "Confusion Matrix"
        fmt = "d"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix saved to {save_path}")


def plot_roc_curves(results: Dict, save_path: str):
    """
    Plot ROC curves for multi-class classification.

    Args:
        results: Evaluation results dictionary
        save_path: Path to save the plot
    """
    if len(results["class_names"]) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(
            results["true_labels"], [p[1] for p in results["probabilities"]]
        )
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        # Multi-class classification
        n_classes = len(results["class_names"])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Convert to binary format for each class
        y_true_binary = np.eye(n_classes)[results["true_labels"]]
        y_score = np.array(results["probabilities"])

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f'{results["class_names"][i]} (AUC = {roc_auc[i]:.2f})',
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-class ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    logger.info(f"ROC curves saved to {save_path}")


def plot_class_performance(results: Dict, save_path: str):
    """
    Plot per-class performance metrics.

    Args:
        results: Evaluation results dictionary
        save_path: Path to save the plot
    """
    class_names = results["class_names"]
    precision = results["precision_per_class"]
    recall = results["recall_per_class"]
    f1 = results["f1_per_class"]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
    bars2 = ax.bar(x, recall, width, label="Recall", alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8)

    ax.set_xlabel("Classes")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Class performance plot saved to {save_path}")


def generate_gradcam_samples(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    save_dir: str,
    num_samples: int = 10,
):
    """
    Generate Grad-CAM visualizations for sample images.

    Args:
        model: Trained model
        dataloader: Test data loader (should return paths)
        device: Device to run on
        class_names: List of class names
        save_dir: Directory to save visualizations
        num_samples: Number of samples per class to visualize
    """
    logger.info("Generating Grad-CAM visualizations...")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Grad-CAM
    grad_cam = SwinGradCAM(model)

    # Denormalization transform
    denorm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    samples_per_class = {class_name: 0 for class_name in class_names}

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(
            tqdm(dataloader, desc="Generating Grad-CAM")
        ):
            if all(count >= num_samples for count in samples_per_class.values()):
                break

            images, labels = images.to(device), labels.to(device)

            for i in range(images.size(0)):
                class_name = class_names[labels[i].item()]
                if samples_per_class[class_name] >= num_samples:
                    continue

                # Get single image
                image = images[i : i + 1]
                label = labels[i].item()
                path = paths[i]

                # Get prediction
                output = model(image)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = F.softmax(output, dim=1)[0, predicted_class].item()

                # Generate Grad-CAM
                cam = grad_cam.generate_cam(image, target_class=predicted_class)

                # Prepare original image for visualization
                orig_image = denorm(image[0]).cpu()
                orig_image = torch.clamp(orig_image, 0, 1)
                orig_image_np = orig_image.permute(1, 2, 0).numpy()

                # Overlay CAM on image
                cam_image = show_cam_on_image(orig_image_np, cam, use_rgb=True)

                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                axes[0].imshow(orig_image_np)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                # Grad-CAM heatmap
                axes[1].imshow(cam, cmap="jet")
                axes[1].set_title("Grad-CAM Heatmap")
                axes[1].axis("off")

                # Overlay
                axes[2].imshow(cam_image)
                axes[2].set_title(
                    f"Grad-CAM Overlay\nPred: {class_names[predicted_class]} ({confidence:.3f})\nTrue: {class_name}"
                )
                axes[2].axis("off")

                # Save
                filename = (
                    f"{class_name}_sample_{samples_per_class[class_name]+1}_gradcam.png"
                )
                save_path = save_dir / filename
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()

                samples_per_class[class_name] += 1

    logger.info(f"Grad-CAM visualizations saved to {save_dir}")


def save_results(results: Dict, save_path: str):
    """
    Save evaluation results to JSON file with proper NumPy type conversion.

    Args:
        results: Evaluation results dictionary
        save_path: Path to save the results
    """
    # Convert numpy arrays and types to native Python types for JSON serialization
    results_copy = convert_numpy_types(results.copy())

    with open(save_path, "w") as f:
        json.dump(results_copy, f, indent=2)

    logger.info(f"Results saved to {save_path}")


def save_predictions_csv(results: Dict, save_path: str):
    """
    Save predictions to CSV file.

    Args:
        results: Evaluation results dictionary
        save_path: Path to save the CSV file
    """
    df = pd.DataFrame(
        {
            "true_label": results["true_labels"],
            "predicted_label": results["predictions"],
            "true_class": [results["class_names"][i] for i in results["true_labels"]],
            "predicted_class": [
                results["class_names"][i] for i in results["predictions"]
            ],
            "correct": [
                t == p for t, p in zip(results["true_labels"], results["predictions"])
            ],
        }
    )

    # Add probability columns
    for i, class_name in enumerate(results["class_names"]):
        df[f"prob_{class_name}"] = [prob[i] for prob in results["probabilities"]]

    df.to_csv(save_path, index=False)
    logger.info(f"Predictions saved to {save_path}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(
        description="Test Swin-V2-B for Ophthalmology Image Classification"
    )

    # Data arguments
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to test dataset folder"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory to save test results",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_window12to16_192to256_22kft1k",
        help="Swin-V2 model variant from timm",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Input image size (default: 256)"
    )

    # Testing arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    # Visualization arguments
    parser.add_argument(
        "--generate_gradcam",
        action="store_true",
        help="Generate Grad-CAM visualizations",
    )
    parser.add_argument(
        "--gradcam_samples",
        type=int,
        default=5,
        help="Number of Grad-CAM samples per class",
    )
    parser.add_argument(
        "--plot_results",
        action="store_true",
        default=True,
        help="Generate result plots",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load test dataset
    logger.info("Loading test dataset...")

    test_transform = get_test_transforms(args.image_size)

    test_dataset = MedicalImageDataset(
        data_path=args.test_data,
        transform=test_transform,
        return_paths=args.generate_gradcam,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load model
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes

    model = load_model(args.model_path, num_classes, args.model_name, device)

    # Evaluate model
    logger.info("Starting model evaluation...")
    start_time = time.time()

    results = evaluate_model(model, test_loader, device, class_names)

    eval_time = time.time() - start_time
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")

    # Print summary results
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Weighted Precision: {results['precision_weighted']:.4f}")
    logger.info(f"Weighted Recall: {results['recall_weighted']:.4f}")
    logger.info(f"Weighted F1-Score: {results['f1_weighted']:.4f}")
    if results["roc_auc"] is not None:
        logger.info(f"ROC AUC: {results['roc_auc']:.4f}")

    logger.info("\nPer-Class Results:")
    for i, class_name in enumerate(class_names):
        logger.info(
            f"{class_name}: Precision={results['precision_per_class'][i]:.4f}, "
            f"Recall={results['recall_per_class'][i]:.4f}, "
            f"F1={results['f1_per_class'][i]:.4f}, "
            f"Support={results['support_per_class'][i]}"
        )

    # Save results
    results_path = output_dir / "evaluation_results.json"
    save_results(results, str(results_path))

    # Save predictions CSV
    predictions_path = output_dir / "predictions.csv"
    save_predictions_csv(results, str(predictions_path))

    # Generate plots
    if args.plot_results:
        logger.info("Generating result plots...")

        # Confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            np.array(results["confusion_matrix"]), class_names, str(cm_path)
        )

        # ROC curves
        roc_path = output_dir / "roc_curves.png"
        plot_roc_curves(results, str(roc_path))

        # Class performance
        perf_path = output_dir / "class_performance.png"
        plot_class_performance(results, str(perf_path))

    # Generate Grad-CAM visualizations
    if args.generate_gradcam:
        logger.info("Generating Grad-CAM visualizations...")
        gradcam_dir = output_dir / "gradcam_visualizations"

        # Create new dataloader that returns paths
        gradcam_dataset = MedicalImageDataset(
            data_path=args.test_data, transform=test_transform, return_paths=True
        )

        gradcam_loader = DataLoader(
            gradcam_dataset,
            batch_size=1,  # Process one image at a time for Grad-CAM
            shuffle=True,
            num_workers=1,
        )

        generate_gradcam_samples(
            model,
            gradcam_loader,
            device,
            class_names,
            str(gradcam_dir),
            args.gradcam_samples,
        )

    # Save summary report
    summary_path = output_dir / "test_summary.txt"
    with open(summary_path, "w") as f:
        f.write("SWIN-V2-B MODEL EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Test Dataset: {args.test_data}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Total Test Samples: {len(test_dataset)}\n")
        f.write(f"Evaluation Time: {eval_time:.2f} seconds\n\n")

        f.write("OVERALL METRICS:\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Weighted Precision: {results['precision_weighted']:.4f}\n")
        f.write(f"Weighted Recall: {results['recall_weighted']:.4f}\n")
        f.write(f"Weighted F1-Score: {results['f1_weighted']:.4f}\n")
        if results["roc_auc"] is not None:
            f.write(f"ROC AUC: {results['roc_auc']:.4f}\n")

        f.write("\nPER-CLASS METRICS:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {results['precision_per_class'][i]:.4f}\n")
            f.write(f"  Recall: {results['recall_per_class'][i]:.4f}\n")
            f.write(f"  F1-Score: {results['f1_per_class'][i]:.4f}\n")
            f.write(f"  Support: {results['support_per_class'][i]}\n")

        f.write(f"\nDetailed classification report saved to: {results_path}\n")
        f.write(f"Predictions saved to: {predictions_path}\n")
        if args.plot_results:
            f.write(f"Plots saved to: {output_dir}\n")
        if args.generate_gradcam:
            f.write(f"Grad-CAM visualizations saved to: {gradcam_dir}\n")

    logger.info(f"Test summary saved to {summary_path}")
    logger.info("Testing completed successfully!")


if __name__ == "__main__":
    main()
