#!/usr/bin/env python3
"""
ConvNext-L Testing Script for Ophthalmology Image Classification
Author: AI Research Assistant
Description: Comprehensive testing script with medical imaging evaluation metrics
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    cohen_kappa_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConvNextTester:
    """
    ConvNext-L tester class with comprehensive medical imaging evaluation metrics
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Create output directories
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model and data loader
        self.model = None
        self.test_loader = None
        self.class_names = None
        self.num_classes = None

        # Results storage
        self.predictions = []
        self.probabilities = []
        self.true_labels = []
        self.image_paths = []

    def load_model_and_config(self):
        """
        Load trained model and configuration
        """
        checkpoint_path = self.config["model_path"]
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract model configuration
        if "config" in checkpoint:
            model_config = checkpoint["config"]
            logger.info("Loaded model configuration from checkpoint")
        else:
            logger.warning("No configuration found in checkpoint, using defaults")
            model_config = {"drop_path_rate": 0.2}

        # Load class information
        if "class_names" in checkpoint:
            self.class_names = checkpoint["class_names"]
            self.num_classes = len(self.class_names)
        else:
            # Try to load from class_info.json
            class_info_path = Path(checkpoint_path).parent.parent / "class_info.json"
            if class_info_path.exists():
                with open(class_info_path, "r") as f:
                    class_info = json.load(f)
                    self.class_names = class_info["class_names"]
                    self.num_classes = class_info["num_classes"]
            else:
                raise ValueError(
                    "Class information not found in checkpoint or class_info.json"
                )

        # Create model
        self.model = timm.create_model(
            "convnext_large_in22k",
            pretrained=False,
            num_classes=self.num_classes,
            drop_path_rate=model_config.get("drop_path_rate", 0.2),
        )

        # Load model weights
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # Handle DataParallel wrapper
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Class names: {self.class_names}")

    def setup_data_loader(self):
        """
        Setup test data loader
        """
        # Test transforms (same as validation)
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Create dataset with custom class to track image paths
        class ImageFolderWithPaths(ImageFolder):
            def __getitem__(self, index):
                original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
                path = self.imgs[index][0]
                tuple_with_path = original_tuple + (path,)
                return tuple_with_path

        test_dataset = ImageFolderWithPaths(
            root=self.config["test_data_path"], transform=test_transform
        )

        # Verify class names match
        if test_dataset.classes != self.class_names:
            logger.warning("Test dataset classes don't match model classes")
            logger.warning(f"Model classes: {self.class_names}")
            logger.warning(f"Test classes: {test_dataset.classes}")

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

        logger.info(f"Test samples: {len(test_dataset)}")

    def run_inference(self):
        """
        Run inference on test dataset
        """
        logger.info("Running inference on test dataset...")

        self.predictions = []
        self.probabilities = []
        self.true_labels = []
        self.image_paths = []

        with torch.no_grad():
            for batch_idx, (data, target, paths) in enumerate(
                tqdm(self.test_loader, desc="Testing")
            ):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                with (
                    torch.cuda.amp.autocast()
                    if torch.cuda.is_available()
                    else torch.no_grad()
                ):
                    output = self.model(data)

                # Get probabilities and predictions
                probs = F.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)

                # Store results
                self.probabilities.extend(probs.cpu().numpy())
                self.predictions.extend(preds.cpu().numpy())
                self.true_labels.extend(target.cpu().numpy())
                self.image_paths.extend(paths)

        self.predictions = np.array(self.predictions)
        self.probabilities = np.array(self.probabilities)
        self.true_labels = np.array(self.true_labels)

        logger.info("Inference completed")

    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        """
        logger.info("Calculating evaluation metrics...")

        # Basic metrics
        accuracy = np.mean(self.predictions == self.true_labels) * 100

        # Classification report
        report = classification_report(
            self.true_labels,
            self.predictions,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        # Confusion matrix
        cm = confusion_matrix(self.true_labels, self.predictions)

        # Cohen's Kappa (important for medical diagnosis)
        kappa = cohen_kappa_score(self.true_labels, self.predictions)

        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = self.true_labels == i
            if np.sum(class_mask) > 0:
                # Sensitivity (Recall)
                sensitivity = np.sum(
                    (self.predictions == i) & (self.true_labels == i)
                ) / np.sum(class_mask)

                # Specificity
                true_negatives = np.sum(
                    (self.predictions != i) & (self.true_labels != i)
                )
                false_positives = np.sum(
                    (self.predictions == i) & (self.true_labels != i)
                )
                specificity = (
                    true_negatives / (true_negatives + false_positives)
                    if (true_negatives + false_positives) > 0
                    else 0
                )

                # AUC-ROC (for binary classification or one-vs-rest)
                if self.num_classes == 2:
                    auc_roc = roc_auc_score(
                        self.true_labels == i, self.probabilities[:, i]
                    )
                else:
                    try:
                        auc_roc = roc_auc_score(
                            self.true_labels == i, self.probabilities[:, i]
                        )
                    except:
                        auc_roc = 0.0

                per_class_metrics[class_name] = {
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "auc_roc": auc_roc,
                    "precision": report[class_name]["precision"],
                    "recall": report[class_name]["recall"],
                    "f1_score": report[class_name]["f1-score"],
                    "support": report[class_name]["support"],
                }

        # Overall metrics
        metrics = {
            "accuracy": accuracy,
            "kappa": kappa,
            "macro_avg": report["macro avg"],
            "weighted_avg": report["weighted avg"],
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        return metrics

    def plot_confusion_matrix(self, cm: np.ndarray):
        """
        Plot and save confusion matrix
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
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Normalized Count"},
        )

        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Also plot raw counts
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Count"},
        )

        plt.title("Confusion Matrix (Raw Counts)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "confusion_matrix_raw.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_roc_curves(self):
        """
        Plot ROC curves for each class
        """
        plt.figure(figsize=(12, 8))

        for i, class_name in enumerate(self.class_names):
            # Binary classification: current class vs all others
            y_true_binary = (self.true_labels == i).astype(int)
            y_scores = self.probabilities[:, i]

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            auc_score = roc_auc_score(y_true_binary, y_scores)

            # Plot
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc_score:.3f})")

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - One vs Rest")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_precision_recall_curves(self):
        """
        Plot Precision-Recall curves for each class
        """
        plt.figure(figsize=(12, 8))

        for i, class_name in enumerate(self.class_names):
            # Binary classification: current class vs all others
            y_true_binary = (self.true_labels == i).astype(int)
            y_scores = self.probabilities[:, i]

            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
            ap_score = average_precision_score(y_true_binary, y_scores)

            # Plot
            plt.plot(recall, precision, label=f"{class_name} (AP = {ap_score:.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "precision_recall_curves.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def save_detailed_results(self, metrics: Dict):
        """
        Save detailed results including per-image predictions
        """
        # Create detailed results DataFrame
        results_df = pd.DataFrame(
            {
                "image_path": self.image_paths,
                "true_label": [self.class_names[label] for label in self.true_labels],
                "predicted_label": [
                    self.class_names[pred] for pred in self.predictions
                ],
                "correct": self.predictions == self.true_labels,
                "true_label_idx": self.true_labels,
                "predicted_label_idx": self.predictions,
            }
        )

        # Add probability columns
        for i, class_name in enumerate(self.class_names):
            results_df[f"prob_{class_name}"] = self.probabilities[:, i]

        # Add confidence (max probability)
        results_df["confidence"] = np.max(self.probabilities, axis=1)

        # Sort by confidence (lowest first to identify uncertain predictions)
        results_df = results_df.sort_values("confidence")

        # Save to CSV
        results_df.to_csv(self.output_dir / "detailed_results.csv", index=False)

        # Save metrics to JSON
        with open(self.output_dir / "evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Create summary report
        self.create_summary_report(metrics, results_df)

    def create_summary_report(self, metrics: Dict, results_df: pd.DataFrame):
        """
        Create a comprehensive summary report
        """
        report_lines = []
        report_lines.append("# ConvNext-L Ophthalmology Classification - Test Results")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Overall Performance
        report_lines.append("## Overall Performance")
        report_lines.append(f"- **Accuracy**: {metrics['accuracy']:.2f}%")
        report_lines.append(f"- **Cohen's Kappa**: {metrics['kappa']:.4f}")

        # Interpret Kappa score
        if metrics["kappa"] > 0.80:
            kappa_interpretation = "High agreement"
        elif metrics["kappa"] > 0.60:
            kappa_interpretation = "Significant agreement"
        else:
            kappa_interpretation = "Low agreement"
        report_lines.append(f"- **Kappa Interpretation**: {kappa_interpretation}")
        report_lines.append("")

        # Macro and Weighted Averages
        report_lines.append("## Average Metrics")
        report_lines.append(
            f"- **Macro Avg Precision**: {metrics['macro_avg']['precision']:.4f}"
        )
        report_lines.append(
            f"- **Macro Avg Recall**: {metrics['macro_avg']['recall']:.4f}"
        )
        report_lines.append(
            f"- **Macro Avg F1-Score**: {metrics['macro_avg']['f1-score']:.4f}"
        )
        report_lines.append(
            f"- **Weighted Avg Precision**: {metrics['weighted_avg']['precision']:.4f}"
        )
        report_lines.append(
            f"- **Weighted Avg Recall**: {metrics['weighted_avg']['recall']:.4f}"
        )
        report_lines.append(
            f"- **Weighted Avg F1-Score**: {metrics['weighted_avg']['f1-score']:.4f}"
        )
        report_lines.append("")

        # Per-Class Performance
        report_lines.append("## Per-Class Performance")
        for class_name, class_metrics in metrics["per_class_metrics"].items():
            report_lines.append(f"### {class_name}")
            report_lines.append(
                f"- **Sensitivity (Recall)**: {class_metrics['sensitivity']:.4f}"
            )
            report_lines.append(
                f"- **Specificity**: {class_metrics['specificity']:.4f}"
            )
            report_lines.append(f"- **Precision**: {class_metrics['precision']:.4f}")
            report_lines.append(f"- **F1-Score**: {class_metrics['f1_score']:.4f}")
            report_lines.append(f"- **AUC-ROC**: {class_metrics['auc_roc']:.4f}")
            report_lines.append(f"- **Support**: {class_metrics['support']}")
            report_lines.append("")

        # Error Analysis
        report_lines.append("## Error Analysis")
        incorrect_predictions = results_df[~results_df["correct"]]
        report_lines.append(f"- **Total Errors**: {len(incorrect_predictions)}")
        report_lines.append(
            f"- **Error Rate**: {len(incorrect_predictions)/len(results_df)*100:.2f}%"
        )

        # Most common errors
        error_combinations = (
            incorrect_predictions.groupby(["true_label", "predicted_label"])
            .size()
            .sort_values(ascending=False)
        )
        report_lines.append("- **Most Common Errors**:")
        for (true_label, pred_label), count in error_combinations.head(5).items():
            report_lines.append(f"  - {true_label} → {pred_label}: {count} cases")
        report_lines.append("")

        # Low confidence predictions
        low_confidence = results_df[results_df["confidence"] < 0.7]
        report_lines.append(
            f"- **Low Confidence Predictions (<70%)**: {len(low_confidence)}"
        )
        report_lines.append(
            f"- **Low Confidence Rate**: {len(low_confidence)/len(results_df)*100:.2f}%"
        )
        report_lines.append("")

        # Recommendations
        report_lines.append("## Clinical Recommendations")
        if metrics["accuracy"] > 95:
            report_lines.append(
                "- **Excellent performance**: Model shows high diagnostic accuracy suitable for clinical assistance."
            )
        elif metrics["accuracy"] > 90:
            report_lines.append(
                "- **Good performance**: Model shows good diagnostic accuracy but may benefit from additional validation."
            )
        else:
            report_lines.append(
                "- **Moderate performance**: Consider additional training or data augmentation."
            )

        if metrics["kappa"] > 0.80:
            report_lines.append(
                "- **High diagnostic consistency**: Model shows excellent agreement with ground truth."
            )
        else:
            report_lines.append(
                "- **Consider improving consistency**: Model may benefit from additional training or class balancing."
            )

        report_lines.append("")
        report_lines.append("## Files Generated")
        report_lines.append(
            "- `evaluation_metrics.json`: Complete metrics in JSON format"
        )
        report_lines.append(
            "- `detailed_results.csv`: Per-image predictions and probabilities"
        )
        report_lines.append("- `confusion_matrix.png`: Normalized confusion matrix")
        report_lines.append("- `confusion_matrix_raw.png`: Raw count confusion matrix")
        report_lines.append("- `roc_curves.png`: ROC curves for each class")
        report_lines.append("- `precision_recall_curves.png`: Precision-Recall curves")

        # Save report
        with open(self.output_dir / "test_report.md", "w") as f:
            f.write("\n".join(report_lines))

    def run_evaluation(self):
        """
        Run complete evaluation pipeline
        """
        logger.info("Starting evaluation pipeline...")

        # Load model and setup data
        self.load_model_and_config()
        self.setup_data_loader()

        # Run inference
        self.run_inference()

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Generate visualizations
        cm = np.array(metrics["confusion_matrix"])
        self.plot_confusion_matrix(cm)
        self.plot_roc_curves()
        self.plot_precision_recall_curves()

        # Save results
        self.save_detailed_results(metrics)

        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
        logger.info(f"Cohen's Kappa: {metrics['kappa']:.4f}")


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="ConvNext-L Testing for Ophthalmology")

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to test dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Output directory for test results",
    )

    # Optional arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    return parser.parse_args()


def main():
    """
    Main testing function
    """
    args = parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    if not os.path.exists(args.test_data_path):
        raise FileNotFoundError(f"Test data path not found: {args.test_data_path}")

    # Create config
    config = {
        "model_path": args.model_path,
        "test_data_path": args.test_data_path,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    # Initialize tester
    tester = ConvNextTester(config)

    # Run evaluation
    tester.run_evaluation()

    logger.info("Testing script completed successfully!")


if __name__ == "__main__":
    main()
