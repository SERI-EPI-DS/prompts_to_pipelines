#!/usr/bin/env python3
"""
ConvNext-L Testing and Inference Script for Ophthalmology Fundus Image Classification

This script provides comprehensive testing and inference capabilities:
- Load trained ConvNext-L models
- Evaluate on test datasets with detailed metrics
- Perform inference on individual images
- Generate prediction reports and visualizations
- Support for ensemble predictions
- Grad-CAM visualization for interpretability

Author: AI Research Assistant
Date: 2025
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import timm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("testing.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ConvNextClassifier(nn.Module):
    """
    ConvNext-L classifier - must match the training script architecture
    """

    def __init__(self, num_classes, pretrained=True, dropout_rate=0.3):
        super(ConvNextClassifier, self).__init__()

        # Load ConvNext-Large model
        self.backbone = timm.create_model(
            "convnext_large_in22k",
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="avg",
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Custom classification head with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(self.feature_dim // 2, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class SingleImageDataset(Dataset):
    """Dataset for single image inference"""

    def __init__(self, image_paths, transform=None):
        self.image_paths = (
            image_paths if isinstance(image_paths, list) else [image_paths]
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_path


class GradCAM:
    """
    Grad-CAM implementation for model interpretability
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, class_idx):
        # Forward pass
        output = self.model(input_image)

        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()

        # Generate CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam

        return cam


class ModelEvaluator:
    """
    Comprehensive model evaluation class
    """

    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)

    def evaluate_dataset(self, data_loader, save_dir=None):
        """Evaluate model on a dataset"""
        self.model.eval()

        all_targets = []
        all_predictions = []
        all_probabilities = []
        all_image_paths = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if len(batch) == 3:  # Dataset with paths
                    data, targets, paths = batch
                    all_image_paths.extend(paths)
                else:  # Standard dataset
                    data, targets = batch

                data = data.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)

                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(
            all_targets, all_predictions, all_probabilities
        )

        # Generate reports
        if save_dir:
            self._save_evaluation_report(
                all_targets,
                all_predictions,
                all_probabilities,
                metrics,
                save_dir,
                all_image_paths,
            )

        return metrics, all_targets, all_predictions, all_probabilities

    def predict_single_image(self, image_path, return_probabilities=True):
        """Predict on a single image"""
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            _, prediction = torch.max(output, 1)

        predicted_class = self.class_names[prediction.item()]
        confidence = probabilities[0][prediction.item()].item()

        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "prediction_index": prediction.item(),
        }

        if return_probabilities:
            result["class_probabilities"] = {
                self.class_names[i]: probabilities[0][i].item()
                for i in range(self.num_classes)
            }

        return result

    def predict_batch(self, image_paths, batch_size=32):
        """Predict on a batch of images"""
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset = SingleImageDataset(image_paths, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        results = []
        self.model.eval()

        with torch.no_grad():
            for images, paths in data_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)

                for i, path in enumerate(paths):
                    result = {
                        "image_path": path,
                        "predicted_class": self.class_names[predictions[i].item()],
                        "confidence": probabilities[i][predictions[i].item()].item(),
                        "prediction_index": predictions[i].item(),
                        "class_probabilities": {
                            self.class_names[j]: probabilities[i][j].item()
                            for j in range(self.num_classes)
                        },
                    }
                    results.append(result)

        return results

    def _calculate_comprehensive_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics"""
        metrics = {}

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Classification report
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )

        metrics["accuracy"] = accuracy
        metrics["macro_avg"] = report["macro avg"]
        metrics["weighted_avg"] = report["weighted avg"]

        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            if str(i) in report:
                metrics[f"{class_name}_precision"] = report[str(i)]["precision"]
                metrics[f"{class_name}_recall"] = report[str(i)]["recall"]
                metrics[f"{class_name}_f1"] = report[str(i)]["f1-score"]
                metrics[f"{class_name}_support"] = report[str(i)]["support"]

        # AUC scores
        if self.num_classes == 2:
            try:
                y_prob_positive = np.array(y_prob)[:, 1]
                auc = roc_auc_score(y_true, y_prob_positive)
                metrics["auc"] = auc
            except ValueError:
                logger.warning("Could not calculate binary AUC")
        else:
            try:
                auc_ovr = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
                auc_ovo = roc_auc_score(
                    y_true, y_prob, multi_class="ovo", average="weighted"
                )
                metrics["auc_ovr"] = auc_ovr
                metrics["auc_ovo"] = auc_ovo
            except ValueError:
                logger.warning("Could not calculate multiclass AUC")

        # Cohen's Kappa
        from sklearn.metrics import cohen_kappa_score

        kappa = cohen_kappa_score(y_true, y_pred)
        metrics["kappa"] = kappa

        return metrics

    def _save_evaluation_report(
        self, y_true, y_pred, y_prob, metrics, save_dir, image_paths=None
    ):
        """Save comprehensive evaluation report"""
        os.makedirs(save_dir, exist_ok=True)

        # Save metrics
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save confusion matrix
        self._save_confusion_matrix(y_true, y_pred, save_dir)

        # Save classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        # Save ROC curves
        if self.num_classes == 2:
            self._save_binary_roc_curve(y_true, y_prob, save_dir)
        else:
            self._save_multiclass_roc_curves(y_true, y_prob, save_dir)

        # Save precision-recall curves
        self._save_precision_recall_curves(y_true, y_prob, save_dir)

        # Save detailed predictions
        if image_paths:
            self._save_detailed_predictions(
                y_true, y_pred, y_prob, image_paths, save_dir
            )

    def _save_confusion_matrix(self, y_true, y_pred, save_dir):
        """Save confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Normalized confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Normalized Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "confusion_matrix_normalized.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _save_binary_roc_curve(self, y_true, y_prob, save_dir):
        """Save ROC curve for binary classification"""
        y_prob_positive = np.array(y_prob)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob_positive)
        auc = roc_auc_score(y_true, y_prob_positive)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.2f})"
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
        plt.savefig(
            os.path.join(save_dir, "roc_curve.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _save_multiclass_roc_curves(self, y_true, y_prob, save_dir):
        """Save ROC curves for multiclass classification"""
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        plt.figure(figsize=(10, 8))

        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], np.array(y_prob)[:, i])
            auc = roc_auc_score(y_true_bin[:, i], np.array(y_prob)[:, i])
            plt.plot(fpr, tpr, lw=2, label=f"{self.class_names[i]} (AUC = {auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multiclass ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "roc_curves_multiclass.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _save_precision_recall_curves(self, y_true, y_prob, save_dir):
        """Save precision-recall curves"""
        from sklearn.preprocessing import label_binarize

        if self.num_classes == 2:
            y_prob_positive = np.array(y_prob)[:, 1]
            precision, recall, _ = precision_recall_curve(y_true, y_prob_positive)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color="darkorange", lw=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, "precision_recall_curve.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

            plt.figure(figsize=(10, 8))

            for i in range(self.num_classes):
                precision, recall, _ = precision_recall_curve(
                    y_true_bin[:, i], np.array(y_prob)[:, i]
                )
                plt.plot(recall, precision, lw=2, label=f"{self.class_names[i]}")

            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Multiclass Precision-Recall Curves")
            plt.legend(loc="lower left")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, "precision_recall_curves_multiclass.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _save_detailed_predictions(self, y_true, y_pred, y_prob, image_paths, save_dir):
        """Save detailed predictions to CSV"""
        results_df = pd.DataFrame(
            {
                "image_path": image_paths,
                "true_label": [self.class_names[i] for i in y_true],
                "predicted_label": [self.class_names[i] for i in y_pred],
                "correct": [y_true[i] == y_pred[i] for i in range(len(y_true))],
                "confidence": [y_prob[i][y_pred[i]] for i in range(len(y_pred))],
            }
        )

        # Add probability columns for each class
        for i, class_name in enumerate(self.class_names):
            results_df[f"prob_{class_name}"] = [
                y_prob[j][i] for j in range(len(y_prob))
            ]

        results_df.to_csv(
            os.path.join(save_dir, "detailed_predictions.csv"), index=False
        )


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]

    model = ConvNextClassifier(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model with {num_classes} classes: {class_names}")

    return model, class_names


def main():
    parser = argparse.ArgumentParser(description="ConvNext-L Testing and Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "predict"],
        required=True,
        help="Mode: test on dataset or predict on images",
    )

    # For testing mode
    parser.add_argument("--test_dir", type=str, help="Test dataset directory")
    parser.add_argument(
        "--output_dir", type=str, default="./test_results", help="Output directory"
    )

    # For prediction mode
    parser.add_argument(
        "--image_path", type=str, help="Single image path for prediction"
    )
    parser.add_argument(
        "--image_dir", type=str, help="Directory of images for batch prediction"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.json",
        help="Output file for predictions",
    )

    # Common arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, class_names = load_model(args.model_path, device)
    evaluator = ModelEvaluator(model, device, class_names)

    if args.mode == "test":
        if not args.test_dir:
            raise ValueError("test_dir is required for test mode")

        logger.info(f"Testing model on dataset: {args.test_dir}")

        # Load test dataset
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_dataset = datasets.ImageFolder(args.test_dir, transform=transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # Evaluate
        os.makedirs(args.output_dir, exist_ok=True)
        metrics, y_true, y_pred, y_prob = evaluator.evaluate_dataset(
            test_loader, args.output_dir
        )

        # Print results
        logger.info("Test Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Weighted F1-Score: {metrics['weighted_avg']['f1-score']:.4f}")
        logger.info(f"Macro F1-Score: {metrics['macro_avg']['f1-score']:.4f}")
        if "kappa" in metrics:
            logger.info(f"Cohen's Kappa: {metrics['kappa']:.4f}")
        if "auc_ovr" in metrics:
            logger.info(f"AUC (OvR): {metrics['auc_ovr']:.4f}")

    elif args.mode == "predict":
        if args.image_path:
            # Single image prediction
            logger.info(f"Predicting on single image: {args.image_path}")
            result = evaluator.predict_single_image(args.image_path)

            logger.info(f"Predicted class: {result['predicted_class']}")
            logger.info(f"Confidence: {result['confidence']:.4f}")
            logger.info("Class probabilities:")
            for class_name, prob in result["class_probabilities"].items():
                logger.info(f"  {class_name}: {prob:.4f}")

            # Save result
            with open(args.output_file, "w") as f:
                json.dump(result, f, indent=2)

        elif args.image_dir:
            # Batch prediction
            logger.info(f"Predicting on images in directory: {args.image_dir}")

            # Get all image files
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(Path(args.image_dir).glob(f"*{ext}"))
                image_paths.extend(Path(args.image_dir).glob(f"*{ext.upper()}"))

            image_paths = [str(p) for p in image_paths]
            logger.info(f"Found {len(image_paths)} images")

            if image_paths:
                results = evaluator.predict_batch(image_paths, args.batch_size)

                # Save results
                with open(args.output_file, "w") as f:
                    json.dump(results, f, indent=2)

                # Save as CSV
                df = pd.DataFrame(results)
                csv_file = args.output_file.replace(".json", ".csv")
                df.to_csv(csv_file, index=False)

                logger.info(f"Predictions saved to {args.output_file} and {csv_file}")

                # Print summary
                predictions_summary = df["predicted_class"].value_counts()
                logger.info("Prediction summary:")
                for class_name, count in predictions_summary.items():
                    logger.info(f"  {class_name}: {count}")
            else:
                logger.warning("No images found in the specified directory")

        else:
            raise ValueError(
                "Either image_path or image_dir is required for predict mode"
            )


if __name__ == "__main__":
    main()
