#!/usr/bin/env python3
"""
ConvNext-L Testing Script for Ophthalmology Image Classification
Author: AI Assistant for Ophthalmology Research
Description: Comprehensive testing and evaluation script for trained ConvNext-L models
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class ConvNextClassifier(nn.Module):
    """ConvNext-L classifier - must match training script architecture"""

    def __init__(self, num_classes, pretrained=True, dropout_rate=0.3):
        super(ConvNextClassifier, self).__init__()

        # Load ConvNext-L with ImageNet pretrained weights
        self.backbone = timm.create_model(
            "convnext_large", pretrained=pretrained, num_classes=0
        )

        # Get the feature dimension
        self.feature_dim = self.backbone.num_features

        # Custom classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class TestDataset:
    """Dataset class for testing with same preprocessing as training"""

    def __init__(self, data_dir, img_size=384):
        self.data_dir = data_dir
        self.img_size = img_size

        # Test-time augmentation transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # TTA transforms for improved accuracy
        self.tta_transforms = [
            transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomVerticalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomRotation(degrees=90),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        ]

    def get_dataset(self, use_tta=False):
        if use_tta:
            return datasets.ImageFolder(self.data_dir, transform=None)
        else:
            return datasets.ImageFolder(self.data_dir, transform=self.transform)


def test_time_augmentation(model, image, transforms_list, device):
    """Apply test-time augmentation for improved predictions"""
    predictions = []

    for transform in transforms_list:
        augmented_image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(augmented_image)
            predictions.append(torch.softmax(output, dim=1))

    # Average predictions
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    return avg_prediction


def evaluate_model(
    model, dataloader, device, class_names, use_tta=False, tta_transforms=None
):
    """Comprehensive model evaluation"""
    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []
    all_filenames = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch_idx, (data, target) in enumerate(pbar):
            if use_tta and tta_transforms:
                # Apply TTA for each image in batch
                batch_probs = []
                for i in range(data.size(0)):
                    image = transforms.ToPILImage()(data[i])
                    tta_prob = test_time_augmentation(
                        model, image, tta_transforms, device
                    )
                    batch_probs.append(tta_prob)
                probs = torch.cat(batch_probs, dim=0)
            else:
                data = data.to(device)
                output = model(data)
                probs = torch.softmax(output, dim=1)

            predicted = torch.argmax(probs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.numpy())

            # Get filenames if available
            if hasattr(dataloader.dataset, "imgs"):
                batch_start = batch_idx * dataloader.batch_size
                batch_end = min(
                    batch_start + dataloader.batch_size, len(dataloader.dataset.imgs)
                )
                batch_filenames = [
                    dataloader.dataset.imgs[i][0] for i in range(batch_start, batch_end)
                ]
                all_filenames.extend(batch_filenames)

    return (
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_targets),
        all_filenames,
    )


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=False):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, y_probs, class_names, save_path):
    """Plot ROC curves for each class"""
    n_classes = len(class_names)

    # Handle binary vs multi-class classification
    if n_classes == 2:
        # For binary classification, use the probability of the positive class
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc_score_val = roc_auc_score(y_true, y_probs[:, 1])

        plt.figure(figsize=(10, 8))
        plt.plot(
            fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc_score_val:.3f})"
        )
        plt.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random Classifier")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Binary Classification")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Return AUC scores in the expected format
        roc_auc = {
            class_names[0]: 1 - roc_auc_score_val,
            class_names[1]: roc_auc_score_val,
        }

    else:
        # For multi-class classification
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Handle the case where label_binarize returns 1D array for 2 classes
        if y_true_bin.ndim == 1:
            y_true_bin = y_true_bin.reshape(-1, 1)

        # Ensure we have the right number of columns
        if y_true_bin.shape[1] != n_classes:
            # Create proper binary matrix
            y_true_bin_new = np.zeros((len(y_true), n_classes))
            for i, class_idx in enumerate(y_true):
                y_true_bin_new[i, class_idx] = 1
            y_true_bin = y_true_bin_new

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        plt.figure(figsize=(12, 8))

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_probs[:, i])

            plt.plot(
                fpr[i],
                tpr[i],
                linewidth=2,
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
            )

        plt.plot([0, 1], [0, 1], "k--", linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for Each Class")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Convert to class name keys
        roc_auc = {class_names[i]: roc_auc[i] for i in range(n_classes)}

    return roc_auc


def plot_precision_recall_curves(y_true, y_probs, class_names, save_path):
    """Plot Precision-Recall curves for each class"""
    n_classes = len(class_names)

    # Handle binary vs multi-class classification
    if n_classes == 2:
        # For binary classification, use the probability of the positive class
        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
        avg_precision_score_val = average_precision_score(y_true, y_probs[:, 1])

        plt.figure(figsize=(10, 8))
        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"PR Curve (AP = {avg_precision_score_val:.3f})",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve for Binary Classification")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Return AP scores in the expected format
        avg_precision = {
            class_names[0]: 1 - avg_precision_score_val,
            class_names[1]: avg_precision_score_val,
        }

    else:
        # For multi-class classification
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Handle the case where label_binarize returns 1D array for 2 classes
        if y_true_bin.ndim == 1:
            y_true_bin = y_true_bin.reshape(-1, 1)

        # Ensure we have the right number of columns
        if y_true_bin.shape[1] != n_classes:
            # Create proper binary matrix
            y_true_bin_new = np.zeros((len(y_true), n_classes))
            for i, class_idx in enumerate(y_true):
                y_true_bin_new[i, class_idx] = 1
            y_true_bin = y_true_bin_new

        plt.figure(figsize=(12, 8))

        avg_precision = dict()
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_probs[:, i]
            )
            avg_precision[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])

            plt.plot(
                recall,
                precision,
                linewidth=2,
                label=f"{class_names[i]} (AP = {avg_precision[i]:.3f})",
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves for Each Class")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Convert to class name keys
        avg_precision = {class_names[i]: avg_precision[i] for i in range(n_classes)}

    return avg_precision


def plot_class_distribution(y_true, class_names, save_path):
    """Plot class distribution in test set"""
    unique, counts = np.unique(y_true, return_counts=True)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        range(len(class_names)),
        [counts[i] if i in unique else 0 for i in range(len(class_names))],
        color="skyblue",
        edgecolor="navy",
        alpha=0.7,
    )

    plt.xlabel("Classes")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution in Test Set")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_detailed_results(y_true, y_pred, y_probs, class_names, filenames, save_path):
    """Save detailed per-sample results"""
    results = []

    for i in range(len(y_true)):
        result = {
            "filename": os.path.basename(filenames[i]) if filenames else f"sample_{i}",
            "true_class": class_names[y_true[i]],
            "predicted_class": class_names[y_pred[i]],
            "correct": y_true[i] == y_pred[i],
            "confidence": float(np.max(y_probs[i])),
        }

        # Add probabilities for each class
        for j, class_name in enumerate(class_names):
            result[f"prob_{class_name}"] = float(y_probs[i][j])

        results.append(result)

    # Save as CSV
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)

    return df


def generate_summary_report(results_dict, save_path):
    """Generate a comprehensive summary report"""
    report = []
    report.append("=" * 80)
    report.append("CONVNEXT-L OPHTHALMOLOGY CLASSIFIER - TEST RESULTS")
    report.append("=" * 80)
    report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Overall metrics
    report.append("OVERALL PERFORMANCE:")
    report.append("-" * 40)
    report.append(
        f"Overall Accuracy: {results_dict['accuracy']:.4f} ({results_dict['accuracy']*100:.2f}%)"
    )
    report.append(f"Total Test Samples: {results_dict['total_samples']}")
    report.append(f"Number of Classes: {results_dict['num_classes']}")
    report.append("")

    # Per-class metrics
    report.append("PER-CLASS PERFORMANCE:")
    report.append("-" * 40)
    class_report = results_dict["classification_report"]
    for class_name in results_dict["class_names"]:
        if class_name in class_report:
            metrics = class_report[class_name]
            report.append(f"{class_name}:")
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall: {metrics['recall']:.4f}")
            report.append(f"  F1-Score: {metrics['f1-score']:.4f}")
            report.append(f"  Support: {metrics['support']}")
            if class_name in results_dict["roc_auc"]:
                report.append(f"  ROC-AUC: {results_dict['roc_auc'][class_name]:.4f}")
            if class_name in results_dict["avg_precision"]:
                report.append(
                    f"  Avg Precision: {results_dict['avg_precision'][class_name]:.4f}"
                )
            report.append("")

    # Macro averages
    report.append("MACRO AVERAGES:")
    report.append("-" * 40)
    macro_avg = class_report["macro avg"]
    report.append(f"Macro Precision: {macro_avg['precision']:.4f}")
    report.append(f"Macro Recall: {macro_avg['recall']:.4f}")
    report.append(f"Macro F1-Score: {macro_avg['f1-score']:.4f}")
    report.append("")

    # Weighted averages
    report.append("WEIGHTED AVERAGES:")
    report.append("-" * 40)
    weighted_avg = class_report["weighted avg"]
    report.append(f"Weighted Precision: {weighted_avg['precision']:.4f}")
    report.append(f"Weighted Recall: {weighted_avg['recall']:.4f}")
    report.append(f"Weighted F1-Score: {weighted_avg['f1-score']:.4f}")
    report.append("")

    # Model info
    if "model_info" in results_dict:
        report.append("MODEL INFORMATION:")
        report.append("-" * 40)
        model_info = results_dict["model_info"]
        if "config" in model_info:
            config = model_info["config"]
            report.append(f"Image Size: {config.get('img_size', 'N/A')}")
            report.append(f"Batch Size: {config.get('batch_size', 'N/A')}")
            report.append(f"Dropout Rate: {config.get('dropout', 'N/A')}")
        report.append("")

    report.append("=" * 80)

    # Save report
    with open(save_path, "w") as f:
        f.write("\n".join(report))

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="ConvNext-L Testing for Ophthalmology")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (.pth file)",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
        help="Path to test dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory to save test results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Use test-time augmentation for improved accuracy",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="Input image size (should match training)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading trained model...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Extract model information
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    model_config = checkpoint.get("config", {})

    print(f"Model classes: {class_names}")
    print(f"Number of classes: {num_classes}")

    # Initialize model
    dropout_rate = model_config.get("dropout", 0.3)
    model = ConvNextClassifier(num_classes, pretrained=False, dropout_rate=dropout_rate)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test dataset
    print("Loading test dataset...")
    img_size = args.img_size
    test_dataset_obj = TestDataset(args.test_data_dir, img_size)
    test_dataset = test_dataset_obj.get_dataset(use_tta=args.use_tta)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Evaluate model
    print("Evaluating model...")
    tta_transforms = test_dataset_obj.tta_transforms if args.use_tta else None
    y_pred, y_probs, y_true, filenames = evaluate_model(
        model, test_loader, device, class_names, args.use_tta, tta_transforms
    )

    # Calculate metrics
    accuracy = np.mean(y_pred == y_true)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Generate classification report
    class_report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    # Save classification report
    report_path = os.path.join(args.output_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(class_report, f, indent=2)

    # Generate and save plots
    print("Generating visualizations...")

    # Confusion matrices
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path, normalize=False)

    cm_norm_path = os.path.join(args.output_dir, "confusion_matrix_normalized.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_norm_path, normalize=True)

    # ROC curves
    roc_path = os.path.join(args.output_dir, "roc_curves.png")
    roc_auc_scores = plot_roc_curves(y_true, y_probs, class_names, roc_path)

    # Precision-Recall curves
    pr_path = os.path.join(args.output_dir, "precision_recall_curves.png")
    avg_precision_scores = plot_precision_recall_curves(
        y_true, y_probs, class_names, pr_path
    )

    # Class distribution
    dist_path = os.path.join(args.output_dir, "class_distribution.png")
    plot_class_distribution(y_true, class_names, dist_path)

    # Save detailed results
    print("Saving detailed results...")
    detailed_path = os.path.join(args.output_dir, "detailed_results.csv")
    results_df = save_detailed_results(
        y_true, y_pred, y_probs, class_names, filenames, detailed_path
    )

    # Prepare results dictionary
    # Filter model_info to avoid circular references
    safe_model_info = {}
    if "class_names" in checkpoint:
        safe_model_info["class_names"] = checkpoint["class_names"]
    if "num_classes" in checkpoint:
        safe_model_info["num_classes"] = checkpoint["num_classes"]
    if "config" in checkpoint:
        # Only include serializable config items
        safe_config = {}
        for key, value in checkpoint["config"].items():
            if isinstance(value, (str, int, float, bool, list)):
                safe_config[key] = value
            elif isinstance(value, np.ndarray):
                safe_config[key] = value.tolist()
            elif hasattr(value, "__dict__") and not callable(value):
                # Skip complex objects that might have circular references
                safe_config[key] = str(value)
            else:
                safe_config[key] = str(value)
        safe_model_info["config"] = safe_config

    results_dict = {
        "accuracy": accuracy,
        "total_samples": len(y_true),
        "num_classes": num_classes,
        "class_names": class_names,
        "classification_report": class_report,
        "roc_auc": roc_auc_scores,  # Already a dictionary with class names as keys
        "avg_precision": avg_precision_scores,  # Already a dictionary with class names as keys
        "model_info": safe_model_info,  # Filtered to avoid circular references
        "test_config": vars(args),
    }

    # Save complete results
    results_path = os.path.join(args.output_dir, "complete_results.json")

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def safe_serialize(obj):
        """Safely serialize objects, handling circular references"""
        try:
            if hasattr(obj, "__dict__") and not isinstance(
                obj, (str, int, float, bool, list, dict)
            ):
                # For complex objects, return a string representation
                return str(obj)
            elif callable(obj):
                # For functions/methods, return their name
                return (
                    f"<function: {obj.__name__}>"
                    if hasattr(obj, "__name__")
                    else "<function>"
                )
            else:
                return convert_numpy(obj)
        except:
            return str(obj)

    class SafeJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            return safe_serialize(obj)

    try:
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=2, cls=SafeJSONEncoder)
    except (ValueError, TypeError) as e:
        print(
            f"Warning: Could not save complete results as JSON due to serialization error: {e}"
        )
        # Save a simplified version without problematic fields
        simplified_results = {
            "accuracy": float(accuracy),
            "total_samples": int(len(y_true)),
            "num_classes": int(num_classes),
            "class_names": list(class_names),
            "roc_auc": {k: float(v) for k, v in roc_auc_scores.items()},
            "avg_precision": {k: float(v) for k, v in avg_precision_scores.items()},
            "test_config": {k: str(v) for k, v in vars(args).items()},
        }
        with open(results_path, "w") as f:
            json.dump(simplified_results, f, indent=2)

    # Generate summary report
    summary_path = os.path.join(args.output_dir, "summary_report.txt")
    summary_text = generate_summary_report(results_dict, summary_path)

    # Print summary
    print("\n" + summary_text)

    print(f"\nAll results saved to: {args.output_dir}")
    print("Files generated:")
    print(f"  - Classification report: {report_path}")
    print(f"  - Confusion matrix: {cm_path}")
    print(f"  - Normalized confusion matrix: {cm_norm_path}")
    print(f"  - ROC curves: {roc_path}")
    print(f"  - Precision-Recall curves: {pr_path}")
    print(f"  - Class distribution: {dist_path}")
    print(f"  - Detailed results: {detailed_path}")
    print(f"  - Complete results: {results_path}")
    print(f"  - Summary report: {summary_path}")


if __name__ == "__main__":
    main()
