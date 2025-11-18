#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import torch
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
import numpy as np
import matplotlib.pyplot as plt

try:
    import timm

    TIMM_AVAILABLE = True
    print(f"timm version: {timm.__version__}")
except ImportError:
    print("Warning: timm not available. Please install with: pip install timm")
    TIMM_AVAILABLE = False

# Import the fixed datasets module
from util.datasets_fixed import build_dataset


def test_model_creation():
    """Test model creation with various configurations"""
    if not TIMM_AVAILABLE:
        print("timm not available for testing")
        return None, None

    test_models = [
        "vit_large_patch16_224",
        "vit_base_patch16_224",
        "vit_small_patch16_224",
    ]

    # Test different global_pool values for timm 0.9.16
    global_pool_options = ["avg", "token", ""]

    for model_name in test_models:
        for global_pool in global_pool_options:
            try:
                print(
                    f"Testing model creation: {model_name} with global_pool='{global_pool}'"
                )
                model = timm.create_model(
                    model_name,
                    pretrained=False,
                    num_classes=2,
                    global_pool=global_pool,
                )
                print(
                    f"✓ Successfully created {model_name} with global_pool='{global_pool}'"
                )
                return model_name, global_pool
            except Exception as e:
                print(
                    f"✗ Failed to create {model_name} with global_pool='{global_pool}': {e}"
                )

    return None, None


def find_test_dataset_path(data_path):
    """Find the correct test dataset path"""
    possible_paths = [
        os.path.join(data_path, "test"),
        os.path.join(data_path, "val"),
        data_path,  # Use the data_path directly if it contains class folders
    ]

    for path in possible_paths:
        if os.path.exists(path):
            # Check if this path contains class directories
            try:
                entries = os.listdir(path)
                class_dirs = [
                    entry
                    for entry in entries
                    if os.path.isdir(os.path.join(path, entry))
                ]
                if class_dirs:
                    print(f"Found test dataset at: {path}")
                    print(f"Classes found: {class_dirs}")
                    return path
            except:
                continue

    print(f"Warning: Could not find test dataset. Tried paths: {possible_paths}")
    return None


def get_args_parser():
    parser = argparse.ArgumentParser("RETFound Testing", add_help=False)

    # Model Parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to test",
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--resume", required=True, help="resume from checkpoint")
    parser.add_argument(
        "--global_pool",
        type=str,
        default="avg",
        help="Global pooling type: avg, token, or empty string",
    )
    parser.add_argument(
        "--cls_token",
        action="store_true",
        help="Use class token instead of global pool for classification",
    )

    # Dataset Parameters
    parser.add_argument("--data_path", required=True, type=str, help="dataset path")
    parser.add_argument(
        "--nb_classes", default=5, type=int, help="number of the classification types"
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--batch_size", default=32, type=int, help="batch size for testing"
    )

    # Random Erase params (needed for dataset building)
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )

    # Augmentation parameters (needed for dataset building)
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help="Use AutoAugment policy",
    )

    return parser


def plot_roc_curve(fpr, tpr, auc_score, output_dir, class_names=None):
    """Plot and save ROC curve"""
    plt.figure(figsize=(8, 6))
    if len(fpr) == len(tpr) and not isinstance(fpr[0], (list, np.ndarray)):
        # Binary classification
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {auc_score:.3f})",
        )
    else:
        # Multi-class classification
        for i in range(len(fpr)):
            class_name = class_names[i] if class_names else f"Class {i}"
            plt.plot(
                fpr[i], tpr[i], lw=2, label=f"{class_name} (AUC = {auc_score[i]:.3f})"
            )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, output_dir, class_names=None):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def calculate_metrics(all_labels, all_preds, predicted_classes, nb_classes):
    """Calculate comprehensive evaluation metrics"""
    results = {}

    # Basic accuracy
    accuracy = np.mean(predicted_classes == all_labels)
    results["accuracy"] = float(accuracy)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, predicted_classes)
    results["confusion_matrix"] = cm.tolist()

    # Per-class metrics
    if nb_classes == 2:
        # Binary classification
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_score = (
            2 * (precision * sensitivity) / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0.0
        )

        results["sensitivity"] = float(sensitivity)
        results["specificity"] = float(specificity)
        results["precision"] = float(precision)
        results["f1_score"] = float(f1_score)

        # AUC Score
        auc_score = roc_auc_score(all_labels, all_preds[:, 1])
        results["auc_score"] = float(auc_score)

        # ROC Curve data
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds[:, 1])
        results["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

    else:
        # Multi-class classification
        # Calculate per-class sensitivity and specificity
        sensitivities = []
        specificities = []
        precisions = []
        f1_scores = []

        for i in range(nb_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fn - fp

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (
                2 * (precision * sensitivity) / (precision + sensitivity)
                if (precision + sensitivity) > 0
                else 0.0
            )

            sensitivities.append(sensitivity)
            specificities.append(specificity)
            precisions.append(precision)
            f1_scores.append(f1)

        results["per_class_sensitivity"] = sensitivities
        results["per_class_specificity"] = specificities
        results["per_class_precision"] = precisions
        results["per_class_f1_score"] = f1_scores

        # Average metrics
        results["macro_avg_sensitivity"] = float(np.mean(sensitivities))
        results["macro_avg_specificity"] = float(np.mean(specificities))
        results["macro_avg_precision"] = float(np.mean(precisions))
        results["macro_avg_f1_score"] = float(np.mean(f1_scores))

        # AUC Score (One-vs-Rest)
        try:
            auc_score = roc_auc_score(
                all_labels, all_preds, multi_class="ovr", average="macro"
            )
            results["auc_score_macro"] = float(auc_score)

            # Per-class AUC
            auc_per_class = roc_auc_score(
                all_labels, all_preds, multi_class="ovr", average=None
            )
            results["per_class_auc"] = auc_per_class.tolist()
        except ValueError as e:
            print(f"Warning: Could not calculate AUC score: {e}")
            results["auc_score_macro"] = None
            results["per_class_auc"] = None

    # Classification report
    class_names = [f"Class_{i}" for i in range(nb_classes)]
    report = classification_report(
        all_labels, predicted_classes, target_names=class_names, output_dict=True
    )
    results["classification_report"] = report

    return results


def main(args):
    if not TIMM_AVAILABLE:
        print(
            "Error: timm is required but not installed. Please install with: pip install timm"
        )
        return

    # Handle cls_token argument (convert to global_pool setting)
    if args.cls_token:
        args.global_pool = "token"

    # Test model creation first
    print("Testing model creation...")
    working_model, working_global_pool = test_model_creation()
    if working_model is None:
        print(
            "Error: Could not create any ViT model. Please check your timm installation."
        )
        return

    # Use the working model and global_pool if the requested ones don't work
    if working_model != args.model:
        print(f"Using {working_model} instead of {args.model}")
        args.model = working_model

    if working_global_pool != args.global_pool:
        print(
            f"Using global_pool='{working_global_pool}' instead of '{args.global_pool}'"
        )
        args.global_pool = working_global_pool

    device = torch.device(args.device)

    # build model using timm
    print(f"Creating model: {args.model} with global_pool='{args.global_pool}'")
    try:
        model = timm.create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool=args.global_pool,
        )
        print(f"Model created successfully: {model.__class__.__name__}")
    except Exception as e:
        print(f"Error creating model: {e}")
        return

    # Load checkpoint
    checkpoint = torch.load(args.resume, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    else:
        model_state_dict = checkpoint

    # Load model state dict
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(f"Loaded checkpoint with message: {msg}")

    model.to(device)
    model.eval()

    print(f"Model loaded from: {args.resume}")
    print(f"Model: {args.model}")
    print(f"Number of classes: {args.nb_classes}")

    # Find the correct test dataset path
    print(f"Looking for test dataset in: {args.data_path}")
    test_dataset_path = find_test_dataset_path(args.data_path)

    if test_dataset_path is None:
        print("Error: Could not find test dataset. Please check your data path.")
        print(
            f"Expected structure: {args.data_path}/test/class1/, {args.data_path}/test/class2/, etc."
        )
        print(f"Or: {args.data_path}/val/class1/, {args.data_path}/val/class2/, etc.")
        print(f"Or: {args.data_path}/class1/, {args.data_path}/class2/, etc.")
        return

    # Use the found test dataset path directly
    args.data_path = test_dataset_path

    try:
        dataset_test = build_dataset(is_train=False, args=args)
        print(f"Successfully loaded test dataset from: {test_dataset_path}")
    except Exception as e:
        print(f"Error loading dataset from {test_dataset_path}: {e}")
        import traceback

        traceback.print_exc()
        return

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    print(f"Test dataset size: {len(dataset_test)}")

    all_preds = []
    all_labels = []
    all_logits = []

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader_test):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(data_loader_test)}")

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output = model(images)
            logits = output.cpu().numpy()
            preds = torch.nn.functional.softmax(output, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    predicted_classes = np.argmax(all_preds, axis=1)

    print("Calculating metrics...")

    # Calculate comprehensive metrics
    results = calculate_metrics(
        all_labels, all_preds, predicted_classes, args.nb_classes
    )

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"Overall Accuracy: {results['accuracy']:.4f}")

    if args.nb_classes == 2:
        print(f"Sensitivity (Recall): {results['sensitivity']:.4f}")
        print(f"Specificity: {results['specificity']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"AUC Score: {results['auc_score']:.4f}")
    else:
        print(f"Macro Average Sensitivity: {results['macro_avg_sensitivity']:.4f}")
        print(f"Macro Average Specificity: {results['macro_avg_specificity']:.4f}")
        print(f"Macro Average Precision: {results['macro_avg_precision']:.4f}")
        print(f"Macro Average F1-Score: {results['macro_avg_f1_score']:.4f}")
        if results["auc_score_macro"] is not None:
            print(f"Macro Average AUC Score: {results['auc_score_macro']:.4f}")

    print("\nConfusion Matrix:")
    print(np.array(results["confusion_matrix"]))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save detailed results
    output_path = os.path.join(args.output_dir, "test_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed results saved to: {output_path}")

    # Plot and save visualizations
    try:
        # Plot confusion matrix
        plot_confusion_matrix(np.array(results["confusion_matrix"]), args.output_dir)
        print(
            f"Confusion matrix plot saved to: {os.path.join(args.output_dir, 'confusion_matrix.png')}"
        )

        # Plot ROC curve
        if args.nb_classes == 2:
            fpr = results["roc_curve"]["fpr"]
            tpr = results["roc_curve"]["tpr"]
            auc_score = results["auc_score"]
            plot_roc_curve(fpr, tpr, auc_score, args.output_dir)
        else:
            if results["per_class_auc"] is not None:
                # For multi-class, we need to compute ROC for each class
                fpr_list = []
                tpr_list = []
                for i in range(args.nb_classes):
                    binary_labels = (all_labels == i).astype(int)
                    fpr, tpr, _ = roc_curve(binary_labels, all_preds[:, i])
                    fpr_list.append(fpr)
                    tpr_list.append(tpr)
                plot_roc_curve(
                    fpr_list, tpr_list, results["per_class_auc"], args.output_dir
                )

        print(
            f"ROC curve plot saved to: {os.path.join(args.output_dir, 'roc_curve.png')}"
        )

    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
