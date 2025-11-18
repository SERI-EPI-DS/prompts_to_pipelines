#!/usr/bin/env python3
"""
test_convnext.py
=================

This script evaluates a fine‑tuned **ConvNeXt‑Large** model on a test dataset
and saves per‑image predictions.  It expects the dataset to be organised in
subdirectories where each folder name corresponds to a class label (the same
``root/class_x/...`` pattern used by ``ImageFolder``【806275724524080†L1609-L1623】).  The script loads the
model weights saved by ``train_convnext.py`` and decodes predicted indices
into human‑readable labels using the ``class_names.json`` file saved during
training.

Features:

* **Automatic class mapping:** attempts to read ``class_names.json`` from the
  same directory as the model checkpoint.  This ensures that the order of
  classes at inference matches the training order.

* **Evaluation metrics:** computes overall accuracy and per‑class accuracy.  If
  scikit‑learn is installed, also outputs a classification report and
  confusion matrix.

* **CSV output:** writes a ``predictions.csv`` file containing the image path,
  ground‑truth label, predicted label, and prediction confidence.

Example usage::

    python test_convnext.py \
        --data-dir path/to/test \
        --model-path path/to/outputs/best_model.pth \
        --output-dir path/to/results

Dependencies: torch, torchvision.  Optionally scikit‑learn for detailed metrics.
"""

import argparse
import csv
import json
import os
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned ConvNeXt‑Large model on a test set."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the test dataset folder (containing subfolders for each class).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model weights (.pth file).",
    )
    parser.add_argument(
        "--class-file",
        type=str,
        default=None,
        help="Optional path to class_names.json. If not provided, the script will look in the same directory as --model-path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save the predictions CSV and metrics summary.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading (default: 4).",
    )
    return parser.parse_args()


def build_model(num_classes: int) -> nn.Module:
    """Instantiate a ConvNeXt‑Large model with the given number of classes.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    Returns
    -------
    nn.Module
        ConvNeXt‑Large model ready for inference.
    """
    weights = models.ConvNeXt_Large_Weights.DEFAULT
    model = models.convnext_large(weights=weights)
    # Replace classifier with custom head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_class_names(class_file: str, fallback_classes: List[str]) -> List[str]:
    """Load class names from JSON if available; otherwise return fallback.

    Parameters
    ----------
    class_file : str
        Path to class_names.json.
    fallback_classes : List[str]
        Class names derived from the dataset.  Used if class_file is missing.

    Returns
    -------
    List[str]
        Ordered list of class names.
    """
    if class_file is not None and os.path.isfile(class_file):
        with open(class_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return fallback_classes


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[List[int], List[int], List[float]]:
    """Run inference on the dataset and collect predictions.

    Returns lists of true labels, predicted labels, and probabilities for the
    predicted class.
    """
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_confs: List[float] = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = softmax(outputs)
            confs, preds = torch.max(probs, 1)
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())
    return all_targets, all_preds, all_confs


def compute_metrics(
    targets: List[int], preds: List[int], class_names: List[str]
) -> str:
    """Compute overall and per‑class accuracy; return a formatted string.

    If scikit‑learn is available, also include a classification report.
    """
    import numpy as np
    from collections import Counter

    n = len(targets)
    if n == 0:
        return "No samples to evaluate."
    targets_arr = np.array(targets)
    preds_arr = np.array(preds)
    overall_acc = (targets_arr == preds_arr).sum() / n
    # Per‑class accuracy
    metrics_lines = [f"Overall accuracy: {overall_acc:.4f}\n"]
    metrics_lines.append("Per‑class accuracy:\n")
    for idx, name in enumerate(class_names):
        mask = targets_arr == idx
        if mask.any():
            class_acc = (preds_arr[mask] == idx).sum() / mask.sum()
            metrics_lines.append(f"  {name}: {class_acc:.4f}\n")
        else:
            metrics_lines.append(f"  {name}: no samples in test set\n")
    # Try to import sklearn for additional metrics
    try:
        from sklearn.metrics import classification_report, confusion_matrix

        report = classification_report(targets, preds, target_names=class_names)
        cm = confusion_matrix(targets, preds)
        metrics_lines.append("\nClassification report:\n")
        metrics_lines.append(report)
        metrics_lines.append("\nConfusion matrix:\n")
        metrics_lines.append(str(cm))
    except ImportError:
        metrics_lines.append(
            "\nscikit‑learn is not installed; skipping detailed classification report.\n"
        )
    return "".join(metrics_lines)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build evaluation transforms (resize + centre crop + normalisation)
    eval_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Load dataset
    dataset = datasets.ImageFolder(args.data_dir, transform=eval_transforms)
    # Determine class names: either from class_names.json or from dataset itself
    if args.class_file is None:
        # Guess class file path relative to model
        model_dir = os.path.dirname(args.model_path)
        candidate = os.path.join(model_dir, "class_names.json")
        class_file = candidate if os.path.isfile(candidate) else None
    else:
        class_file = args.class_file
    class_names = load_class_names(class_file, dataset.classes)
    # Build model
    num_classes = len(class_names)
    model = build_model(num_classes=num_classes)
    # Load weights
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    # DataLoader
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    # Run inference
    targets, preds, confs = evaluate_model(model, dataloader, device)
    # Compute metrics
    metrics_summary = compute_metrics(targets, preds, class_names)
    # Save predictions to CSV
    csv_path = os.path.join(args.output_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "true_label", "predicted_label", "confidence"])
        for (path, _), true_idx, pred_idx, conf in zip(
            dataset.samples, targets, preds, confs
        ):
            true_label = (
                class_names[true_idx] if true_idx < len(class_names) else str(true_idx)
            )
            pred_label = (
                class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
            )
            writer.writerow([path, true_label, pred_label, f"{conf:.4f}"])
    # Save metrics to a text file
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_summary)
    print(
        f"Evaluation complete. Predictions saved to {csv_path}. Metrics saved to {metrics_path}."
    )


if __name__ == "__main__":
    main()
