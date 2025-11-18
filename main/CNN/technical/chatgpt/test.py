#!/usr/bin/env python3
"""
test.py
=======

This script evaluates a fine‑tuned ConvNeXt‑L model on a held out test
set of color fundus photographs.  It loads the model weights saved
during training and produces a CSV file with one row per test image.
Each row contains the relative filename, the predicted probability
scores for every class and the final predicted label.  The order of
class columns matches the order of class names from the training
procedure (stored in ``class_names.json``).

Usage example::

    python test.py \
        --data_root /path/to/dataset \
        --weights /path/to/best_model.pth \
        --results_dir /path/to/save/results

Note that the dataset directory passed via ``--data_root`` must contain
a ``test`` folder with images organized in class subfolders.  The
labels inside the ``test`` folder are not used; the model simply
performs inference on all images and writes predictions to a CSV
regardless of their directory names.
"""

import argparse
import csv
import json
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights


def build_model(num_classes: int) -> nn.Module:
    """Construct a ConvNeXt‑L model with a custom classification head.

    Args:
        num_classes: Number of classes for the output layer.

    Returns:
        A model instance with the final classifier adapted to ``num_classes``.
    """
    weights = ConvNeXt_Large_Weights.DEFAULT
    model = convnext_large(weights=weights)
    # Replace classifier as in training
    if isinstance(model.classifier, nn.Sequential):
        last_layer = model.classifier[-1]
        in_features = last_layer.in_features
        new_layers = list(model.classifier.children())
        new_layers[-1] = nn.Linear(in_features, num_classes)
        model.classifier = nn.Sequential(*new_layers)
    elif isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError(
            "Unexpected classifier structure on ConvNeXt model during test."
        )
    return model


def load_class_names(results_dir: str) -> List[str]:
    """Load the list of class names saved during training.

    Args:
        results_dir: Directory containing ``class_names.json``.

    Returns:
        A list of class names.
    """
    class_file = os.path.join(results_dir, "class_names.json")
    if not os.path.isfile(class_file):
        raise FileNotFoundError(
            f"Could not find class_names.json in {results_dir}; training must save this file"
        )
    with open(class_file, "r") as f:
        class_names = json.load(f)
    return class_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ConvNeXt‑L model on a test set"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing the 'test' subfolder",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the model weights (.pth file) saved during training",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory where the prediction CSV will be saved",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading (default: 4)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use for inference if available (default: 0)",
    )
    args = parser.parse_args()

    # Ensure the results directory exists
    os.makedirs(args.results_dir, exist_ok=True)

    test_dir = os.path.join(args.data_root, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}. Please provide a data_root containing a 'test' folder."
        )

    # Load class names from training
    class_names = load_class_names(args.results_dir)
    num_classes = len(class_names)

    # Construct model and load weights
    model = build_model(num_classes=num_classes)
    device_str = (
        f"cuda:{args.gpu}"
        if torch.cuda.is_available() and torch.cuda.device_count() > args.gpu
        else "cpu"
    )
    device = torch.device(device_str)
    model = model.to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Define transformation identical to validation (resize + center crop + normalize)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Create dataset and data loader.  The labels from ImageFolder are
    # ignored; only the images and their paths are used for inference.
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Prepare CSV file for writing results
    csv_path = os.path.join(args.results_dir, "predictions.csv")
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Header: filename, one column per class (probability), predicted label
        header = (
            ["filename"] + [f"score_{cls}" for cls in class_names] + ["predicted_label"]
        )
        writer.writerow(header)

        model.eval()
        with torch.no_grad():
            sample_idx = 0
            for inputs, _ in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                probs_np = probs.cpu().numpy()
                preds = np.argmax(probs_np, axis=1)
                batch_size_actual = probs_np.shape[0]
                # For each sample in the batch, extract its filename
                for i in range(batch_size_actual):
                    # Retrieve corresponding path from test_dataset.samples
                    img_path, _ = test_dataset.samples[sample_idx]
                    rel_path = os.path.relpath(img_path, test_dir)
                    scores = probs_np[i].tolist()
                    pred_label = class_names[preds[i]]
                    row = (
                        [rel_path] + [f"{score:.6f}" for score in scores] + [pred_label]
                    )
                    writer.writerow(row)
                    sample_idx += 1

    print(f"Inference complete.  Predictions saved to {csv_path}.")


if __name__ == "__main__":
    main()
