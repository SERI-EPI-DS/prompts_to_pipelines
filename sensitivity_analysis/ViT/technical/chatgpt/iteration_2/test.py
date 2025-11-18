#!/usr/bin/env python3
"""
test.py
=======

This script evaluates a fine‑tuned Swin‑V2‑B model on a held‑out test
set of fundus images.  It loads the saved weights from training,
constructs the appropriate model architecture, performs inference on
the test images and writes a CSV file containing the file name of
each test image, the predicted scores (probabilities) for each
class and the final predicted label.

If the test dataset is labelled (i.e. organised into class folders
under ``test``), the script also reports the overall accuracy at the
end of evaluation.  The CSV file always contains the model outputs
regardless of whether labels are present.

Example usage
-------------

    python test.py --data_dir /data --model_path /workspace/results/best_model.pth \
        --results_dir /workspace/results

"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing data and model paths and other
        hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fine‑tuned Swin‑V2‑B classifier on a held‑out test set. "
            "The data directory must contain a 'test' subfolder organised "
            "in the ImageFolder format."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root of the dataset containing the 'test' folder",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the .pth file storing the trained model weights",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory where the CSV with predictions will be saved",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    return parser.parse_args()


def build_model_for_inference(num_classes: int) -> nn.Module:
    """Instantiate a Swin‑V2‑B model for inference with the given number of classes.

    Parameters
    ----------
    num_classes : int
        Number of output classes for the classification head.

    Returns
    -------
    nn.Module
        The constructed model with the classification head adjusted.
    """
    # Similar to training, attempt to load pretrained weights when possible
    weights = None
    try:
        from torchvision.models import Swin_V2_B_Weights

        weights = Swin_V2_B_Weights.DEFAULT
    except (ImportError, AttributeError):
        weights = None
    model = models.swin_v2_b(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load a model checkpoint from disk.

    Parameters
    ----------
    path : str
        Path to the checkpoint file saved during training.
    device : torch.device
        Device to map the loaded tensors to.

    Returns
    -------
    dict
        A dictionary containing the checkpoint data.
    """
    return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Expected a 'test' directory inside {args.data_dir}, but it was not found."
        )

    # Define normalisation consistent with training/inference transforms of Swin‑V2
    # The mean and standard deviation values correspond to ImageNet and are
    # specified in the TorchVision documentation【885833164807516†L155-L162】.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Build dataset and dataloader
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load checkpoint and construct model
    checkpoint = load_checkpoint(args.model_path, device)
    # Fallback to dataset's class count if not specified in checkpoint
    num_classes = checkpoint.get("num_classes", len(test_dataset.classes))
    model = build_model_for_inference(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Determine the list of class names for prediction labels.  Prefer the
    # class_names saved in the checkpoint (if available) to ensure
    # consistency with training.  Otherwise use the ImageFolder classes.
    class_names: List[str] = checkpoint.get("class_names", test_dataset.classes)

    # Prepare output directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "test_results.csv"

    filenames: List[str] = []
    all_probs: List[List[float]] = []
    predictions: List[str] = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, pred_indices = torch.max(probs, dim=1)

            # Record per‑sample results
            for i in range(inputs.size(0)):
                # test_dataset.samples returns a list of (filepath, class_idx) tuples
                path, _ = test_dataset.samples[batch_idx * args.batch_size + i]
                filenames.append(os.path.basename(path))
                all_probs.append(probs[i].cpu().tolist())
                predictions.append(class_names[pred_indices[i].item()])

            # Update accuracy if ground truth labels exist
            total_correct += (pred_indices == targets).sum().item()
            total_samples += targets.size(0)

    # Write results to CSV
    header = ["filename"] + [f"score_{cls}" for cls in class_names] + ["prediction"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for fname, probs_row, pred in zip(filenames, all_probs, predictions):
            writer.writerow([fname] + [f"{p:.6f}" for p in probs_row] + [pred])

    print(f"Saved test predictions to {csv_path}")

    # Compute and display accuracy if labels are present in the test directory
    if total_samples > 0:
        acc = total_correct / total_samples
        print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
