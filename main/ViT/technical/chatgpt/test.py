#!/usr/bin/env python3
"""
test.py
--------
Script to evaluate a trained Swin‑V2‑B model on a held‑out test set and
write predictions to a CSV file.  It loads the model architecture,
applies the same preprocessing used during validation and infers
probabilities for each class.  Results are saved in a structured CSV
containing the filename of each test image, the predicted scores for
every class and the final predicted label.

Usage:
    python test.py --data_root /path/to/data --model_path /path/to/best_model.pth --output_dir ./results

The CSV will be written to ``<output_dir>/test_results.csv``.  If the
test directory contains ground truth subfolders, the script also
computes and logs the overall accuracy.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import swin_v2_b, Swin_V2_B_Weights


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test a fine‑tuned Swin‑V2‑B model")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing the 'test' subfolder and optionally 'train' for class names",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (best_model.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the predictions CSV",
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
        help="Number of worker processes for data loading (default: 4)",
    )
    args = parser.parse_args()
    return args


def load_class_names(data_root: str) -> List[str]:
    """Infer class names by examining the 'train' directory structure.

    Args:
        data_root: Root of the dataset containing a 'train' directory.

    Returns:
        Sorted list of class names.
    """
    train_dir = Path(data_root) / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Expected directory '{train_dir}' does not exist. "
            "Class names cannot be determined."
        )
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    return class_names


def build_model(
    num_classes: int, checkpoint_path: str, device: torch.device
) -> nn.Module:
    """Construct a Swin‑V2‑B model and load saved weights.

    Args:
        num_classes: Number of classes in the dataset.
        checkpoint_path: Path to the saved model checkpoint (.pth file).
        device: Device on which to load the model.

    Returns:
        Model ready for inference with loaded parameters.
    """
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    model = swin_v2_b(weights=weights)
    # Replace classification head
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    model.to(device)

    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_test_loader(data_root: str, batch_size: int, num_workers: int) -> DataLoader:
    """Create DataLoader for the test set.

    Args:
        data_root: Root directory containing 'test' subfolder.
        batch_size: Batch size for inference.
        num_workers: Number of worker processes for data loading.

    Returns:
        DataLoader over the test dataset.
    """
    test_dir = os.path.join(data_root, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory '{test_dir}' does not exist.")

    # Use the same normalisation as training【990691079611788†L158-L162】
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize(
                (272, 272), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine class names from training set
    class_names = load_class_names(args.data_root)
    num_classes = len(class_names)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test loader
    test_loader = get_test_loader(args.data_root, args.batch_size, args.num_workers)

    # Build and load model
    model = build_model(num_classes, args.model_path, device)

    # Prepare CSV output
    csv_path = os.path.join(args.output_dir, "test_results.csv")
    header = ["filename"] + [f"prob_{cls}" for cls in class_names] + ["prediction"]
    total_correct = 0
    total_samples = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)

                # Collect file names from dataset.imgs list, using the current batch indices
                batch_start_index = total_samples
                for idx_in_batch in range(images.size(0)):
                    # dataset.imgs or samples holds tuples of (path, class_idx)
                    img_path, _ = test_loader.dataset.samples[
                        batch_start_index + idx_in_batch
                    ]
                    filename = os.path.basename(img_path)
                    prob_values = probs[idx_in_batch].cpu().tolist()
                    pred_label = class_names[preds[idx_in_batch].item()]
                    writer.writerow([filename] + prob_values + [pred_label])

                total_correct += preds.eq(labels).sum().item()
                total_samples += images.size(0)

    # Print accuracy if ground truth labels are meaningful (for test sets without labels this will simply produce 0)
    if total_samples > 0:
        accuracy = total_correct / total_samples
        print(f"Test accuracy (if labels are available): {accuracy:.4f}")
    print(f"Saved predictions to {csv_path}")


if __name__ == "__main__":
    main()
