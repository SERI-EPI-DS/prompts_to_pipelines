#!/usr/bin/env python3
"""
test_swinv2.py
===============

This script evaluates a fine‑tuned Swin Transformer V2 classifier on a test
dataset and outputs performance metrics as well as per‑image predictions.

The test dataset must be organised in the same way as the training and
validation sets: a folder containing subfolders, one per class. You can
override the name of the test subfolder with the ``--test-folder`` argument.

It loads a checkpoint saved by ``train_swinv2.py`` and recreates the model
accordingly. Predictions and ground truth labels are written to a CSV file so
that you can analyse them later.

Usage example:

    python test_swinv2.py \
        --data-dir /path/to/my_dataset \
        --test-folder test \
        --checkpoint ./models/best_model_epoch10.pth \
        --output-file results.csv

Requirements:
* PyTorch ≥ 1.10
* timm ≥ 0.6.12
* pandas (optional, for saving CSV; fallback to Python CSV module)

"""

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for this script. Install with `pip install timm`."
    ) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned SwinV2 model on a test set."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory of the dataset containing the test folder.",
    )
    parser.add_argument(
        "--test-folder",
        type=str,
        default="test",
        help="Name of the subfolder inside data_dir containing test images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth) saved during training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for testing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="test_results.csv",
        help="CSV file to write predictions. Each row contains filename, true label and predicted label.",
    )
    return parser.parse_args()


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, List[str], int, str]:
    """Load a model checkpoint and reconstruct the model.

    Returns the model, class names, image size and model name.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("classes")
    if class_names is None:
        raise KeyError(
            "The checkpoint does not contain class names. Ensure you saved the model via train_swinv2.py."
        )
    img_size = checkpoint.get("img_size")
    model_name = checkpoint.get("model_name")
    if img_size is None or model_name is None:
        raise KeyError(
            "The checkpoint is missing required metadata (img_size/model_name)."
        )
    num_classes = len(class_names)
    # Recreate model
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        img_size=img_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, img_size, model_name


def create_dataloader(
    data_dir: str,
    test_folder: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Create a test DataLoader with deterministic preprocessing."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transforms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_path = os.path.join(data_dir, test_folder)
    if not os.path.isdir(test_path):
        raise FileNotFoundError(f"Test folder not found: {test_path}")
    dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, List[Tuple[str, str, str]]]:
    """Evaluate the model and collect predictions.

    Returns overall accuracy and a list of predictions (filename, true_label, pred_label).
    """
    running_corrects = 0
    total = 0
    predictions = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)
        # Record predictions
        batch_paths = [
            dataloader.dataset.samples[i][0]
            for i in range(total - labels.size(0), total)
        ]
        for path, true_idx, pred_idx in zip(
            batch_paths, labels.cpu().tolist(), preds.cpu().tolist()
        ):
            filename = os.path.basename(path)
            true_label = dataloader.dataset.classes[true_idx]
            pred_label = dataloader.dataset.classes[pred_idx]
            predictions.append((filename, true_label, pred_label))
    accuracy = running_corrects / total if total > 0 else 0.0
    return accuracy, predictions


def save_predictions(
    predictions: List[Tuple[str, str, str]],
    output_file: str,
) -> None:
    """Save predictions to a CSV file. Columns: filename, true_label, predicted_label."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        df = pd.DataFrame(
            predictions, columns=["filename", "true_label", "predicted_label"]
        )
        df.to_csv(output_path, index=False)
    except ImportError:
        # Fallback to csv module
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "true_label", "predicted_label"])
            for row in predictions:
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Evaluating checkpoint: %s", args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model and metadata from checkpoint
    model, class_names, img_size, model_name = load_checkpoint(args.checkpoint, device)
    logging.info(
        f"Loaded model {model_name} with image size {img_size} and {len(class_names)} classes."
    )

    # Create dataloader
    test_loader = create_dataloader(
        data_dir=args.data_dir,
        test_folder=args.test_folder,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    # Evaluate
    accuracy, predictions = evaluate(model, test_loader, device)
    logging.info(f"Test accuracy: {accuracy:.4f}")

    # Save predictions
    save_predictions(predictions, args.output_file)
    logging.info(f"Predictions written to {args.output_file}")


if __name__ == "__main__":
    main()
