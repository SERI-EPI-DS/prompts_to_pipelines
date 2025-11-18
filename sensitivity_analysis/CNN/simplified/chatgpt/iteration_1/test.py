"""
test_convnext.py
================

This script evaluates a fine‑tuned ConvNeXt model on a test dataset and writes
per‑image predictions to a CSV file.  It expects the dataset to be organised
in a ``test`` directory containing subdirectories for each class (if labels
are available).  During evaluation, the script applies the same input
transformations that were used for training (centre cropping and normalisation)
via ``timm.data.create_transform``【670582884494607†L170-L243】【587683307028815†L264-L277】.  If a ``class_mapping.json`` file
is supplied, the script will use this to map numeric indices back to human
readable class names.

Usage example::

    python test_convnext.py \
        --data-dir /path/to/dataset \
        --model-path ./results/best_model.pth \
        --class-mapping ./results/class_mapping.json \
        --output-csv ./results/test_predictions.csv

Requirements::

    pip install timm torch torchvision pandas sklearn

"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


def prepare_test_loader(
    data_dir: Path,
    model_name: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, Dict[int, str]]:
    """Prepare a dataloader for the test dataset.

    Parameters
    ----------
    data_dir : Path
        Path to the dataset root containing a ``test`` subdirectory.
    model_name : str
        Name of the ConvNeXt variant (e.g. ``convnext_large.fb_in22k_ft_in1k``) to
        determine the correct input resolution and normalisation statistics.
    batch_size : int
        Number of images per batch.
    num_workers : int
        Number of worker processes for loading data.

    Returns
    -------
    Tuple[DataLoader, Dict[int, str]]
        The test dataloader and a mapping from class indices to class names.  If
        the ``test`` directory does not contain subdirectories (i.e. labels are
        unavailable), the mapping will be empty.
    """
    # Create a temporary model to resolve the data config
    tmp_model = timm.create_model(model_name, pretrained=True, num_classes=0)
    data_config = timm.data.resolve_data_config({}, model=tmp_model)
    test_transform = timm.data.create_transform(**data_config, is_training=False)

    test_dir = data_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find 'test' directory under {data_dir}. "
            "Ensure the dataset includes a test subset."
        )

    # If there are subdirectories under test_dir, ImageFolder will assign labels
    # accordingly.  Otherwise, all images will fall under a single generic
    # class.  It's fine for inference as we only need to read the image
    # contents.
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    return test_loader, idx_to_class


def load_model(
    model_name: str,
    num_classes: int,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load a fine‑tuned ConvNeXt model from disk.

    Parameters
    ----------
    model_name : str
        Name of the ConvNeXt variant (e.g. ``convnext_large.fb_in22k_ft_in1k``).
    num_classes : int
        Number of output classes.  Must match the value used during training.
    checkpoint_path : Path
        Path to the saved model checkpoint (.pth file) produced by the training
        script.
    device : torch.device
        Device onto which the model and weights should be loaded.

    Returns
    -------
    nn.Module
        The loaded model in evaluation mode.
    """
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[str], List[int], List[int], List[float]]:
    """Generate predictions for all images in the dataloader.

    Returns lists containing the image file paths, true labels (if available),
    predicted labels and confidence scores of the maximum probability.
    """
    all_paths: List[str] = []
    all_true_labels: List[int] = []
    all_pred_labels: List[int] = []
    all_confidences: List[float] = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", unit="batch"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = softmax(outputs)
            confidences, preds = torch.max(probs, 1)
            all_confidences.extend(confidences.cpu().tolist())
            all_pred_labels.extend(preds.cpu().tolist())
            all_true_labels.extend(targets.tolist())
    # Extract file paths from dataset (ImageFolder stores them in samples attribute)
    # This only works because the DataLoader preserves the order of samples
    all_paths = [path for path, _ in dataloader.dataset.samples]
    return all_paths, all_true_labels, all_pred_labels, all_confidences


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned ConvNeXt model on a test dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing a test subfolder",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (.pth file)",
    )
    parser.add_argument(
        "--class-mapping",
        type=str,
        default=None,
        help="Optional path to a JSON file mapping class indices to labels",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="convnext_large.fb_in22k_ft_in1k",
        help="Name of the ConvNeXt variant used during training",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="predictions.csv",
        help="CSV file to write predictions and probabilities",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    mapping_path = Path(args.class_mapping) if args.class_mapping else None
    output_csv = Path(args.output_csv)

    # Prepare dataloader
    test_loader, idx_to_class_dataset = prepare_test_loader(
        data_dir=data_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    # Determine the class mapping.  Prefer the supplied mapping file if
    # available because it reflects the training set order.  Otherwise, fall
    # back to the dataset's inferred mapping.
    if mapping_path and mapping_path.exists():
        with open(mapping_path, "r") as f:
            idx_to_class = {int(k): v for k, v in json.load(f).items()}
    else:
        idx_to_class = idx_to_class_dataset

    num_classes = len(idx_to_class)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    model = load_model(
        model_name=args.model_name,
        num_classes=num_classes,
        checkpoint_path=model_path,
        device=device,
    )

    # Generate predictions
    file_paths, true_labels, pred_labels, confidences = predict(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    # Map integer labels to class names where possible
    true_labels_str = [idx_to_class.get(lbl, "") for lbl in true_labels]
    pred_labels_str = [idx_to_class.get(lbl, "") for lbl in pred_labels]

    # Build a DataFrame for export
    df = pd.DataFrame(
        {
            "image_path": file_paths,
            "true_label": true_labels_str,
            "predicted_label": pred_labels_str,
            "confidence": confidences,
        }
    )

    # Compute overall accuracy if true labels are available
    if any(true_labels_str):
        correct = sum(p == t for p, t in zip(pred_labels_str, true_labels_str))
        accuracy = correct / len(pred_labels_str)
        print(f"Test accuracy: {accuracy:.4f}")
    else:
        print(
            "No true labels found in the test set. Only predictions will be recorded."
        )

    # Write predictions to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    main()
