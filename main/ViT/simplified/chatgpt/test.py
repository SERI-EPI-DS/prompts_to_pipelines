"""
test_swinv2.py
===============

This script evaluates a fine‑tuned Swin Transformer V2 model on a test
split and optionally writes the predicted labels for each image to a
CSV file.  The expected dataset layout matches that used during
training: ``root/train``, ``root/val`` and ``root/test`` with
class‑named subfolders.  The checkpoint file produced by
``train_swinv2.py`` must be provided via ``--checkpoint``.  When
executed, the script loads the model, prepares the appropriate data
transformations and reports the overall accuracy.  A CSV file is
generated mapping image paths to predicted class names when
``--output_csv`` is given.

Like the training script, this module relies on ``timm`` to create
the model backbone.  Swin V2 models apply windowed self‑attention to
image patches and shift these windows to capture global context
efficiently【38878464741457†L120-L129】.  The particular pre‑trained
checkpoint ``swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`` was
initially trained on ImageNet‑22k and then fine‑tuned on
ImageNet‑1k【739641659479825†L55-L59】; however you may substitute any
compatible model name supported by ``timm``.
"""

import argparse
import os
import csv
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
from timm.data import resolve_model_data_config, create_transform


def parse_args() -> argparse.Namespace:
    """Parse evaluation options from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fine‑tuned Swin V2 model on a test dataset and optionally "
            "write predictions to a CSV file."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset root containing a 'test' subdirectory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pth checkpoint file produced during training.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
        help="Model architecture to instantiate (must match the training script).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes used for loading data.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device on which to run evaluation ('cuda' or 'cpu'). If unspecified, uses CUDA when available.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to output a CSV with columns 'image_path' and 'predicted_label'.",
    )
    return parser.parse_args()


def create_test_loader(
    root_dir: str,
    transform,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, List[str], List[Tuple[str, int]]]:
    """Create a DataLoader for the test set.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root containing a 'test' subdirectory.
    transform : callable
        Transformation applied to test images.
    batch_size : int
        Number of images per batch.
    num_workers : int
        Number of worker processes.

    Returns
    -------
    Tuple[DataLoader, List[str], List[Tuple[str, int]]]
        The DataLoader, list of class names, and the underlying list of
        samples (path, label) used by the dataset.  The samples list is
        required for mapping predictions back to file paths.
    """
    test_path = os.path.join(root_dir, "test")
    if not os.path.isdir(test_path):
        raise FileNotFoundError(
            "Couldn't find 'test' folder inside the provided data directory"
        )
    dataset = ImageFolder(test_path)
    # apply transform lazily; dataset.transform will be called by the loader
    dataset.transform = transform
    class_names = dataset.classes
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # dataset.samples is a list of (path, label) pairs used internally
    return loader, class_names, dataset.samples


def main() -> None:
    args = parse_args()
    # Determine device
    device_str = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Instantiate a temporary model to retrieve the data config
    tmp_model = timm.create_model(args.model_name, pretrained=True)
    data_config = resolve_model_data_config(tmp_model)
    transform = create_transform(**data_config, is_training=False)

    # Create test loader and get samples list for mapping
    loader, class_names, samples = create_test_loader(
        args.data_dir, transform, args.batch_size, args.num_workers
    )
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    # Initialize model and load checkpoint
    model = timm.create_model(
        args.model_name, pretrained=False, num_classes=num_classes
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Perform inference
    total = 0
    correct = 0
    predictions: List[str] = []
    with torch.no_grad():
        # Keep track of the index offset into samples list
        sample_index = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            # Convert predictions to class names
            predictions_batch = [class_names[p.item()] for p in preds]
            predictions.extend(predictions_batch)
            # Accumulate accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            sample_index += labels.size(0)
    accuracy = correct / max(total, 1)
    print(f"Test accuracy: {accuracy:.4f}")

    # Write predictions to CSV if requested
    if args.output_csv is not None:
        csv_path = args.output_csv
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "predicted_label"])
            for (path, _), pred_label in zip(samples, predictions):
                writer.writerow([path, pred_label])
        print(f"Saved predictions to {csv_path}")


if __name__ == "__main__":
    main()
