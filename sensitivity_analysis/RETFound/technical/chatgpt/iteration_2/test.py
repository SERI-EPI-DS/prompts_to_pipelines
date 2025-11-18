#!/usr/bin/env python3
"""
Evaluate a fine‑tuned RETFound classifier on a held‑out test set.

This script loads the best model weights saved during training, applies
the same preprocessing used for validation, performs inference on all
images in the `test` folder, and writes a CSV containing the file
name, predicted probabilities for each class, and the final predicted
class label.  The class order matches the training dataset order,
which is stored inside the checkpoint file by `train.py`.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for testing."""
    parser = argparse.ArgumentParser(description="Test RETFound ViT‑L classifier")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "data" / "test"),
        help="Directory containing the test images organised by class.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent
            / "project"
            / "results"
            / "best_model.pth"
        ),
        help="Path to the trained model checkpoint (best_model.pth).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent
            / "project"
            / "results"
            / "test_predictions.csv"
        ),
        help="Location to write the CSV file with predictions.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for loading data.",
    )
    return parser.parse_args()


def build_transform(input_size: int) -> transforms.Compose:
    """Create the deterministic test transform matching validation settings."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_model(model_path: str) -> Tuple[nn.Module, List[str], int, int]:
    """Load the trained model checkpoint and return the model and metadata.

    Parameters
    ----------
    model_path : str
        Path to the checkpoint saved by `train.py`.

    Returns
    -------
    model : nn.Module
        The vision transformer with weights restored.
    class_names : List[str]
        A list of class names in the order used during training.
    num_classes : int
        Number of classes.
    input_size : int
        Image size used during training, stored in the checkpoint.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint '{model_path}' not found")
    checkpoint = torch.load(model_path, map_location="cpu")
    class_names: List[str] = checkpoint.get("class_names")  # type: ignore
    input_size: int = checkpoint.get("input_size", 224)  # type: ignore
    if class_names is None:
        # Fall back to reading class names from the directory structure
        raise ValueError(
            "Class names not found in checkpoint. Re‑train using train.py so that class_names are saved."
        )
    num_classes = len(class_names)
    # Import RETFound model definition
    script_dir = Path(__file__).resolve().parent
    retfound_dir = script_dir.parent.parent / "RETFound"
    sys.path.insert(0, str(retfound_dir))
    try:
        from models_vit import RETFound_mae  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import RETFound models. Ensure the RETFound repository "
            "is present adjacent to the project directory."
        ) from exc
    model = RETFound_mae(
        img_size=input_size,
        num_classes=num_classes,
        drop_path_rate=0.0,
        global_pool=True,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Remove mismatched classification head parameters if necessary
    for key in ["head.weight", "head.bias"]:
        if key in state_dict and state_dict[key].shape[0] != num_classes:
            del state_dict[key]
    _ = model.load_state_dict(state_dict, strict=False)
    # Ensure the classifier head matches num_classes
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    return model, class_names, num_classes, input_size


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    output_csv: str,
) -> None:
    """Run inference on the test loader and write predictions to CSV."""
    model.eval()
    results: List[List[str]] = []
    # Keep track of sample index to correctly retrieve file names from dataset.samples
    sample_index = 0
    dataset = loader.dataset  # type: ignore
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        probs_cpu = probs.cpu().numpy()
        preds = probs_cpu.argmax(axis=1)
        batch_size = images.size(0)
        for i in range(batch_size):
            # dataset.samples holds (path, target) for every image in the order
            # they will be loaded by the DataLoader
            img_path, _ = dataset.samples[sample_index + i]
            filename = os.path.basename(img_path)
            row = [filename]
            # Append probability scores in the same order as class_names
            row.extend([float(prob) for prob in probs_cpu[i].tolist()])
            row.append(class_names[preds[i]])
            results.append(row)
        sample_index += batch_size
    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    header = ["file_name"] + [f"score_{cls}" for cls in class_names] + ["prediction"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)


def main() -> None:
    args = parse_args()
    # Load model and metadata
    model, class_names, num_classes, input_size = load_model(args.model_path)
    # Build transforms and dataset
    transform = build_transform(input_size)
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"Test directory '{args.data_path}' not found")
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # Send model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Run inference and write CSV
    run_inference(model, loader, class_names, device, args.output_csv)
    print(f"Saved prediction CSV to '{args.output_csv}'.")


if __name__ == "__main__":
    main()
