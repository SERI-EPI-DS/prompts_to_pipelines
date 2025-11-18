#!/usr/bin/env python3
"""
Evaluate a fine‑tuned ConvNeXt‑L model on a held‑out test set and write
predictions to a CSV file.

This script loads a model checkpoint produced by ``train.py``, restores the
weights and class names, runs inference over the test split of a dataset
structured for ``torchvision.datasets.ImageFolder``, and writes a CSV report.
Each row of the output file contains the image filename, the predicted
probabilities for each class, and the final predicted class.

Example usage:

```
python test.py --data_dir /path/to/data --weights_path /path/to/best_model.pth \
    --results_dir ./results
```

The CSV file will be saved to ``results_dir/test_predictions.csv``.
"""

import argparse
import csv
import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that returns (image, label, path) instead of (image, label)."""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        # Get original tuple (image, class)
        image, label = super().__getitem__(index)
        # Path string
        path, _ = self.samples[index]
        return image, label, path


def load_model(weights_path: str, device: torch.device) -> Tuple[nn.Module, List[str]]:
    """
    Restore a ConvNeXt‑L model from a checkpoint and return it along with the
    class names.

    Args:
        weights_path: Path to the ``.pth`` file containing the model state
            dictionary saved by ``train.py``.
        device: Torch device to load the model onto.

    Returns:
        model: The ConvNeXt‑L model with weights loaded.
        class_names: A list of class names in the order used during training.
    """
    checkpoint = torch.load(weights_path, map_location=device)
    class_names = checkpoint.get("class_names")
    if class_names is None:
        raise ValueError(
            "The checkpoint does not contain class names; ensure it was produced by train.py"
        )
    # Instantiate a new model and adjust the classifier
    model = models.convnext_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned ConvNeXt‑L model on a test set and save predictions."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing the test folder.",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to write the CSV predictions file.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading.",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model and class names
    model, class_names = load_model(args.weights_path, device)
    num_classes = len(class_names)

    # Define the same transforms used for validation/inference.  According to
    # TorchVision’s ConvNeXt Large weights, inference preprocessing resizes
    # images to 232 and centre crops to 224, then normalizes using the
    # ImageNet mean and std【436985033393033†L160-L167】.
    test_transforms = transforms.Compose(
        [
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_dir = os.path.join(args.data_dir, "test")
    test_dataset = ImageFolderWithPaths(test_dir, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Write predictions to CSV
    csv_path = os.path.join(args.results_dir, "test_predictions.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Header: file_name, score_<class> per class, pred_class
        header = (
            ["file_name"] + [f"score_{cls}" for cls in class_names] + ["pred_class"]
        )
        writer.writerow(header)
        # Iterate over test samples
        for batch in test_loader:
            inputs, _, paths = batch
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
            # For each image in the batch
            for i in range(inputs.size(0)):
                file_name = os.path.basename(paths[i])
                prob_vector = probs[i].tolist()
                pred_idx = int(torch.argmax(probs[i]).item())
                pred_class = class_names[pred_idx]
                row = [file_name] + [f"{p:.6f}" for p in prob_vector] + [pred_class]
                writer.writerow(row)
    print(f"Saved predictions to {csv_path}")


if __name__ == "__main__":
    main()
