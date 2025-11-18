"""
Testing script for a fine‑tuned ConvNeXt‑L classifier.

This script loads a trained model checkpoint together with the saved
``class_to_idx`` mapping and produces predictions on a held‑out test set.
Images are expected to be organised in a ``test`` subdirectory under
``data_root`` in the same class‑based folder structure used for training.
Output consists of a CSV file where each row contains the input filename,
the model's predicted probability for each class, and the final predicted
class label.  The CSV is saved into the user specified results directory.

Example usage:

.. code-block:: bash

    python test.py \
        --data_root /path/to/data \
        --model_path /path/to/results/best_model.pth \
        --class_map /path/to/results/class_to_idx.json \
        --output_dir /path/to/results

The script assumes that PyTorch and TorchVision are installed and that a
CUDA capable GPU is available; however, it will fall back to CPU
execution if a GPU is not detected.
"""

import argparse
import csv
import json
import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode, autoaugment


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for testing.

    Returns
    -------
    argparse.Namespace
        A namespace containing ``data_root``, ``model_path``, ``class_map`` and
        ``output_dir`` among other optional parameters.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned ConvNeXt‑L model on a test set"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=(
            "Path to the root of the dataset.  Must contain a 'test' subdirectory "
            "with the same class based folder structure as used for training."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model checkpoint (.pth file) produced during training.",
    )
    parser.add_argument(
        "--class_map",
        type=str,
        required=True,
        help="Path to the JSON file containing the class_to_idx mapping saved during training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the output CSV of predictions will be written.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).  Adjust based on GPU memory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for loading data (default: 4)",
    )
    return parser.parse_args()


def build_transforms(img_size: int = 224) -> transforms.Compose:
    """Construct the deterministic transform pipeline for inference.

    Parameters
    ----------
    img_size : int, optional
        Target crop size for ConvNeXt (default: 224).

    Returns
    -------
    transforms.Compose
        A composition of resizing, centre cropping, tensor conversion and
        normalisation operations.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize(
                int(img_size * 256 / 224),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def create_model(num_classes: int) -> torch.nn.Module:
    """Recreate the ConvNeXt‑Large model architecture for inference.

    Parameters
    ----------
    num_classes : int
        Number of classes to configure the classifier for.

    Returns
    -------
    torch.nn.Module
        A ConvNeXt‑Large model with its final linear layer sized for
        ``num_classes`` and pre‑trained weights loaded where available.
    """
    try:
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
    except AttributeError:
        weights = None
    model = models.convnext_large(weights=weights)
    # Replace the classification head with the correct number of outputs.
    if hasattr(model, "classifier") and isinstance(
        model.classifier, torch.nn.Sequential
    ):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    else:
        # Fallback for future changes in torchvision implementation.
        in_features = model.get_classifier().in_features  # type: ignore[attr-defined]
        model.fc = torch.nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]
    return model


def load_class_mapping(path: str) -> Dict[str, int]:
    """Load the class_to_idx mapping from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file saved during training.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping class names to numerical indices.
    """
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return mapping


def main() -> None:
    args = parse_args()

    # Create the results directory if it does not exist.
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv_path = os.path.join(args.output_dir, "test_predictions.csv")

    # Load class mapping and compute the reverse mapping.  We rely on this
    # mapping to interpret the model outputs consistently with the training
    # procedure.  The order of classes in this mapping defines the order of
    # probability columns in the output CSV.
    class_to_idx = load_class_mapping(args.class_map)
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # Build the inference transform pipeline.
    test_transform = build_transforms(img_size=224)

    # Construct the test dataset and loader.  We use ImageFolder to load
    # images from a directory tree; the labels it assigns are irrelevant
    # during testing but must be present in the structure.  The ``samples``
    # attribute contains the list of file paths in the order they will be
    # returned by the loader.
    test_dir = os.path.join(args.data_root, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Could not find test directory '{test_dir}'.  Ensure the data root contains a 'test' folder."
        )
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build model architecture and load the saved weights.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes)
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        # Attempt to load the state dict directly if the file is a bare
        # state dict.  ``strict=False`` allows missing keys if the final
        # classifier layer sizes differ.
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    # Prepare for writing results.  Column names start with 'filename',
    # followed by one column per class (probabilities) and end with
    # 'predicted_class'.
    # The order of class probability columns follows the ordering of
    # ``idx_to_class`` sorted by index to be deterministic.
    class_indices_sorted = [
        idx for idx, _ in sorted(idx_to_class.items(), key=lambda x: x[0])
    ]
    class_names_sorted = [idx_to_class[idx] for idx in class_indices_sorted]
    header = (
        ["filename"]
        + [f"score_{cls_name}" for cls_name in class_names_sorted]
        + ["predicted_class"]
    )

    # Track the global sample index to correctly reference filenames from
    # ``test_dataset.samples`` while iterating over the loader.
    sample_idx = 0

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        # Disable gradient calculation for inference.
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
                # Convert logits to probabilities.  Softmax yields values in [0, 1] that sum to 1.
                probs = F.softmax(outputs, dim=1).cpu()
                batch_size = inputs.size(0)
                for i in range(batch_size):
                    # Extract the file path for the current sample.  The ``samples`` list
                    # contains (path, class) tuples in the same order as items are
                    # returned by the loader.  ``sample_idx`` is incremented as we
                    # progress through the dataset.
                    path, _label = test_dataset.samples[sample_idx]
                    sample_idx += 1
                    # Extract the probability vector for this sample and convert
                    # to a list of floats in the sorted class index order.
                    prob_vec = probs[i]
                    prob_list = [prob_vec[idx].item() for idx in class_indices_sorted]
                    predicted_idx = int(torch.argmax(prob_vec).item())
                    predicted_class = idx_to_class[predicted_idx]
                    # Write the row: filename relative to test root, probabilities and prediction.
                    relative_path = os.path.relpath(path, test_dir)
                    writer.writerow([relative_path] + prob_list + [predicted_class])

    print(f"Saved predictions to {output_csv_path}")


if __name__ == "__main__":
    main()
