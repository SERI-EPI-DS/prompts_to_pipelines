"""
test.py
--------

This script evaluates a fine‑tuned Swin‑V2‑B model on a held‑out test set and
exports predictions to a CSV file.  Each row in the CSV contains the image
filename, the model's probability for each class, and the final predicted
label.  The script reads the class names from a ``classes.txt`` file saved
during training to ensure consistency in class ordering between training and
testing.

Example usage:

```
python test.py --data-dir /path/to/data \
               --model-path /path/to/results/best_model.pt \
               --output-dir /path/to/results
```

The data directory must contain a ``test`` subfolder organised in the
``ImageFolder`` format (although only the file paths are used – the labels
within the test folder are ignored).  The CSV will be saved into the
specified output directory under the name ``test_predictions.csv``.
"""

import argparse
import csv
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for testing configuration."""
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned Swin‑V2‑B model on a test set"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing the 'test' subfolder",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the checkpoint (.pt) file with the trained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the CSV of predictions",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Mini‑batch size for inference"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    return parser.parse_args()


def load_classes(output_dir: str) -> List[str]:
    """
    Load class names from the ``classes.txt`` file saved during training.  If
    the file is not found, this function returns an empty list.

    Args:
        output_dir: Directory where ``classes.txt`` is expected to reside.

    Returns:
        A list of class names in the order used during training.
    """
    classes_path = os.path.join(output_dir, "classes.txt")
    if not os.path.isfile(classes_path):
        return []
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes


def create_test_loader(
    data_dir: str, batch_size: int, num_workers: int
) -> Tuple[DataLoader, List[str]]:
    """
    Create a DataLoader for the test dataset.  The labels in the test folder
    are ignored; instead, the dataset is used only to iterate over the
    images.  Normalisation parameters follow those used during training.

    Returns:
        A tuple (loader, file_paths) where ``loader`` is the DataLoader and
        ``file_paths`` is a list mapping dataset indices to full image paths.
    """
    # Use default weights to obtain normalisation stats
    weights = models.Swin_V2_B_Weights.DEFAULT
    # Fallback to ImageNet mean/std if not provided in meta
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    try:
        meta = weights.meta  # type: ignore[attr-defined]
        mean = meta.get("mean", default_mean)
        std = meta.get("std", default_std)
    except Exception:
        mean, std = default_mean, default_std

    transform = transforms.Compose(
        [
            transforms.Resize(
                size=272, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_dir = os.path.join(data_dir, "test")
    # When using ImageFolder for test data, the labels are inferred from folder
    # names; however, the actual labels are irrelevant for this script.  We
    # simply use the ``samples`` property later to retrieve file paths.
    dataset = datasets.ImageFolder(test_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Extract file paths in the same order as dataset indices
    file_paths = [s[0] for s in dataset.samples]
    return loader, file_paths


def build_model(num_classes: int) -> torch.nn.Module:
    """Recreate the Swin‑V2‑B model architecture for inference."""
    weights = models.Swin_V2_B_Weights.DEFAULT
    model = models.swin_v2_b(weights=weights)
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def perform_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    classes: List[str],
    file_paths: List[str],
    output_csv_path: str,
) -> None:
    """
    Run the model on the test dataloader, compute probabilities, determine
    predicted labels and write results to a CSV file.

    Args:
        model: Trained neural network.
        dataloader: DataLoader for the test images.
        device: Device on which the model resides.
        classes: List of class names in the order used during training.
        file_paths: Full paths to the images corresponding to dataloader indices.
        output_csv_path: Path to save the resulting CSV file.
    """
    model.eval()

    num_classes = len(classes)
    # Prepare header for CSV: filename + probability columns + predicted label
    header = ["filename"] + [f"{cls}_prob" for cls in classes] + ["predicted_label"]

    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        offset = 0  # running index to map batch outputs to file_paths
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            batch_size = probs.shape[0]

            # For each sample in the batch, write the probabilities and label
            for i in range(batch_size):
                idx = offset + i
                file_path = file_paths[idx]
                filename = os.path.basename(file_path)
                prob_list = probs[i].tolist()
                # Determine predicted label index
                pred_idx = int(probs[i].argmax())
                pred_label = classes[pred_idx]
                writer.writerow([filename] + prob_list + [pred_label])
            offset += batch_size


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load class names from training output directory.  If the file is not
    # present, raise an error because class ordering would be ambiguous.
    classes = load_classes(args.output_dir)
    if not classes:
        raise FileNotFoundError(
            f"Expected classes.txt in {args.output_dir}.  "
            "Make sure to run training first and specify the same output directory."
        )

    # Create test loader and file path mapping
    test_loader, file_paths = create_test_loader(
        args.data_dir, args.batch_size, args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the model architecture and load the trained weights
    model = build_model(num_classes=len(classes))
    checkpoint = torch.load(args.model_path, map_location=device)
    # Check if the checkpoint is a simple state_dict or a dict with extra keys
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Perform inference and write CSV
    output_csv_path = os.path.join(args.output_dir, "test_predictions.csv")
    perform_inference(model, test_loader, device, classes, file_paths, output_csv_path)
    print(f"Inference complete. Results saved to {output_csv_path}")


if __name__ == "__main__":
    main()
