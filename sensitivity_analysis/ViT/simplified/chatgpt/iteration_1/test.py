"""
test_swinv2_classifier.py
==========================

This script loads a fine‑tuned SwinV2 model saved by
``train_swinv2_classifier.py`` and evaluates it on a held‑out test set.  The
test directory must have the same folder structure as the training data:
each class corresponds to a subdirectory of ``test`` containing images.  The
script computes top‑1 accuracy and generates a classification report and
confusion matrix.  Predicted labels for each file are also written to a
CSV file in the output directory.

The Swin Transformer builds hierarchical feature maps by merging
image patches in deeper layers and applying self‑attention only within
local windows, which significantly reduces the computation cost【555662335336767†L67-L74】.  SwinV2
improves training stability and cross‑resolution transfer through a
residual‑post‑norm architecture, cosine attention, and log‑spaced
continuous position bias【555662335336767†L67-L80】.  These features make it a strong
backbone for fundus image classification.  The `timm` library used here
supports resizing patch and position embeddings when loading pre‑trained
weights【608117212520077†L166-L184】, enabling evaluation at custom resolutions if needed.

Usage example:

```
python test_swinv2_classifier.py \
    --data_dir /path/to/dataset/test \
    --checkpoint_path /path/to/output/best_model.pth \
    --output_dir /path/to/results
```

Dependencies
------------
This script requires PyTorch, torchvision, timm, tqdm and scikit‑learn.  You
can install the latter via:

```
pip install scikit‑learn
```

"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    classification_report = None
    confusion_matrix = None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for testing.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned SwinV2 classifier on a test dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing test data (with class subdirs)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth) produced by the training script",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where evaluation results and predictions will be saved",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini‑batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    return parser.parse_args()


def build_val_transform(img_size: int) -> transforms.Compose:
    """Build a transform pipeline for evaluation.

    Parameters
    ----------
    img_size : int
        Target input size for the model.

    Returns
    -------
    transforms.Compose
        Transformations applied to test images.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def prepare_dataloader(
    data_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int
) -> Tuple[DataLoader, List[str]]:
    """Prepare a DataLoader for the test set.

    Parameters
    ----------
    data_dir : str
        Root directory for test data containing subdirectories for each class.
    transform : callable
        Transformations applied to images.
    batch_size : int
        Batch size for DataLoader.
    num_workers : int
        Number of worker processes for data loading.

    Returns
    -------
    tuple
        (dataloader, class_names)
    """
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, dataset.classes


def load_checkpoint_metadata(checkpoint_path: str) -> Tuple[dict, int, str]:
    """Load metadata from a checkpoint without instantiating the model.

    The training script stores ``model_name``, ``img_size`` and
    ``class_indices_path`` alongside the model state dict.  This helper
    function returns these values so that the caller can decide how to
    recreate the model.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved checkpoint file (.pth).

    Returns
    -------
    tuple
        (state_dict, img_size, class_indices_path, model_name)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    img_size = checkpoint["img_size"]
    class_indices_path = checkpoint.get("class_indices_path")
    model_name = checkpoint.get("model_name")
    return state_dict, img_size, class_indices_path, model_name


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint metadata to determine model architecture and image size
    state_dict, img_size, class_indices_path, model_name = load_checkpoint_metadata(
        args.checkpoint_path
    )
    if class_indices_path is None or not os.path.isfile(class_indices_path):
        raise FileNotFoundError(
            "The checkpoint does not specify a valid path to class_indices.json."
        )
    # Load class mapping to determine number of classes
    with open(class_indices_path, "r", encoding="utf-8") as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}
    num_classes = len(idx_to_class)
    # Ensure that the model architecture used during training exists in the current timm installation.
    available_models = timm.list_models()
    if model_name not in available_models:
        raise RuntimeError(
            f"The model '{model_name}' saved in the checkpoint is not available in your timm installation. "
            "Please update timm to a newer version that includes this model before running the test script."
        )
    # Instantiate model using the recorded model_name and img_size.  The
    # classification head is sized according to the number of classes in the checkpoint.
    model = timm.create_model(
        model_name,
        pretrained=False,
        img_size=img_size,
        num_classes=num_classes,
    )
    # Load the saved weights.  The strict flag ensures that the architecture matches exactly.
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Prepare dataloader
    transform = build_val_transform(img_size)
    loader, class_names = prepare_dataloader(
        args.data_dir, transform, args.batch_size, args.num_workers
    )

    # Evaluate model
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_paths: List[str] = []
    # Iterate through test loader
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False)
        sample_index = 0
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            # Save file paths relative to data_dir for each item in the batch
            batch_samples = loader.dataset.samples[
                sample_index : sample_index + labels.size(0)
            ]
            all_paths.extend(
                [os.path.relpath(path, args.data_dir) for path, _ in batch_samples]
            )
            sample_index += labels.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    print(f"Test loss: {avg_loss:.4f} | Test accuracy: {acc:.4f}")

    # Save predictions to CSV
    pred_file = os.path.join(args.output_dir, "predictions.csv")
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write("image_path,true_label,pred_label\n")
        for path, true_idx, pred_idx in zip(all_paths, all_labels, all_preds):
            f.write(f"{path},{idx_to_class[true_idx]},{idx_to_class[pred_idx]}\n")

    # Generate classification report and confusion matrix if sklearn is available
    report_file = os.path.join(args.output_dir, "classification_report.txt")
    cm_file = os.path.join(args.output_dir, "confusion_matrix.npy")
    if classification_report is not None and confusion_matrix is not None:
        report = classification_report(
            all_labels,
            all_preds,
            target_names=[idx_to_class[i] for i in range(num_classes)],
        )
        cm = confusion_matrix(all_labels, all_preds)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        np.save(cm_file, cm)
        print(f"Classification report saved to {report_file}")
        print(f"Confusion matrix saved to {cm_file}")
    else:
        print(
            "sklearn is not installed; classification report and confusion matrix will not be generated."
        )

    print(f"Predictions saved to {pred_file}")


if __name__ == "__main__":
    main()
