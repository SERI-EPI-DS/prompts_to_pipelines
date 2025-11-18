"""
test_convnext.py
=================

This script evaluates a fine‑tuned ConvNeXt‑L model on a hold‑out test set.
The dataset should follow the same directory structure as the one used for
training: a root folder containing a ``test`` subfolder and inside it
subdirectories for each class【51527509087205†L524-L536】.  Each image in the
test set will be classified, and the results will be written to disk in
CSV and text formats.  The script also reports the overall accuracy,
per‑class metrics and a confusion matrix.

The ConvNeXt model comes from the paper *“A ConvNet for the 2020s”*,
where the authors modernize a ResNet and show that the resulting
ConvNeXt family competes favorably with Vision Transformers【750288342696214†L23-L28】.
The pre‑trained weights available in ``torchvision`` include an
associated set of inference transforms that resize images and perform
normalization using ImageNet statistics【297744549850868†L160-L167】.  We
adopt the same normalization values here—mean `[0.485, 0.456, 0.406]`
and standard deviation `[0.229, 0.224, 0.225]`【668952771882936†L710-L721】—to
ensure compatibility with the fine‑tuned model.

Example usage:

.. code-block:: bash

   python test_convnext.py \
       --data-dir /path/to/dataset \
       --model-path /path/to/best_model.pth \
       --output-dir ./results

Dependencies: ensure that ``torch``, ``torchvision``, ``numpy``,
``scikit‑learn`` and ``pandas`` are available in your environment.
"""

import argparse
import os
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def build_test_loader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    input_size: int = 224,
) -> Tuple[DataLoader, List[str], List[Tuple[str, int]]]:
    """Construct a DataLoader for the test set.

    Args:
        data_dir: Root directory containing a ``test`` subfolder.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes to load data.
        input_size: Side length of the input crop.

    Returns:
        A tuple containing the DataLoader, class names and the list of samples
        (file path and ground truth label index) in the test set.
    """
    test_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory '{test_dir}' does not exist")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transforms = transforms.Compose(
        [
            transforms.Resize(
                int(input_size * 1.12)
            ),  # Resize slightly larger before crop
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    class_names = test_dataset.classes
    # We'll store the list of (file_path, label) for later
    samples = test_dataset.samples  # List of tuples (path, class_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader, class_names, samples


def load_model(model_path: str, num_classes: int) -> nn.Module:
    """Load a ConvNeXt‑L model and restore weights from the given path.

    Args:
        model_path: Path to the saved model weights (state_dict).
        num_classes: Number of output classes (derived from the test set).

    Returns:
        A PyTorch model ready for evaluation.
    """
    # Initialize model with the same architecture as training
    weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
    model = models.convnext_large(weights=weights)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last_layer = model.classifier[-1]
        if isinstance(last_layer, nn.Linear):
            in_features = last_layer.in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unexpected classifier structure in ConvNeXt model")
    else:
        raise RuntimeError("ConvNeXt model does not have a classifier attribute")
    # Load weights
    state_dict = torch.load(model_path, map_location="cpu")
    # Some checkpoints may save only the model_state_dict; handle both cases
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    samples: List[Tuple[str, int]],
    output_dir: str,
) -> None:
    """Run inference on the test set and write results to disk.

    Args:
        model: The trained model.
        dataloader: DataLoader for the test set.
        device: Device to perform inference on.
        class_names: List mapping class indices to class names.
        samples: List of tuples (file_path, ground_truth_label).
        output_dir: Directory in which to save the prediction results.
    """
    model.eval()
    model.to(device)

    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().tolist())

    # Compute overall accuracy
    acc = accuracy_score(all_targets, all_preds)
    print(f"[INFO] Test accuracy: {acc:.4f}")

    # Classification report
    report = classification_report(all_targets, all_preds, target_names=class_names)
    print("[INFO] Classification report:\n", report)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    # Save confusion matrix as CSV for easy analysis
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "classification_report.txt")
    cm_path = os.path.join(output_dir, "confusion_matrix.csv")
    preds_path = os.path.join(output_dir, "predictions.csv")

    with open(report_path, "w") as f:
        f.write(f"Test accuracy: {acc:.4f}\n\n")
        f.write(report)

    cm_df.to_csv(cm_path)

    # Prepare per‑image predictions
    # Flatten probabilities and predictions lists
    results = []
    for (filepath, target), pred, prob_array in zip(samples, all_preds, all_probs):
        filename = os.path.basename(filepath)
        true_label = class_names[target]
        pred_label = class_names[pred]
        confidence = float(max(prob_array))
        results.append(
            {
                "filename": filename,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
            }
        )
    results_df = pd.DataFrame(results)
    results_df.to_csv(preds_path, index=False)
    print(f"[INFO] Saved predictions to {preds_path}")
    print(f"[INFO] Saved classification report to {report_path}")
    print(f"[INFO] Saved confusion matrix to {cm_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned ConvNeXt‑L classifier on a test set"
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
        help="Path to the saved model weights (.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size (must match training)",
    )
    args = parser.parse_args()

    test_loader, class_names, samples = build_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
    )

    num_classes = len(class_names)
    model = load_model(args.model_path, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        samples=samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
