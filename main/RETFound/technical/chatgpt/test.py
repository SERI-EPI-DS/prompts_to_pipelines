"""
Testing script for RETFound MAE fine‑tuned models.

This script loads a trained model and evaluates it on a held‑out test set.  The
predicted class probabilities and final class labels for each image are saved
to a CSV file.  The expected directory structure matches the training
script: a ``test`` folder under the specified data root containing one
sub‑directory per class.

Usage example:

```
python test.py \
    --data_root /path/to/dataset \
    --weights /path/to/finetuned_model.pth \
    --output_dir /path/to/save/results \
    --batch_size 64
```

The resulting CSV will be named ``test_predictions.csv`` and placed inside
``output_dir``.  Each row contains the filename, probabilities for each class
and the predicted class label.

Reference: the RETFound MAE model uses a Vision Transformer architecture with
patch size 16, embedding dimension 1024 and depth 24【463435877362505†L259-L279】.
"""

import argparse
import csv
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
Attempt to import the RETFound Vision Transformer.  As with the training
script, this block searches for a local ``RETFound`` directory if the
package is not installed on the Python path.  This allows the script to
run without requiring RETFound to be installed as a package.
"""
try:
    from RETFound.models_vit import RETFound_mae  # type: ignore
except ImportError:
    import sys
    import pathlib

    _current_file = pathlib.Path(__file__).resolve()
    for parent in _current_file.parents:
        candidate = parent / "RETFound"
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
            try:
                from models_vit import RETFound_mae  # type: ignore

                break
            except ImportError:
                sys.path.pop(0)
                continue
    else:
        raise ImportError(
            "Unable to locate RETFound.models_vit. Please set PYTHONPATH to include "
            "the RETFound repository or clone it adjacent to this script."
        )


def build_test_transform(img_size: int) -> transforms.Compose:
    """Return the deterministic transform for evaluation on test images."""
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned RETFound model on a test set"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing train/val/test folders",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to fine‑tuned model weights (.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the test predictions CSV",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size used during training (default: 224)",
    )
    args = parser.parse_args()

    test_dir = os.path.join(args.data_root, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Could not find test directory: {test_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_test_transform(args.img_size)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    num_classes = len(test_dataset.classes)
    # Recreate the model and load weights
    model = RETFound_mae(img_size=args.img_size)
    in_features = model.head.in_features if hasattr(model, "head") else model.embed_dim
    model.head = nn.Linear(in_features, num_classes)
    state_dict = torch.load(args.weights, map_location="cpu")
    cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"Loaded model weights with message: {msg}")
    model = model.to(device)
    model.eval()

    # Prepare CSV output
    class_names: List[str] = test_dataset.classes
    csv_path = os.path.join(args.output_dir, "test_predictions.csv")
    header = ["filename"] + [f"prob_{cls}" for cls in class_names] + ["pred_class"]
    with open(csv_path, "w", newline="", encoding="utf8") as cf:
        writer = csv.writer(cf)
        writer.writerow(header)
        global_index = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                # Transfer to CPU for writing
                probs_np = probs.cpu().numpy()
                batch_size_actual = inputs.size(0)
                for i in range(batch_size_actual):
                    # Determine filename.  The dataset stores samples in order
                    # corresponding to ``test_dataset.samples``.  We track a
                    # ``global_index`` that increments across batches.
                    filename = test_dataset.samples[global_index][0]
                    prob_list = probs_np[i].tolist()
                    # Identify the index of the maximum probability without
                    # relying on external libraries.  ``list.index(max(list))``
                    # returns the first occurrence of the maximum value.
                    pred_idx = prob_list.index(max(prob_list))
                    pred_label = class_names[pred_idx]
                    writer.writerow([filename] + prob_list + [pred_label])
                    global_index += 1
    print(f"Saved test predictions to {csv_path}")


if __name__ == "__main__":
    main()
