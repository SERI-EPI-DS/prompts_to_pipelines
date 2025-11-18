"""
Evaluate a fine‑tuned RETFound classifier on a dataset split (e.g. test).

This script loads a model checkpoint produced by `train_classifier.py` and
computes loss and accuracy on a specified split of the dataset (usually
`test`).  It also writes a CSV file containing the image path, true class
name and predicted class name for each image.

Example usage:

```sh
python test_classifier.py \
    --data_dir /path/to/dataset_root \
    --checkpoint /path/to/best_model.pth \
    --split test \
    --output_dir ./experiments/retfound_messidor2
```

The script infers the class mapping from the checkpoint (``class_to_idx``)
whenever possible.  If this field is missing you must specify
``--num_classes`` and ensure that the dataset directory names match the
order used during training.
"""

import argparse
import csv
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import (
    VisionTransformer,
    build_transforms,
    load_pretrained_weights,
    evaluate,
    predict,
)


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    print(f"Using device: {device}")

    # Load checkpoint.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = ckpt.get("args", {})
    class_to_idx = ckpt.get("class_to_idx", None)
    if class_to_idx is not None:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
    else:
        idx_to_class = None
        num_classes = args.num_classes
        if num_classes is None:
            raise RuntimeError(
                "The checkpoint does not contain class mapping (class_to_idx)."
                " Please specify --num_classes."
            )

    # Build transforms and dataset.
    _, eval_tf = build_transforms(args.input_size)
    split_dir = Path(args.data_dir) / args.split
    dataset = datasets.ImageFolder(str(split_dir), transform=eval_tf)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Instantiate model architecture.  We don't load HuggingFace weights here
    # because the checkpoint already contains the fine‑tuned weights.
    model = VisionTransformer(
        num_classes=num_classes,
        global_pool=True,
        drop_path_rate=args.drop_path,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # Loss function (used only for reporting loss; no optimisation).
    criterion = torch.nn.CrossEntropyLoss()

    loss, acc = evaluate(model, loader, criterion, device)
    print(
        f"{args.split.capitalize()} loss: {loss:.4f}, {args.split.capitalize()} accuracy: {acc:.2f}%"
    )

    # Generate predictions and write to CSV.
    preds, targets = predict(model, loader, device)
    image_paths = [path for path, _ in dataset.samples]
    if idx_to_class is None:
        # Use the folder names as labels.  They are ordered alphabetically.
        idx_to_class = {i: cls_name for i, cls_name in enumerate(dataset.classes)}
    csv_path = Path(args.output_dir) / f"{args.split}_predictions.csv"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "true_label", "predicted_label"])
        for img_path, tgt, pred in zip(image_paths, targets, preds):
            writer.writerow([img_path, idx_to_class[tgt], idx_to_class[pred]])
    print(f"Saved predictions to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned RETFound classifier"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Root directory of the dataset"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth) from training",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write predictions CSV",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes (required if checkpoint lacks class mapping)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input image size (should match training)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader worker count"
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.0,
        help="Drop path rate (must match model configuration)",
    )
    parser.add_argument(
        "--force_cpu", action="store_true", help="Force CPU even if CUDA available"
    )
    args = parser.parse_args()
    main(args)
