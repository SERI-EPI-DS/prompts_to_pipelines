#!/usr/bin/env python3
# Test a fine-tuned ConvNeXt-L classifier and export CSV of predictions.
#
# Example:
#   python test.py \
#     --data-root /path/to/dataset \
#     --test-folder test \
#     --checkpoint /path/to/outputs/best_ema.pth \
#     --outdir ./outputs/eval_run1 \
#     --img-size 384
#
# The script will produce predictions.csv with columns:
#   filename, prob_<class_0>, ..., prob_<class_N-1>, pred_label
#
# If the test split is class-structured (ImageFolder), accuracy is also reported.

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode
from torchvision.models import convnext_large, ConvNeXt_Large_Weights


def build_eval_transforms(img_size: int, mean, std):
    return T.Compose(
        [
            T.Resize(
                int(img_size * 1.15),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.CenterCrop(img_size),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean, std),
        ]
    )


def replace_classifier_with(model: nn.Module, num_classes: int):
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for i in range(len(model.classifier) - 1, -1, -1):
            if isinstance(model.classifier[i], nn.Linear):
                in_features = model.classifier[i].in_features
                model.classifier[i] = nn.Linear(in_features, num_classes)
                return model
    raise RuntimeError("Unexpected ConvNeXt classifier layout; cannot replace head.")


def main():
    p = argparse.ArgumentParser(
        description="Evaluate ConvNeXt-L and export CSV predictions"
    )
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--test-folder", type=str, default="test")
    p.add_argument(
        "--checkpoint", type=str, required=True, help="Path to best.pth or best_ema.pth"
    )
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Try to load class index mapping from the same folder as checkpoint, else from data
    ckpt_path = Path(args.checkpoint)
    ckpt_dir = ckpt_path.parent
    mapping_path = ckpt_dir / "class_index.json"
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        idx_to_class = {int(k): v for k, v in mapping.get("idx_to_class", {}).items()}
        class_to_idx = mapping.get("class_to_idx", {})
    else:
        # Fall back to reading from the test dataset
        idx_to_class, class_to_idx = None, None

    # Build dataset
    test_dir = Path(args.data_root) / args.test_folder
    # We need mean/std; pick from ImageNet weights
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
    mean = tuple(weights.meta.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(weights.meta.get("std", (0.229, 0.224, 0.225)))
    tfm = build_eval_transforms(args.img_size, mean, std)

    test_ds = datasets.ImageFolder(str(test_dir), transform=tfm)
    if idx_to_class is None:
        # Construct from dataset
        idx_to_class = {v: k for k, v in test_ds.class_to_idx.items()}
        class_to_idx = test_ds.class_to_idx

    num_classes = len(idx_to_class)
    device = torch.device(args.device)

    # Model
    model = convnext_large(weights=None)
    model = replace_classifier_with(model, num_classes)
    model.to(device)

    # Load checkpoint
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model_state", ckpt)  # support raw state_dict too
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    # Predict
    all_rows = []
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu()

            for i in range(images.size(0)):
                path, true_idx = test_ds.samples[total + i]
                fname = Path(path).name
                prob_list = probs[i].cpu().tolist()
                pred_idx = int(preds[i].item())
                pred_label = idx_to_class[pred_idx]
                row = [fname] + prob_list + [pred_label]
                all_rows.append(row)

            if hasattr(loader.dataset, "targets") or test_ds.samples:
                targets_cpu = targets.cpu()
                correct += (preds == targets_cpu).sum().item()
                total += targets_cpu.size(0)
            else:
                total += images.size(0)

    # Save CSV
    header = (
        ["filename"]
        + [f"prob_{idx_to_class[i]}" for i in range(num_classes)]
        + ["pred_label"]
    )
    csv_path = outdir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in all_rows:
            writer.writerow(row)

    # Report accuracy if labels are present
    if num_classes > 1 and total > 0:
        acc = 100.0 * correct / total
        print(f"Test Top-1 Accuracy: {acc:.2f}%  ({correct}/{total})")
    print(f"Saved: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
