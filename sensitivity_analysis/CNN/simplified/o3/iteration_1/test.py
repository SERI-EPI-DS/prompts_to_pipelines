#!/usr/bin/env python3
"""
Evaluate a trained ConvNeXt-Large model on the *test* split and write
predictions to CSV.

• Fix: fall back to standard ImageNet mean/std when
  `weights.meta["mean"]` / `["std"]` are unavailable (or the enum is missing).

Example:
    python test_convnext.py \
        --data_dir /path/to/dataset \
        --model_path ./checkpoints/best_model.pth \
        --output_csv results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_large

# ───────────────────── Robust weight-enum import ─────────────────────
try:
    from torchvision.models import ConvNeXt_Large_Weights  # torchvision ≥ 0.13

    _HAS_ENUM = True
except ImportError:  # older torchvision
    ConvNeXt_Large_Weights = None
    _HAS_ENUM = False


def _imagenet_stats() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Classic ImageNet mean / std (RGB)."""
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ────────────────────────────── Main ──────────────────────────────
@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Test ConvNeXt-Large classifier")
    p.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Dataset root containing a 'test' folder",
    )
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output_csv", type=Path, default=Path("predictions.csv"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Mean / std for normalisation ───
    if _HAS_ENUM:
        weights_enum = ConvNeXt_Large_Weights.IMAGENET1K_V1
        if "mean" in weights_enum.meta and "std" in weights_enum.meta:
            mean, std = weights_enum.meta["mean"], weights_enum.meta["std"]
        else:  # Bullet-proof fallback
            mean, std = _imagenet_stats()
    else:  # Very old torchvision
        mean, std = _imagenet_stats()

    tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # ─── Dataset / loader ───
    test_ds = datasets.ImageFolder(args.data_dir / "test", transform=tfms)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ─── Model ───
    num_classes = len(test_ds.classes)
    model = convnext_large(weights=None)  # we’ll load our own checkpoint
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    # ─── Inference ───
    preds, labels, paths = [], [], [p for p, _ in test_ds.samples]
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(y.tolist())

    class_names = test_ds.classes
    df = pd.DataFrame(
        {
            "image_path": paths,
            "true_label": [class_names[i] for i in labels],
            "predicted_label": [class_names[i] for i in preds],
        }
    )
    df.to_csv(args.output_csv, index=False)
    acc = (df.true_label == df.predicted_label).mean()

    print(f"📄 Saved predictions to: {args.output_csv}")
    print(f"✅ Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
