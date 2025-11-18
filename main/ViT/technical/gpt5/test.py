#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test a trained Swin-V2-B classifier.

- Loads best.pt (or custom checkpoint) from the training run
- Runs inference on data_root/test
- Writes CSV with: filename, per-class probabilities, predicted_label
- If test labels exist (ImageFolder-style), also computes accuracy and saves metrics.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import swin_v2_b
from torchvision.transforms.functional import InterpolationMode
import pandas as pd
from PIL import Image
import os


# --------------------------- Dataset Fallback ---------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageListDataset(Dataset):
    """Fallback dataset when test/ is not in class-subfolders. Returns (image, dummy_label, rel_path)."""

    def __init__(self, root: Path, transform):
        self.root = Path(root)
        self.samples = []
        for dirpath, _, filenames in os.walk(self.root):
            for fn in filenames:
                if Path(fn).suffix.lower() in IMG_EXTS:
                    p = Path(dirpath) / fn
                    rel = str(p.relative_to(self.root))
                    self.samples.append((p, rel))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, rel = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # return dummy label -1
        return img, -1, rel


# --------------------------- Helpers ---------------------------


def build_eval_transform(image_size: int, mean: List[float], std: List[float]):
    return T.Compose(
        [
            T.Resize(
                int(image_size * 1.15),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    has_labels: bool,
    save_csv: Path,
) -> Dict:
    model.eval()
    probs_all = []
    preds_all = []
    files_all = []
    labels_all = [] if has_labels else None

    for batch in loader:
        if has_labels:
            images, labels, paths = batch  # custom collate below adds paths
        else:
            images, labels, paths = batch  # -1 labels from fallback

        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

        pred_idx = probs.argmax(dim=1).cpu().tolist()
        preds_all.extend(pred_idx)
        probs_all.extend(probs.cpu().tolist())
        files_all.extend(list(paths))

        if has_labels:
            labels_all.extend(labels.tolist())

    # Save CSV
    df = pd.DataFrame({"filename": files_all})
    for i, cname in enumerate(class_names):
        df[f"score_{cname}"] = [row[i] for row in probs_all]
    df["predicted_label"] = [class_names[i] for i in preds_all]
    df.to_csv(save_csv, index=False)

    metrics = {}
    if has_labels:
        labels_names = [class_names[i] for i in labels_all]
        correct = sum(1 for p, y in zip(preds_all, labels_all) if p == y)
        acc = correct / max(1, len(labels_all))
        metrics = {"num_samples": len(labels_all), "accuracy": acc}
    return metrics


def custom_collate_with_paths(dataset_has_labels: bool):
    # Wrap default collate to also pass relative paths forward
    from torch.utils.data._utils.collate import default_collate

    def collate_fn(batch):
        if dataset_has_labels:
            imgs, labels = zip(*[(b[0], b[1]) for b in batch])
            paths = [
                dataset.samples[i][0] if hasattr(dataset, "samples") else None
                for i, dataset in []
            ]  # unused
        # For ImageFolder we can reconstruct rel paths from dataset.samples; simpler approach below:
        # We'll detect batch element shapes.
        elems = batch[0]
        if len(elems) == 2:
            # ImageFolder default (img, label) — we need to also pass a filename; reconstruct from underlying dataset
            # The DataLoader will have dataset attribute accessible via closure; but here we don't have it.
            # Simpler approach: we will create a custom wrapper for ImageFolder below that returns (img, label, rel_path)
            raise RuntimeError("Use ImageFolderWithPaths for path-aware batches.")
        return default_collate(batch)

    return collate_fn  # (Unused; kept for reference)


class ImageFolderWithPaths(ImageFolder):
    """ImageFolder that also returns relative file path as third element."""

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        rel = str(Path(path).relative_to(self.root))
        return img, target, rel


# --------------------------- Main ---------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Test a Swin-V2-B checkpoint on test/ and export predictions."
    )
    ap.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root folder containing test/ (and optionally class subfolders)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output folder where results will be saved",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint (default: <out-dir>/best.pt)",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt) if args.ckpt is not None else (out_dir / "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    class_to_idx = ckpt.get("class_to_idx", None)
    if class_to_idx is None:
        raise RuntimeError(
            "Checkpoint missing class_to_idx. Re-train and ensure training script saves mapping."
        )

    classes = meta.get("classes", None)
    if classes is None:
        # As a fallback, sort by index from mapping
        inv = {v: k for k, v in class_to_idx.items()}
        classes = [inv[i] for i in range(len(inv))]

    image_size = int(meta.get("image_size", 256))
    mean = meta.get("mean", [0.485, 0.456, 0.406])
    std = meta.get("std", [0.229, 0.224, 0.225])

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = swin_v2_b(weights=None)  # structure only; we load finetuned head below
    in_feats = model.head.in_features
    model.head = nn.Linear(in_feats, len(classes))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device)
    model.eval()

    # Transform
    eval_tfms = build_eval_transform(image_size, mean, std)

    # Dataset: try labeled ImageFolder first; if it fails, fallback to ImageListDataset
    test_dir = data_root / "test"
    has_labels = False
    dataset = None
    try:
        dataset = ImageFolderWithPaths(root=str(test_dir), transform=eval_tfms)
        # Verify class ordering matches training:
        # If test set has labels, it must use same class names; order can differ, so we align via mapping.
        # We'll keep has_labels True but ONLY for metrics; predictions CSV always follows training class order.
        has_labels = len(dataset.classes) >= 1
    except Exception:
        dataset = None

    if dataset is None or len(dataset) == 0:
        # Unlabeled or non-ImageFolder test/ — use fallback
        dataset = ImageListDataset(root=test_dir, transform=eval_tfms)
        has_labels = False

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )

    # Inference
    save_csv = out_dir / "test_predictions.csv"
    metrics = run_inference(
        model, loader, device, classes, has_labels, save_csv=save_csv
    )

    # Optional metrics if labels exist
    if has_labels and len(metrics) > 0:
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"Saved predictions to: {save_csv}")
    if has_labels and "accuracy" in metrics:
        print(
            f"Test accuracy: {metrics['accuracy']:.4f} over {metrics['num_samples']} samples."
        )


if __name__ == "__main__":
    main()
