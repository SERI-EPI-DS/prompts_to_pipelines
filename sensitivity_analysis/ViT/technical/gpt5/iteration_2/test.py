#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a fine-tuned Swin-V2-B on the held-out test set and save predictions.

- Loads best model from --weights (a checkpoint saved by train.py)
- Reads class names from classes.json in the same folder (or from the checkpoint)
- Writes a CSV with:
    filename, pred_class, pred_index, <class_0_prob>, <class_1_prob>, ...

Tested with:
  Python 3.11
  PyTorch 2.3.1
  TorchVision 0.18.1
  CUDA 12.1
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import InterpolationMode
from torchvision.models.swin_transformer import swin_v2_b, Swin_V2_B_Weights


def get_mean_std_from_weights(weights: Swin_V2_B_Weights):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    interpolation = InterpolationMode.BICUBIC
    size = 256
    try:
        meta = weights.meta
        mean = meta.get("mean", mean)
        std = meta.get("std", std)
        size = (
            int(meta.get("min_size", size))
            if isinstance(meta.get("min_size"), (int, float))
            else size
        )
    except Exception:
        pass
    try:
        _t = weights.transforms()
        if hasattr(_t, "transforms"):
            for t in _t.transforms:
                if hasattr(t, "interpolation"):
                    interpolation = t.interpolation
                if hasattr(t, "size") and isinstance(t.size, (tuple, list)):
                    size = max(size, max(t.size))
    except Exception:
        pass
    return mean, std, interpolation, size


def make_loader(test_dir: Path, img_size: int, batch_size: int, num_workers: int):
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    mean, std, interpolation, _ = get_mean_std_from_weights(weights)
    tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14), interpolation=interpolation),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    ds = ImageFolder(test_dir, transform=tf)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    return ds, dl


def load_model(weights_path: Path, num_classes: int, device: torch.device):
    model = swin_v2_b(weights=None)  # architecture; head will be replaced by checkpoint
    # Replace head now to ensure shape compatibility when strict=False
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features, num_classes)
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)  # support raw state_dict too
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_classes(out_dir: Path, ckpt_path: Path) -> List[str]:
    classes_json = out_dir / "classes.json"
    if classes_json.exists():
        with open(classes_json, "r") as f:
            j = json.load(f)
            if "classes" in j and isinstance(j["classes"], list):
                return j["classes"]

    # Fallback: try checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "classes" in ckpt and isinstance(ckpt["classes"], list):
            return ckpt["classes"]
        if "state_dict" in ckpt and "classes" in ckpt["state_dict"]:
            return ckpt["state_dict"]["classes"]

    raise RuntimeError("Could not find class names in classes.json or checkpoint.")


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    data_root = Path(args.data_dir).expanduser().resolve()
    test_dir = data_root / "test"

    classes = load_classes(out_dir, Path(args.weights).expanduser().resolve())
    num_classes = len(classes)

    ds, dl = make_loader(test_dir, args.img_size, args.batch_size, args.num_workers)
    # Ensure class order matches training (ImageFolder sorts alphabetically)
    assert ds.classes == classes, (
        "Test classes do not match training classes.\n"
        f"Test classes: {ds.classes}\nTrain classes: {classes}\n"
        "Please ensure folder names are identical."
    )

    model = load_model(Path(args.weights).expanduser().resolve(), num_classes, device)

    softmax = torch.nn.Softmax(dim=1)

    csv_path = out_dir / "test_predictions.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV header
    header = ["filename", "pred_class", "pred_index"] + [f"{c}_prob" for c in classes]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with torch.no_grad():
            for images, targets in dl:
                images = images.to(device, non_blocking=True)
                with autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(images)
                    probs = softmax(logits).cpu()

                # Get the subset of filenames for this batch, in the same order as 'images'
                # ds.samples is aligned with DataLoader order (no shuffle), each (path, class_idx)
                # We can fetch the filenames by index via the sampler indices kept internally by DataLoader.
                # More robustly, track them from the batch itself using the dataset indices exposed by default collate:
                # However, ImageFolder doesn't return indices. We'll map by using the running pointer.
                # Since shuffle=False and drop_last=False, we can compute filenames based on the batch window.
                # Compute start index for this batch:
                batch_size = images.size(0)
                # Current number of rows already written:
                rows_written = (
                    f.tell()
                )  # bytes written; cannot infer batch index easily
                # We'll maintain a simple external counter instead; so do an explicit enumerate loop outside.

    # Re-run with enumerate to keep it clean and ordered
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        idx = 0
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dl):
                images = images.to(device, non_blocking=True)
                with autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1).cpu()

                b = images.size(0)
                # filenames for this batch: use underlying dataset order
                start = batch_idx * args.batch_size
                end = start + b
                batch_samples = ds.samples[start:end]  # list of (path, class_idx)

                pred_indices = probs.argmax(dim=1).tolist()
                for i in range(b):
                    path_i = Path(batch_samples[i][0])
                    rel_path = (
                        path_i.relative_to(test_dir)
                        if test_dir in path_i.parents or path_i == test_dir
                        else path_i.name
                    )
                    row = [
                        str(rel_path),
                        classes[pred_indices[i]],
                        pred_indices[i],
                    ] + [f"{p:.6f}" for p in probs[i].tolist()]
                    writer.writerow(row)
                    idx += 1

    print(f"Saved predictions to: {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Test fine-tuned Swin-V2-B on held-out test set."
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data root containing test/ subfolder",
    )
    p.add_argument(
        "--out_dir", type=str, required=True, help="Where results & classes.json live"
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to best_model.pth (or checkpoint)",
    )
    p.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Input resolution (should match train)",
    )
    p.add_argument("--batch_size", type=int, default=64, help="Eval batch size")
    p.add_argument("--num_workers", type=int, default=6, help="DataLoader workers")
    p.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
