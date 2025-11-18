#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust testing script for ConvNeXt-L classifier.

- Loads checkpoint (best_model.pt by default) and reconstructs class names
  even if `idx_to_class` uses int or str keys (or if only class_to_idx is present).
- Falls back to reading classes.json from results_dir if needed.
- Writes CSV with per-class probabilities, predicted class, and true class.

Env: Python 3.11, PyTorch 2.3.1, TorchVision 0.18.1, CUDA 12.1
"""
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_eval_transform(img_size: int):
    short_side = int((256 / 224) * img_size)
    return transforms.Compose(
        [
            transforms.Resize(short_side, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


def load_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state" not in ckpt:
        if "state_dict" in ckpt:
            ckpt["model_state"] = ckpt["state_dict"]
        else:
            # if a raw state_dict was saved
            if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
                ckpt = {"model_state": ckpt}
    if "model_state" not in ckpt:
        raise KeyError("Checkpoint missing key: model_state")
    return ckpt


def build_model_from_ckpt(
    ckpt: Dict[str, Any], device: torch.device, num_classes: int
) -> Tuple[torch.nn.Module, int]:
    img_size = int(ckpt.get("img_size", 224))
    model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, img_size


def _invert_class_to_idx(class_to_idx: Dict[str, int], num_classes: int) -> List[str]:
    inv = {int(v): str(k) for k, v in class_to_idx.items()}
    return [inv.get(i, f"class_{i}") for i in range(num_classes)]


def _normalize_idx_to_class(idx_to_class: Any, num_classes: int) -> List[str]:
    # list/tuple OR dict with int/str keys
    if isinstance(idx_to_class, (list, tuple)):
        lst = list(idx_to_class)
        if len(lst) < num_classes:
            lst += [f"class_{i}" for i in range(len(lst), num_classes)]
        return [str(x) for x in lst[:num_classes]]
    if isinstance(idx_to_class, dict):
        names: List[str] = []
        for i in range(num_classes):
            if i in idx_to_class:
                names.append(str(idx_to_class[i]))
            elif str(i) in idx_to_class:
                names.append(str(idx_to_class[str(i)]))
            else:
                names.append(f"class_{i}")
        return names
    return [f"class_{i}" for i in range(num_classes)]


def resolve_class_names(
    ckpt: Dict[str, Any], results_dir: Path, num_classes: int
) -> List[str]:
    if "idx_to_class" in ckpt:
        names = _normalize_idx_to_class(ckpt["idx_to_class"], num_classes)
        return names
    if "class_to_idx" in ckpt:
        return _invert_class_to_idx(ckpt["class_to_idx"], num_classes)
    classes_json = results_dir / "classes.json"
    if classes_json.exists():
        with open(classes_json, "r") as f:
            data = json.load(f)
        if "idx_to_class" in data:
            return _normalize_idx_to_class(data["idx_to_class"], num_classes)
        if "class_to_idx" in data:
            return _invert_class_to_idx(data["class_to_idx"], num_classes)
    return [f"class_{i}" for i in range(num_classes)]


def parse_args():
    p = argparse.ArgumentParser(description="Test ConvNeXt-L fundus classifier")
    p.add_argument(
        "--data_root", type=str, required=True, help="Root containing test/ folder"
    )
    p.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Where to write test_predictions.csv",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (.pt). Defaults to results_dir/checkpoints/best_model.pt",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Override number of classes if missing from checkpoint",
    )
    return p.parse_args()


class FilenameImageFolder(datasets.ImageFolder):
    """ImageFolder that returns (image_tensor, label_index, filepath)."""

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, target, path


@torch.no_grad()
def run_inference(
    model, loader: DataLoader, device, class_names: List[str], save_path: Path
):
    softmax = nn.Softmax(dim=1)
    header = (
        ["filename"]
        + [f"prob_{c}" for c in class_names]
        + ["pred_index", "pred_class", "true_class"]
    )
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for images, targets, paths in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = softmax(logits).cpu()
            pred_idx = probs.argmax(dim=1).tolist()
            for i in range(images.size(0)):
                filename = os.path.basename(paths[i])
                row = [filename] + [float(p) for p in probs[i].tolist()]
                row += [
                    int(pred_idx[i]),
                    class_names[pred_idx[i]],
                    loader.dataset.classes[int(targets[i])],
                ]
                w.writerow(row)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU.")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else results_dir / "checkpoints" / "best_model.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)

    # Determine number of classes
    if args.num_classes is not None:
        num_classes = int(args.num_classes)
    else:
        if "num_classes" in ckpt:
            num_classes = int(ckpt["num_classes"])
        else:
            # infer from classifier weight shape
            head = ckpt["model_state"]
            key = next(
                (k for k in head.keys() if k.endswith("classifier.2.weight")), None
            )
            if key is None:
                # fallback: any 2D weight
                key = next(
                    (
                        k
                        for k, v in head.items()
                        if isinstance(v, torch.Tensor) and v.ndim == 2
                    ),
                    None,
                )
            if key is None:
                raise KeyError(
                    "Unable to infer num_classes; pass --num_classes explicitly."
                )
            num_classes = int(head[key].shape[0])

    class_names = resolve_class_names(ckpt, results_dir, num_classes)

    model, img_size = build_model_from_ckpt(ckpt, device, num_classes)

    test_dir = os.path.join(args.data_root, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    transform = build_eval_transform(img_size)
    test_ds = FilenameImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    out_csv = results_dir / "test_predictions.csv"
    run_inference(model, test_loader, device, class_names, out_csv)
    print(f"Inference complete. Saved predictions to: {out_csv}")


if __name__ == "__main__":
    main()
