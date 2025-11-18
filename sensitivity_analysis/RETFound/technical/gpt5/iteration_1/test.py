#!/usr/bin/env python3
"""
Evaluate the fine-tuned RETFound_mae (ViT-L/16) classifier on the held-out test split.

Key fixes:
- Import RETFound's models_vit.py by absolute file path (importlib) to avoid name collisions.
- Build ViT-L robustly: try known factory names; else instantiate VisionTransformer with ViT-L/16 config.

Outputs:
- <output_dir>/test_predictions.csv with:
    filename, score_<class> for each class, predicted_label
- Prints overall top-1 accuracy (assuming test is an ImageFolder with class subdirs).

Env: Python 3.11, PyTorch 2.3+, TorchVision 0.18+, CUDA 12.1, single GPU.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import torch.nn as nn
from functools import partial


# -----------------------------
# CLI
# -----------------------------
def get_args():
    p = argparse.ArgumentParser(description="Test RETFound_mae (ViT-L) classifier")
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder containing test/ subfolder",
    )
    p.add_argument(
        "--retfound_dir",
        type=str,
        required=True,
        help="Path to RETFound repo folder (for models_vit.py)",
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to fine-tuned best_model.pth from train.py",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to write test_predictions.csv",
    )
    p.add_argument(
        "--classes_json",
        type=str,
        default=None,
        help="Optional path to classes.json from training",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--fp16", action="store_true", default=True)
    return p.parse_args()


# -----------------------------
# Transforms
# -----------------------------
def build_eval_tf(img_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize(
                int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


# -----------------------------
# Safe file-based import + robust model build
# -----------------------------
import importlib.util


def import_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def load_retfound_models(retfound_dir: str):
    mv_path = os.path.join(retfound_dir, "models_vit.py")
    models_vit = import_from_file("retfound_models_vit", mv_path)
    return models_vit


def build_vitl_model(
    models_vit, num_classes: int, img_size: int, drop_path: float, global_pool: bool
):
    """
    Try typical factory names first; else fallback to direct VisionTransformer ViT-L/16 config.
    """
    factories = [
        "vit_large_patch16",
        "vit_large_patch16_224",
        "vit_large_patch16_mae",
        "RETFound_vit_large_patch16",
    ]
    for name in factories:
        if hasattr(models_vit, name):
            return getattr(models_vit, name)(
                num_classes=num_classes,
                drop_path_rate=drop_path,
                global_pool=global_pool,
                img_size=img_size,
            )
    # Fallback to direct construction
    VT = getattr(models_vit, "VisionTransformer", None)
    if VT is None:
        import timm.models.vision_transformer as tvit

        VT = tvit.VisionTransformer
    return VT(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        drop_path_rate=drop_path,
        global_pool=global_pool,
    )


# -----------------------------
# Dataset wrapper to expose file paths
# -----------------------------
class WithPath(datasets.ImageFolder):
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, path


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint metadata for consistent model/head
    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt["state_dict"]
    ckpt_num_classes = ckpt.get("num_classes", None)
    ckpt_img_size = ckpt.get("img_size", args.img_size)
    ckpt_global_pool = ckpt.get("global_pool", True)

    # Dataset
    test_dir = os.path.join(args.data_root, "test")
    tf = build_eval_tf(ckpt_img_size)
    test_set = WithPath(test_dir, transform=tf)

    # Classes for CSV header (prefer training order if provided)
    if args.classes_json and os.path.exists(args.classes_json):
        mapping = json.load(open(args.classes_json, "r"))
        header_classes = mapping.get("classes", test_set.classes)
    else:
        header_classes = test_set.classes

    num_classes_header = len(header_classes)

    # Model
    models_vit = load_retfound_models(args.retfound_dir)

    # Determine num_classes for the model head: prefer checkpoint’s metadata
    if ckpt_num_classes is None:
        ckpt_num_classes = num_classes_header
    model = build_vitl_model(
        models_vit=models_vit,
        num_classes=ckpt_num_classes,
        img_size=ckpt_img_size,
        drop_path=0.0,
        global_pool=ckpt_global_pool,
    )
    # Load weights strictly to catch mismatches early
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    # DataLoader
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Inference
    rows = []
    total = 0
    correct = 0

    with torch.no_grad():
        for images, targets, paths in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.fp16, dtype=torch.float16):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            probs_cpu = probs.detach().cpu().tolist()
            preds_cpu = preds.detach().cpu().tolist()
            for pth, pr, pd in zip(paths, probs_cpu, preds_cpu):
                row = {"filename": os.path.basename(pth)}
                # If header class count doesn't match prob length, fall back to test_set order
                if len(pr) == num_classes_header:
                    class_names = header_classes
                else:
                    class_names = test_set.classes
                for cname, score in zip(class_names, pr):
                    row[f"score_{cname}"] = f"{float(score):.6f}"
                row["predicted_label"] = class_names[pd]
                rows.append(row)

    acc = 100.0 * correct / max(1, total)
    print(f"Test top-1 accuracy: {acc:.2f}%")

    # Write CSV (deterministic column order)
    header = ["filename"] + [f"score_{c}" for c in header_classes] + ["predicted_label"]
    # Ensure all keys present (in case of fallback class order)
    all_keys = set().union(*[set(r.keys()) for r in rows]) if rows else set(header)
    for h in header:
        all_keys.add(h)
    # Keep preferred order, then append any extra keys (stable)
    final_header = [h for h in header if h in all_keys] + [
        k for k in sorted(all_keys) if k not in header
    ]

    out_csv = os.path.join(args.output_dir, "test_predictions.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote predictions: {out_csv}")


if __name__ == "__main__":
    main()
