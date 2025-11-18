#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a trained RETFound ViT-L checkpoint on the held-out test split.

Key features:
- Robust builder selection: --model=auto picks 'retfound_mae' if present, else 'vit_large_patch16'.
- Class order matches training via class_to_idx saved in the checkpoint.
- Outputs:
    <results_dir>/test_predictions.csv   (filename, per-class probs, pred label)
    <results_dir>/test_metrics.json      (loss, acc, macro-OVR AUC, class list)
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score


# -------------------- CLI --------------------
def get_args():
    p = argparse.ArgumentParser(description="Test RETFound ViT-L classifier")

    # Paths
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to dataset root containing train/val/test folders",
    )
    p.add_argument(
        "--retfound_dir",
        type=str,
        required=True,
        help="Path to RETFound repo (contains models_vit.py, util/*)",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Folder containing best_model.pth and where outputs will be saved",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (.pth). Default: <results_dir>/best_model.pth",
    )

    # Model
    p.add_argument(
        "--model",
        type=str,
        default="auto",
        help=(
            "Model builder in models_vit.py. "
            "'auto' tries 'retfound_mae' then 'vit_large_patch16'. "
            "You may also pass an explicit builder name."
        ),
    )

    # Data / loader
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)

    return p.parse_args()


# -------------------- Dataset helpers --------------------
class WithPaths(Dataset):
    """
    Wrap a dataset to also return the file path: (img, target, path).
    Assumes the base dataset exposes .samples (ImageFolder-style).
    """

    def __init__(self, base_ds):
        self.base = base_ds
        # ImageFolder has .samples (list of (path, class_idx))
        if hasattr(base_ds, "samples"):
            self._paths = [p for p, _ in base_ds.samples]
        elif hasattr(base_ds, "imgs"):  # older alias
            self._paths = [p for p, _ in base_ds.imgs]
        else:
            raise AttributeError(
                "Underlying dataset does not expose .samples/.imgs for file paths."
            )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]
        return img, target, self._paths[idx]


def build_test_dataset(args):
    from util.datasets import build_dataset  # from the RETFound repo

    ds_args = SimpleNamespace(
        input_size=args.input_size,
        color_jitter=None,
        aa="rand-m9-mstd0.5-inc1",
        reprob=0.0,
        remode="pixel",
        recount=1,
        data_path=str(args.data_root),
    )
    test_ds = build_dataset(is_train="test", args=ds_args)
    return test_ds


# -------------------- Model selection --------------------
def _pick_model_fn(models_vit, wanted: str):
    # list available callables
    avail = [
        k
        for k, v in models_vit.__dict__.items()
        if callable(v) and not k.startswith("_")
    ]

    prefs = []
    if wanted and wanted != "auto":
        prefs = [wanted]
    else:
        prefs = ["retfound_mae", "vit_large_patch16"]

    for name in prefs:
        if hasattr(models_vit, name):
            return getattr(models_vit, name), name

    # heuristic fallback
    for name in avail:
        if name.startswith("retfound_") or name.startswith("vit_large"):
            return getattr(models_vit, name), name

    raise ValueError(
        "Could not find a suitable ViT-L builder in models_vit.\n"
        f"Requested: {wanted!r}\n"
        f"Available builders: {', '.join(sorted(avail)) or '(none)'}"
    )


def build_model(args, num_classes):
    import models_vit  # from repo (sys.path inserted in main)

    model_fn, chosen_name = _pick_model_fn(models_vit, args.model)
    print(f"[Info] Using model builder: {chosen_name}")
    # Some builders accept (drop_path_rate, global_pool); be tolerant:
    try:
        model = model_fn(num_classes=num_classes, drop_path_rate=0.0, global_pool=True)
    except TypeError:
        model = model_fn(num_classes=num_classes)
    return model


# -------------------- Main --------------------
@torch.no_grad()
def main():
    args = get_args()

    # Put RETFound repo on import path
    import sys

    sys.path.insert(0, args.retfound_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint
    ckpt_path = args.checkpoint or str(Path(args.results_dir) / "best_model.pth")
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Class info & num_classes from training
    class_to_idx = ckpt.get("class_to_idx", None)
    if class_to_idx is None:
        raise KeyError(
            "Checkpoint is missing 'class_to_idx'. Re-run training script to save it."
        )
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    num_classes = ckpt.get("num_classes", len(class_names))

    # Dataset & loader (inference-time transforms only)
    test_base = build_test_dataset(args)
    # Sanity: If the test set class folders differ in order/names, we still report using training order.
    test_ds = WithPaths(test_base)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build model & load weights
    model = build_model(args, num_classes).to(device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[Info] Loaded checkpoint.")
    if missing:
        print("  Missing keys:", missing)
    if unexpected:
        print("  Unexpected keys:", unexpected)
    model.eval()

    # Evaluation
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    all_probs = []
    all_targets = []
    all_files = []

    for images, targets, paths in test_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = ce(logits, targets)
            probs = torch.softmax(logits, dim=1)
        total_loss += loss.item()
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        all_files.extend([os.path.basename(p) for p in paths])

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    preds = all_probs.argmax(1)

    acc = accuracy_score(all_targets, preds)
    try:
        auc = roc_auc_score(
            np.eye(num_classes)[all_targets],
            all_probs,
            average="macro",
            multi_class="ovr",
        )
    except Exception:
        auc = float("nan")

    avg_loss = total_loss / len(test_loader.dataset)

    # Save per-image CSV (training class order)
    probs_headers = [f"prob_{c}" for c in class_names]
    csv_path = Path(args.results_dir) / "test_predictions.csv"
    with open(csv_path, "w") as f:
        f.write("filename," + ",".join(probs_headers) + ",pred_label\n")
        for fname, prob_vec, pred_idx in zip(all_files, all_probs, preds):
            pred_label = class_names[pred_idx]
            row = [fname] + [f"{p:.6f}" for p in prob_vec.tolist()] + [pred_label]
            f.write(",".join(row) + "\n")

    # Save summary metrics
    metrics_path = Path(args.results_dir) / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "loss": float(avg_loss),
                "acc": float(acc),
                "auc_macro_ovr": None if np.isnan(auc) else float(auc),
                "num_classes": int(num_classes),
                "classes": class_names,
            },
            f,
            indent=2,
        )

    print(f"Wrote predictions to: {csv_path}")
    print(f"Wrote metrics to: {metrics_path}")
    print(
        f"Test ACC: {acc:.4f} | AUC (macro-OVR): {auc if not np.isnan(auc) else 'n/a'}"
    )


if __name__ == "__main__":
    main()
