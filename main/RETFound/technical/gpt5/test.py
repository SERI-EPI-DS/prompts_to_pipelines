#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a fine-tuned RETFound_mae (ViT-L/16) checkpoint on the held-out test set.

Writes a CSV with:
  filename, pred_label, pred_index, <class_0_prob>, <class_1_prob>, ...

Also prints test accuracy (if labels exist via folder structure) and saves a meta JSON.
"""
import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------- RETFound helpers ----------
def add_retfound_to_syspath(retfound_root: str):
    p = Path(retfound_root).resolve()
    if not p.exists():
        raise FileNotFoundError(f"RETFound repo path not found: {p}")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def build_model(
    num_classes: int,
    input_size: int = 224,
    drop_path_rate: float = 0.0,
    global_pool: bool = True,
):
    import models_vit as rv

    model = None
    if hasattr(rv, "RETFound_mae"):
        try:
            model = rv.RETFound_mae(
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                img_size=input_size,
                global_pool=global_pool,
            )
        except TypeError:
            model = rv.RETFound_mae(num_classes=num_classes)
    elif hasattr(rv, "vit_large_patch16"):
        try:
            model = rv.vit_large_patch16(
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                img_size=input_size,
                global_pool=global_pool,
            )
        except TypeError:
            model = rv.vit_large_patch16(num_classes=num_classes)
    else:
        raise RuntimeError(
            "Could not locate RETFound_mae or vit_large_patch16 in models_vit.py"
        )

    # reset classifier just in case
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_f = model.head.in_features
        model.head = nn.Linear(in_f, num_classes)
    elif (
        hasattr(model, "heads")
        and hasattr(model.heads, "head")
        and isinstance(model.heads.head, nn.Linear)
    ):
        in_f = model.heads.head.in_features
        model.heads.head = nn.Linear(in_f, num_classes)
    return model


@torch.no_grad()
def _maybe_interpolate_pos_embed(model, state_dict: Dict[str, torch.Tensor]):
    try:
        from util.pos_embed import interpolate_pos_embed

        interpolate_pos_embed(model, state_dict)
    except Exception:
        pass


def load_checkpoint(
    model: nn.Module, ckpt_path: str, device: torch.device
) -> List[str] | None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt.get("classes", None)

    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state_dict = {
        k.replace("module.", "") for k in []
    } or state_dict  # no-op to keep pattern clear
    new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    _maybe_interpolate_pos_embed(model, new_state)

    ms = model.state_dict()
    for k in list(new_state.keys()):
        if "head" in k and k in ms and new_state[k].shape != ms[k].shape:
            new_state.pop(k)
    model.load_state_dict(new_state, strict=False)
    model.to(device).eval()
    return classes


# ---------- Data ----------
def make_loader(
    test_dir: str, input_size: int, batch_size: int, workers: int
) -> Tuple[datasets.ImageFolder, DataLoader]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.15)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    ds = datasets.ImageFolder(test_dir, transform=tfm)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    return ds, loader


# ---------- Main ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Test RETFound_mae (ViT-L/16)")
    ap.add_argument(
        "--data-root", type=str, required=True, help="Root containing test/ subfolder"
    )
    ap.add_argument(
        "--retfound-root", type=str, required=True, help="Path to RETFound repo"
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint-best.pth",
    )
    ap.add_argument(
        "--out-csv", type=str, required=True, help="Path to save CSV of predictions"
    )
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--global-pool", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    add_retfound_to_syspath(args.retfound_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    test_dir = os.path.join(args.data_root, "test")
    ds, loader = make_loader(test_dir, args.input_size, args.batch_size, args.workers)

    # class list from dataset (fallback)
    fallback_classes = ds.classes
    num_classes = len(fallback_classes)

    # build & load
    model = build_model(
        num_classes=num_classes,
        input_size=args.input_size,
        global_pool=args.global_pool,
    )
    classes = load_checkpoint(model, args.checkpoint, device) or fallback_classes
    idx_to_class = {i: c for i, c in enumerate(classes)}

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    n_correct = 0
    n_total = 0

    softmax = nn.Softmax(dim=1)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename", "pred_label", "pred_index"] + [
            f"score_{c}" for c in classes
        ]
        writer.writerow(header)

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(images)
                probs = softmax(logits).detach().cpu()

            # --- FIX: compute paths by slicing ds.samples with a stable batch_start ---
            batch_size_curr = images.size(0)
            batch_start = n_total
            paths_labels_batch = ds.samples[batch_start : batch_start + batch_size_curr]

            for i in range(batch_size_curr):
                path_i, true_idx = paths_labels_batch[i]
                fname = os.path.basename(path_i)

                prob_i = probs[i]
                pred_idx = int(prob_i.argmax().item())
                pred_label = idx_to_class.get(pred_idx, str(pred_idx))

                writer.writerow(
                    [fname, pred_label, pred_idx]
                    + [f"{p:.6f}" for p in prob_i.tolist()]
                )

                # accuracy (if folder-based labels exist)
                if 0 <= true_idx < len(classes) and pred_idx == true_idx:
                    n_correct += 1

            n_total += batch_size_curr  # update AFTER finishing the batch

    if n_total > 0:
        print(f"[test] images={n_total}, accuracy={n_correct / n_total:.4f}")

    meta_path = os.path.splitext(args.out_csv)[0] + "_meta.json"
    with open(meta_path, "w") as jf:
        json.dump({"classes": classes}, jf, indent=2)
    print(f"[write] CSV -> {args.out_csv}")
    print(f"[write] meta -> {meta_path}")


if __name__ == "__main__":
    main()
