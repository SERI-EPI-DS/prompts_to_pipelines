#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a fine-tuned RETFound (ViT-L/16) classifier on an ImageFolder split.
- Robustly constructs the model even if 'vit_large_patch16' isn't exported.
- Loads a checkpoint saved by the paired training script (handles DDP prefixes).
- Writes metrics and per-image predictions CSV.
"""

import argparse, json, os, sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ----------------- Args -----------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Dataset root with train/val/test subfolders.",
    )
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint .pth",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to write metrics & predictions CSV",
    )
    p.add_argument(
        "--retfound_repo",
        type=str,
        default="",
        help="Path to RETFound repo (if this script isn't inside it).",
    )
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


# ----------------- Helpers -----------------
def ensure_repo_on_path(repo_path: str):
    if repo_path:
        sys.path.insert(0, repo_path)
    else:
        here = Path(__file__).resolve().parent
        if (here / "models_vit.py").exists():
            sys.path.insert(0, str(here))


def build_val_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_retfound_vit_large(
    models_vit,
    *,
    num_classes: int,
    drop_path_rate: float = 0.0,
    global_pool: bool = True,
):
    """
    Robustly construct a ViT-L/16 compatible with RETFound weights, even if the
    constructor name differs across forks/repos.
    """
    try:
        print(f"[info] using models_vit from: {models_vit.__file__}")
    except Exception:
        pass

    # Try common names first (older RETFound MAE, timm-style variants, forks)
    for name in (
        "vit_large_patch16",
        "vit_large_patch16_224",
        "retfound_vit_large_patch16",
    ):
        ctor = getattr(models_vit, name, None)
        if callable(ctor):
            return ctor(
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                global_pool=global_pool,
            )

    # Fallback: construct VisionTransformer with ViT-L/16 config used by RETFound
    if hasattr(models_vit, "VisionTransformer"):
        from functools import partial
        import torch.nn as nn

        return models_vit.VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
        )

    raise RuntimeError(
        "Could not construct ViT-L/16 from models_vit; ensure the RETFound repo is on sys.path."
    )


# ----------------- Eval -----------------
@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device, class_names: List[str], out_csv: Path
) -> Tuple[float, float]:
    import csv

    ce = nn.CrossEntropyLoss(reduction="sum")
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "true_label", "pred_label", "pred_idx", "conf"])

        bs = loader.batch_size if loader.batch_size is not None else 1
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            # metrics
            loss_sum += ce(logits, labels).item()
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # map back to file paths via dataset order (shuffle=False)
            base = batch_idx * bs
            for j in range(labels.size(0)):
                path, true_idx = loader.dataset.samples[base + j]
                w.writerow(
                    [
                        path,
                        class_names[true_idx],
                        class_names[pred[j].item()],
                        int(pred[j].item()),
                        float(conf[j].item()),
                    ]
                )

    avg_loss = loss_sum / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


# ----------------- Main -----------------
def main():
    args = get_args()
    ensure_repo_on_path(args.retfound_repo)

    try:
        import models_vit  # from RETFound repo
    except Exception as e:
        raise RuntimeError(
            f"Could not import RETFound modules. "
            f"Pass --retfound_repo to point to the repo. Original error: {e}"
        )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    # Strip possible 'module.' prefixes (DDP)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # Determine number of classes
    nb_classes = None
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        nb_classes = ckpt["args"].get("nb_classes", None)
    if nb_classes is None and "class_to_idx" in ckpt:
        nb_classes = len(ckpt["class_to_idx"])
    assert (
        nb_classes is not None
    ), "nb_classes missing in checkpoint; expected in ckpt['args'] or ckpt['class_to_idx']."

    # Build model robustly (no KeyError even if ctor name differs)
    model = build_retfound_vit_large(
        models_vit,
        num_classes=nb_classes,
        drop_path_rate=0.0,
        global_pool=True,
    )
    # Load weights strictly
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Still raise clearly—eval should match training head
        raise RuntimeError(
            f"State dict mismatch.\nMissing: {missing}\nUnexpected: {unexpected}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Data
    split_dir = Path(args.data_dir) / args.split
    tf = build_val_transform(args.input_size)
    ds = datasets.ImageFolder(split_dir, transform=tf)
    # consistent class order
    class_to_idx = ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Evaluate
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_csv = out_dir / f"predictions_{args.split}.csv"

    loss, acc = evaluate(model, loader, device, class_names, preds_csv)

    with open(out_dir / f"metrics_{args.split}.json", "w") as f:
        json.dump({"loss": loss, "acc": acc, "num_images": len(ds)}, f, indent=2)

    print(f"{args.split} | loss: {loss:.4f} | acc: {acc:.4f} | n={len(ds)}")
    print(f"Wrote: {preds_csv} and metrics_{args.split}.json")


if __name__ == "__main__":
    main()
