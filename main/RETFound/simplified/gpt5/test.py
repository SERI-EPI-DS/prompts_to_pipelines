#!/usr/bin/env python3
"""
Evaluate a trained RETFound classifier and export predictions.

Example:
    python test.py \
        --data_root /path/to/dataset \
        --test_subdir test \
        --checkpoint /results/retfound_exp1/checkpoint-best.pth \
        --output_dir /results/retfound_exp1/eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import from RETFound repo
import models_vit
from util.pos_embed import interpolate_pos_embed

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(input_size=224):
    return transforms.Compose(
        [
            transforms.Resize(
                int(input_size * 1.14),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---- Robust RETFound model builder ----
def _make_vit_large_patch16_from_class(
    nb_classes: int, drop_path: float, global_pool: bool
):
    from functools import partial

    model = models_vit.VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=drop_path,
        global_pool=global_pool,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=nb_classes,
    )
    return model


def _build_vit_large_patch16(nb_classes: int, drop_path: float, global_pool: bool):
    fn = getattr(models_vit, "vit_large_patch16", None)
    if callable(fn):
        return fn(
            num_classes=nb_classes, drop_path_rate=drop_path, global_pool=global_pool
        )
    print(
        "[WARN] models_vit.vit_large_patch16 not found; constructing VisionTransformer directly."
    )
    return _make_vit_large_patch16_from_class(nb_classes, drop_path, global_pool)


# ---------------------------------------


def load_model(
    checkpoint_path: str,
    nb_classes: int,
    drop_path: float = 0.0,
    global_pool: bool = True,
    device="cuda",
):
    model = _build_vit_large_patch16(
        nb_classes=nb_classes, drop_path=drop_path, global_pool=global_pool
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    # Interpolate pos-emb just in case
    interpolate_pos_embed(model, state)
    missing = model.load_state_dict(state, strict=False)
    # print("Missing after resume:", missing)

    model.to(device)
    model.eval()
    return model, ckpt


def plot_confusion(cm, class_names, out_png):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_roc(y_true, probs, class_names, out_png):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    n_classes = probs.shape[1]
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        ax.plot(fpr, tpr, label="class-1 vs rest")
    else:
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_true == i).astype(np.int32), probs[:, i])
            ax.plot(fpr, tpr, label=class_names[i])
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--test_subdir", type=str, default="test")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_tf = build_eval_transform(args.input_size)
    test_root = Path(args.data_root) / args.test_subdir
    test_ds = datasets.ImageFolder(root=test_root, transform=eval_tf)

    # Keep label order consistent with training if mapping exists
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = Path(args.checkpoint).parent / "class_mapping.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            mapping = json.load(f)
        class_names = mapping["classes"]
        name_to_idx_ckpt = {n: i for i, n in enumerate(class_names)}
        # remap targets to checkpoint ordering
        orig_classes = test_ds.classes
        remap = [name_to_idx_ckpt[c] for c in orig_classes]

        def target_map(t):
            return remap[t]

        test_ds = datasets.ImageFolder(
            root=test_root, transform=eval_tf, target_transform=target_map
        )
    else:
        class_names = test_ds.classes

    nb_classes = len(class_names)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model, ckpt = load_model(args.checkpoint, nb_classes=nb_classes, device=device)

    all_logits, all_y, all_paths = [], [], []
    with torch.no_grad():
        # capture file paths before DataLoader loops
        all_paths = [p for (p, _) in test_ds.samples]
        for imgs, ys in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_y.append(ys)

    logits = torch.cat(all_logits, 0).numpy()
    y_true = torch.cat(all_y, 0).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_pred = probs.argmax(1)

    acc = float(accuracy_score(y_true, y_pred))
    try:
        if nb_classes == 2:
            auroc = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            auroc = float(roc_auc_score(y_true, probs, multi_class="ovo"))
    except Exception:
        auroc = float("nan")
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    # Save metrics
    (out_dir / "metrics.json").write_text(
        json.dumps({"accuracy": acc, "macro_auroc": auroc, "report": report}, indent=2)
    )

    # Save predictions CSV
    import csv

    header = ["image_path", "true", "pred", "prob_pred"] + [
        f"prob_{c}" for c in class_names
    ]
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for path, t, p, row in zip(all_paths, y_true, y_pred, probs):
            w.writerow(
                [path, class_names[t], class_names[p], float(row[p])]
                + [float(x) for x in row]
            )

    # Plots
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")
    try:
        plot_roc(y_true, probs, class_names, out_dir / "roc_curve.png")
    except Exception:
        pass

    print(f"Done. Acc={acc:.4f}  Macro-AUROC={auroc:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
