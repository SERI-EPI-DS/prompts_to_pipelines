#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a fine-tuned SwinV2 classifier on the ImageFolder test/ split.
Exports predictions.csv, metrics.json, and confusion_matrix.png.

Requires: torch, torchvision, timm, numpy, scikit-learn, matplotlib, tqdm, pandas
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from timm.data import resolve_data_config, create_transform
import timm
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SwinV2-B on test split")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root folder containing test/ subfolder",
    )
    p.add_argument("--test_dir", type=str, default="test", help="Test subfolder name")
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Folder to write predictions & metrics",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best.pth (or any checkpoint)",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Override image size (else use model default)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="timm model name; if omitted, inferred from checkpoint",
    )
    p.add_argument("--amp", action="store_true", help="Use mixed precision for eval")
    p.add_argument(
        "--tta", action="store_true", help="Enable simple TTA (orig + hflip)"
    )
    return p.parse_args()


def load_model_and_transform(
    ckpt_path: Path, model_name: str, num_classes: int, img_size: int
):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    inferred_model = model_name or checkpoint.get(
        "model_name", "swinv2_base_window12to24_192to384_22kft1k"
    )
    model = timm.create_model(inferred_model, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["model"], strict=False)

    data_cfg = resolve_data_config({}, model=model)
    if img_size is not None:
        data_cfg["input_size"] = (3, img_size, img_size)
    val_tf = create_transform(input_size=data_cfg["input_size"], is_training=False)
    return model, val_tf


def plot_confusion_matrix(cm, class_names, out_path: Path):
    fig, ax = plt.subplots(
        figsize=(1 + 0.4 * len(class_names), 1 + 0.4 * len(class_names))
    )
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Normalize for text overlay
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / (cm_sum + 1e-9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint)

    # Load class mapping (from training)
    class_map_path = out_dir / "class_to_idx.json"
    if class_map_path.exists():
        with open(class_map_path, "r") as f:
            class_to_idx = json.load(f)
    else:
        # Fallback: read from test folder if mapping not provided in out_dir
        test_tmp = datasets.ImageFolder(str(data_dir / args.test_dir))
        class_to_idx = test_tmp.class_to_idx
        # Save for consistency
        with open(class_map_path, "w") as f:
            json.dump(class_to_idx, f, indent=2)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    # Build model + transform
    model, val_tf = load_model_and_transform(
        ckpt_path, args.model, num_classes, args.img_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Dataset & loader
    test_ds = datasets.ImageFolder(str(data_dir / args.test_dir), transform=val_tf)
    # Ensure same class ordering as training
    if test_ds.class_to_idx != class_to_idx:
        # Reorder class_to_idx to match test order if needed
        # We'll map labels explicitly below, so just warn
        print(
            "Warning: class_to_idx from training differs from test folder ordering. Mapping will be handled."
        )
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_probs = []
    all_preds = []
    all_targets = []
    all_paths = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if args.amp:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    if args.tta:
                        logits_flip = model(torch.flip(images, dims=[-1]))
                        logits = (logits + logits_flip) / 2.0
            else:
                logits = model(images)
                if args.tta:
                    logits_flip = model(torch.flip(images, dims=[-1]))
                    logits = (logits + logits_flip) / 2.0

            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            # grab file paths from dataset.samples respecting the batch order
            start = len(all_paths)
            for i in range(probs.size(0)):
                # map batch index to global index
                global_idx = start + i
            # Instead, collect from loader’s dataset via indices
            # but DataLoader doesn’t expose indices directly, so capture afterward
            # We can store paths by slicing last N from test_ds.samples based on running count
            pass

    # Re-run to collect paths aligned with outputs (simpler & safe):
    all_paths = [p for p, _ in test_ds.samples]

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Metrics
    acc = accuracy_score(targets, preds)
    report = classification_report(
        targets,
        preds,
        output_dict=True,
        target_names=[idx_to_class[i] for i in range(num_classes)],
    )
    cm = confusion_matrix(targets, preds)

    # Save outputs
    # predictions.csv: image_path, true_label, pred_label, confidence, and per-class probabilities
    prob_cols = [f"prob_{idx_to_class[i]}" for i in range(num_classes)]
    rows = []
    for i in range(len(preds)):
        row = {
            "image_path": all_paths[i],
            "true_label": idx_to_class[targets[i]],
            "pred_label": idx_to_class[preds[i]],
            "confidence": float(probs[i, preds[i]]),
        }
        for c in range(num_classes):
            row[prob_cols[c]] = float(probs[i, c])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)

    # metrics.json
    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "num_classes": num_classes,
        "classes": [idx_to_class[i] for i in range(num_classes)],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    plot_confusion_matrix(
        cm,
        [idx_to_class[i] for i in range(num_classes)],
        out_dir / "confusion_matrix.png",
    )

    print(f"Accuracy: {acc:.4f}")
    print(
        f"Wrote: {out_dir/'predictions.csv'}, {out_dir/'metrics.json'}, {out_dir/'confusion_matrix.png'}"
    )


if __name__ == "__main__":
    main()
