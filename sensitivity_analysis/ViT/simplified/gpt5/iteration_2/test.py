## test\_swinv2\_b.py
#!/usr/bin/env python3
"""
Evaluate a trained Swin-V2-B classifier on a held-out ImageFolder test set.

Expected directory:
    DATA_DIR/
      test/
        class_a/ ...
        class_b/ ...
        ...

Outputs:
- metrics.json: accuracy and (if scikit-learn installed) precision/recall/F1 & ROC-AUC (macro)
- predictions.csv: per-image predictions with probabilities
- confusion_matrix.csv: confusion matrix (rows=true, cols=pred)

Requirements: torch, torchvision, numpy, pandas (for CSV), optionally scikit-learn
"""

import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

try:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    HAVE_SK = True
except Exception:
    HAVE_SK = False


def build_eval_transform(img_size: int, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15), antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_model(weights_path: str, num_classes: int, device: torch.device):
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    model = swin_v2_b(weights=weights)
    in_feats = model.head.in_features
    model.head = nn.Linear(in_feats, num_classes)

    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval().to(device)

    img_size = ckpt.get("img_size", 256)
    return model, img_size


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--data_dir", type=str, required=True, help="Root folder containing test/"
    )
    p.add_argument("--test_subdir", type=str, default="test")
    p.add_argument(
        "--weights", type=str, required=True, help="Path to model_best.pth or last.pth"
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--classes_json",
        type=str,
        default="",
        help="Optional path to classes.json saved during training",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine classes / label mapping
    test_dir = os.path.join(args.data_dir, args.test_subdir)
    tmp_ds = datasets.ImageFolder(test_dir)  # to read classes from disk
    if args.classes_json and os.path.isfile(args.classes_json):
        with open(args.classes_json, "r") as f:
            saved = json.load(f)
        saved_classes = saved.get("classes")
        if saved_classes is not None:
            # Ensure test classes align with training
            if saved_classes != tmp_ds.classes:
                print(
                    "[WARN] Class names/order differ between training and test. Proceeding with training classes; mapping by name."
                )
                # Build index mapping by class name
                name_to_idx_test = {name: i for i, name in enumerate(tmp_ds.classes)}
                # Reorder
                tmp_ds.class_to_idx = {name: i for i, name in enumerate(saved_classes)}
                tmp_ds.classes = list(saved_classes)
                # Remap samples' targets to training order
                new_samples = []
                for path, _ in tmp_ds.samples:
                    cls_name = os.path.basename(os.path.dirname(path))
                    new_samples.append((path, tmp_ds.class_to_idx[cls_name]))
                tmp_ds.samples = new_samples
                tmp_ds.targets = [t for _, t in new_samples]
    classes = tmp_ds.classes
    num_classes = len(classes)

    # Load model
    model, inferred_img = load_model(
        args.weights, num_classes=num_classes, device=device
    )

    # Build transforms consistent with training
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std = weights.meta.get("std", [0.229, 0.224, 0.225])
    eval_tfms = build_eval_transform(inferred_img, mean, std)

    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    paths = [s[0] for s in test_ds.samples]
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
            y_true.append(targets.cpu().numpy())
            y_pred.append(torch.argmax(probs, dim=1).cpu().numpy())
            y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    # ---- Metrics ----
    acc = float((y_true == y_pred).mean())
    metrics = {"accuracy": acc}

    if HAVE_SK:
        report = classification_report(
            y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
        )
        metrics.update(
            {
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
            }
        )
        try:
            # ROC-AUC macro for multiclass (requires probability scores)
            y_true_onehot = np.eye(num_classes)[y_true]
            auc_macro = roc_auc_score(
                y_true_onehot, y_prob, average="macro", multi_class="ovr"
            )
            metrics["macro_roc_auc"] = float(auc_macro)
        except Exception:
            pass

        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    else:
        # Basic confusion matrix (numpy)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

    # Save metrics and artifacts
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                k: (float(v) if isinstance(v, (np.floating, np.ndarray)) else v)
                for k, v in metrics.items()
            },
            f,
            indent=2,
        )

    # Save confusion matrix CSV (rows=true, cols=pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(os.path.join(args.output_dir, "confusion_matrix.csv"))

    # Save per-image predictions
    prob_cols = [f"prob_{cls}" for cls in classes]
    pred_rows = []
    for path, t, p, probs in zip(paths, y_true, y_pred, y_prob):
        row = {
            "image_path": path,
            "true_label": classes[int(t)],
            "pred_label": classes[int(p)],
            "pred_index": int(p),
            "confidence": float(np.max(probs)),
        }
        row.update({pc: float(val) for pc, val in zip(prob_cols, probs)})
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    print(f"Test accuracy: {acc:.4f}")
    print(
        f"Wrote: metrics.json, predictions.csv, confusion_matrix.csv in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
