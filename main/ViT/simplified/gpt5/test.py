# =============================
# test.py — Evaluate and export predictions
# =============================
# Usage example:
#   python test.py \
#     --data_dir /path/to/dataset \
#     --test_folder test \
#     --checkpoint ./runs/swinv2b_exp1/best.pt \
#     --output_dir ./runs/swinv2b_exp1

import argparse as _argparse
import csv as _csv
import os as _os
import json as _json
from typing import List as _List

import numpy as _np
import torch as _torch
import torch.nn as nn
from torch.utils.data import DataLoader as _DataLoader
from torchvision import datasets as _datasets, transforms as _transforms
from torchvision.models.swin_transformer import (
    swin_v2_b as _swin_v2_b,
    Swin_V2_B_Weights as _Swin_V2_B_Weights,
)


class _ImageFolderWithPaths(_datasets.ImageFolder):
    """ImageFolder that also returns the filename for each sample."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        path, _ = self.samples[index]  # (path, class_idx)
        fname = _os.path.basename(path)
        return img, target, fname


def _get_args():
    p = _argparse.ArgumentParser(description="Test Swin-V2-B and export predictions")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--test_folder", type=str, default="test")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best.pt produced by training",
    )
    p.add_argument("--output_dir", type=str, default="./runs/swinv2b")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def _load_model(ckpt_path: str, num_classes: int):
    weights = _Swin_V2_B_Weights.IMAGENET1K_V1
    model = _swin_v2_b(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    state = _torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    return model, weights


def _main():
    args = _get_args()
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    # Load classes / mapping from training artifacts if available
    classes_json = _os.path.join(_os.path.dirname(args.checkpoint), "classes.json")
    if _os.path.isfile(classes_json):
        with open(classes_json, "r") as f:
            meta = _json.load(f)
        classes: _List[str] = meta["classes"]
        train_class_to_idx = meta["class_to_idx"]
    else:
        # Fallback: infer classes from test folder ordering
        tmp_ds = _datasets.ImageFolder(_os.path.join(args.data_dir, args.test_folder))
        classes = tmp_ds.classes
        train_class_to_idx = tmp_ds.class_to_idx

    # Build model & eval transforms
    model, weights = _load_model(args.checkpoint, num_classes=len(classes))
    model.to(device)
    model.eval()

    try:
        eval_tfms = weights.transforms()
    except Exception:
        eval_tfms = _transforms.Compose(
            [
                _transforms.Resize(
                    256, interpolation=_transforms.InterpolationMode.BICUBIC
                ),
                _transforms.CenterCrop(256),
                _transforms.ToTensor(),
                _transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    test_dir = _os.path.join(args.data_dir, args.test_folder)
    test_ds = _ImageFolderWithPaths(test_dir, transform=eval_tfms)

    # Map test dataset label indices -> training order indices (to align metrics with model output order)
    test_cti = test_ds.class_to_idx  # {class_name: test_idx}
    test_idx_to_name = {v: k for k, v in test_cti.items()}
    map_test_to_train = {
        ti: train_class_to_idx[test_idx_to_name[ti]] for ti in test_idx_to_name
    }

    loader = _DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    out_dir = args.output_dir
    _os.makedirs(out_dir, exist_ok=True)
    csv_path = _os.path.join(out_dir, "test_predictions.csv")

    y_true_train_order = []
    y_pred_train_order = []

    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        header = ["filename"] + [f"prob_{c}" for c in classes] + ["pred_class"]
        writer.writerow(header)

        with _torch.no_grad():
            for images, targets_test_idx, fnames in loader:
                images = images.to(device)
                logits = model(images)
                probs = _torch.softmax(logits, dim=1).cpu()
                preds = probs.argmax(dim=1)  # in training/model class order

                # Convert test dataset targets into training/model order for fair comparison
                targets_train_idx = _torch.tensor(
                    [map_test_to_train[int(t)] for t in targets_test_idx],
                    dtype=_torch.long,
                )

                # write rows
                for i in range(images.size(0)):
                    row = (
                        [fnames[i]]
                        + [f"{p:.6f}" for p in probs[i].tolist()]
                        + [classes[preds[i].item()]]
                    )
                    writer.writerow(row)

                y_true_train_order.append(targets_train_idx)
                y_pred_train_order.append(preds.cpu())

    # Compute metrics in training/model label order
    y_true = _torch.cat(y_true_train_order).numpy()
    y_pred = _torch.cat(y_pred_train_order).numpy()
    acc = float((y_true == y_pred).mean())

    # Macro F1
    f1s = []
    for c in range(len(classes)):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1s.append(f1)
    macro_f1 = float(_np.mean(f1s))

    # Confusion matrix in training/model order
    cm = _np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Save metrics
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classes": classes,
        "note": "Metrics computed in training/model class order.",
    }
    with open(_os.path.join(out_dir, "test_metrics.json"), "w") as f:
        _json.dump(metrics, f, indent=2)

    # Save confusion matrix CSV
    with open(_os.path.join(out_dir, "confusion_matrix.csv"), "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["true/pred"] + classes)
        for i, c in enumerate(classes):
            writer.writerow([c] + [int(v) for v in cm[i]])

    print(f"Saved predictions to {csv_path}")
    print(f"Test accuracy: {acc:.4f} | macro-F1: {macro_f1:.4f}")


if __name__ == "__main__":
    _main()
