"""
test_retfound_classifier.py
==========================

This script evaluates a fine‑tuned RETFound classifier on a test set of
images.  It loads a saved checkpoint from the training script, rebuilds
the Vision Transformer model with the correct number of classes and
reports a suite of metrics, including accuracy, AUC‑ROC, AUC‑PR,
F1‑score, sensitivity, specificity, precision, geometric mean and
Matthews correlation coefficient.  A normalised confusion matrix is
also saved as a PNG file.

Example usage::

    python test_retfound_classifier.py \
        --data_path /path/to/dataset \
        --checkpoint /path/to/finetuned_model/checkpoints/model_best.pth \
        --output_dir ./test_results

If your test split is named differently than ``test`` you can override
it using ``--test_folder``.  The class names are read from the
checkpoint if available, otherwise they are inferred from the folder
structure.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

import timm
from timm.data import create_transform

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    multilabel_confusion_matrix,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- Attempt to import the RETFound ViT wrapper. ---
# In many installations the ``models_vit`` module from the RETFound
# repository may not be available or may not expose ``vit_large_patch16``.
# We therefore mirror the fallback logic used in the training script.
vit_large_patch16 = None
try:
    import models_vit as _models_vit

    if hasattr(_models_vit, "vit_large_patch16"):
        vit_large_patch16 = _models_vit.vit_large_patch16
except Exception:
    vit_large_patch16 = None
if vit_large_patch16 is None:
    # Provide a simple wrapper around timm's ViT‑large model.  We attempt to
    # instantiate either ``vit_large_patch16_224`` or ``vit_large_patch16``
    # depending on the available timm version.
    def vit_large_patch16(
        num_classes: int, drop_path_rate: float = 0.0, global_pool: bool = True
    ):
        try:
            model = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
            )
        except Exception:
            model = timm.create_model(
                "vit_large_patch16",
                pretrained=False,
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
            )
        return model


try:
    from util.pos_embed import interpolate_pos_embed
except ImportError:
    # Attempt to import from a local ``util`` directory if present.
    util_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "util")
    sys.path.append(util_dir)
    from pos_embed import interpolate_pos_embed


def build_transforms(input_size: int = 224) -> transforms.Compose:
    """Create an evaluation transform.  This replicates the behaviour used
    during training and validation in the RETFound code.

    Parameters
    ----------
    input_size : int
        Input image size.

    Returns
    -------
    transforms.Compose
        The evaluation transform.
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def compute_misc_measures(
    confusion_matrix: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float]:
    """Compute macro‑averaged performance metrics from a multilabel confusion matrix.

    See the training script for a detailed description of the returned values.
    """
    accs = []
    sensitivities = []
    specificities = []
    precisions = []
    geometric_means = []
    f1s = []
    mccs = []
    for cm in confusion_matrix:
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        if total == 0:
            continue
        eps = 1e-7
        acc = (tp + tn) / total
        sensitivity = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        precision = tp / (tp + fp + eps)
        g_mean = np.sqrt(sensitivity * specificity)
        f1 = (2 * precision * sensitivity) / (precision + sensitivity + eps)
        mcc = ((tp * tn) - (fp * fn)) / (
            np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
        )
        accs.append(acc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)
        geometric_means.append(g_mean)
        f1s.append(f1)
        mccs.append(mcc)
    return (
        float(np.mean(accs)),
        float(np.mean(sensitivities)),
        float(np.mean(specificities)),
        float(np.mean(precisions)),
        float(np.mean(geometric_means)),
        float(np.mean(f1s)),
        float(np.mean(mccs)),
    )


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], output_path: str) -> None:
    """Create and save a normalised confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm_norm.shape[1]),
        yticks=np.arange(cm_norm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Normalized confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[dict, float]:
    """Evaluate a model on a dataset and compute a suite of metrics.
    Returns a metrics dictionary and the macro AUC‑ROC.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_probs: List[np.ndarray] = []
    all_targets: List[int] = []
    all_onehot: List[np.ndarray] = []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        probs = nn.Softmax(dim=1)(outputs)
        preds = probs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        all_onehot.append(
            nn.functional.one_hot(labels, num_classes=num_classes).cpu().numpy()
        )
    mean_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    probs_cat = np.concatenate(all_probs, axis=0)
    targets_cat = np.concatenate(all_targets, axis=0)
    onehot_cat = np.concatenate(all_onehot, axis=0)
    conf_matrix = multilabel_confusion_matrix(
        targets_cat, targets_cat, labels=list(range(num_classes))
    )
    acc, sensitivity, specificity, precision, g_mean, f1_val, mcc = (
        compute_misc_measures(conf_matrix)
    )
    try:
        auc_roc = roc_auc_score(
            onehot_cat, probs_cat, multi_class="ovr", average="macro"
        )
    except Exception:
        auc_roc = float("nan")
    try:
        auc_pr = average_precision_score(onehot_cat, probs_cat, average="macro")
    except Exception:
        auc_pr = float("nan")
    metrics = {
        "loss": mean_loss,
        "acc": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "g_mean": g_mean,
        "f1": f1_val,
        "mcc": mcc,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
    }
    return metrics, auc_roc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned RETFound classifier on a test set"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Root folder containing train/val/test splits",
    )
    parser.add_argument(
        "--test_folder",
        type=str,
        default="test",
        help="Name of the test split subdirectory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint saved by the training script",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_output",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu')"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else "cpu"
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Build dataset and infer classes
    test_dir = os.path.join(args.data_path, args.test_folder)
    transform = build_transforms()
    dataset_test = ImageFolder(test_dir, transform=transform)
    num_classes = len(dataset_test.classes)
    # Build model
    model = vit_large_patch16(
        num_classes=num_classes, drop_path_rate=0.0, global_pool=True
    )
    model.to(device)
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        # If class names were saved, use them
        class_names = ckpt.get("classes", dataset_test.classes)
    else:
        state_dict = ckpt
        class_names = dataset_test.classes
    # Interpolate position embeddings if necessary
    interpolate_pos_embed(model, state_dict)
    missing = model.load_state_dict(state_dict, strict=False)
    if missing.missing_keys:
        # Initialise missing classifier weights
        nn.init.trunc_normal_(model.head.weight, std=2e-5)
    # Create DataLoader
    test_loader = DataLoader(
        dataset_test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
    )
    criterion = nn.CrossEntropyLoss()
    metrics, _ = evaluate(model, criterion, test_loader, device, num_classes)
    print("Test metrics:", metrics)
    # Save metrics to CSV
    import csv

    metrics_csv = os.path.join(args.output_dir, "test_metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())
    # Build confusion matrix for plotting
    all_preds = []
    all_targets = []
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.numpy())
    preds_cat = np.concatenate(all_preds, axis=0)
    targets_cat = np.concatenate(all_targets, axis=0)
    # Build full confusion matrix (n_classes x n_classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(targets_cat, preds_cat):
        cm[true, pred] += 1
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
