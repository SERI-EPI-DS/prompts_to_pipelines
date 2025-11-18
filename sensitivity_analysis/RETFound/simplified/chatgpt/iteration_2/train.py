"""
train_retfound_classifier.py
===========================

This script fine‑tunes the RETFound foundation model for a classification
task on retinal images.  It expects your dataset to be organised into
training, validation and test splits, with one subdirectory per class in
each split (e.g. ``dataset/train/normal/``, ``dataset/train/disease/``).

The model architecture is based on the Vision Transformer (ViT) used
within the original RETFound repository.  We load the pre‑trained
foundation weights and replace the classification head to suit your
number of classes.  Training runs for a configurable number of epochs
and saves the best performing checkpoint (based on validation AUC) into
``output_dir``.  After training, the model is evaluated on the test
split and the results (accuracy, AUC‑ROC, AUC‑PR, F1‑score, MCC and
confusion matrix) are written to disk.

Usage example::

    python train_retfound_classifier.py \
        --data_path /path/to/dataset \
        --output_dir ./finetuned_model \
        --pretrained_weights /path/to/RETFound_cfp_weights.pth \
        --epochs 50 --batch_size 16

The script will automatically detect the number of classes from the
training subdirectories.  If your splits are named differently than
``train``, ``val`` and ``test`` you can override them using
``--train_folder``, ``--val_folder`` and ``--test_folder``.  The
default hyperparameters follow the recommendations in the RETFound
paper and official fine‑tuning code (e.g. 50 epochs, base learning
rate 5e‑3, layer decay 0.65, weight decay 0.05 and drop‑path 0.2)【286788897147266†L345-L365】.

"""

import argparse
import datetime
import os
import sys
from pathlib import Path
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

import timm
from timm.data import create_transform

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    multilabel_confusion_matrix,
)
import matplotlib

matplotlib.use("Agg")  # allow headless plotting
import matplotlib.pyplot as plt

# --- Utilities from the RETFound repository ---
# To load the RETFound architecture we first attempt to import
# ``vit_large_patch16`` from ``models_vit``.  If that module is present but
# does not provide the symbol, or if the import fails entirely, we fall
# back to using the ViT‑large implementation from timm.
vit_large_patch16 = None
try:
    import models_vit as _models_vit

    # Check that the module actually defines ``vit_large_patch16``.
    if hasattr(_models_vit, "vit_large_patch16"):
        vit_large_patch16 = _models_vit.vit_large_patch16
except Exception:
    # Either ``models_vit`` could not be imported or another error was raised;
    # we'll handle this by falling back to timm.
    vit_large_patch16 = None

if vit_large_patch16 is None:
    # Define a simple wrapper around timm's ViT‑large model.  This
    # fallback assumes timm is installed and available.  It attempts to
    # instantiate either ``vit_large_patch16_224`` (preferred) or
    # ``vit_large_patch16`` depending on the timm version.  The drop path
    # rate is passed through when supported.  Global pooling is ignored as
    # timm does not provide this option directly.
    def vit_large_patch16(
        num_classes: int, drop_path_rate: float = 0.0, global_pool: bool = True
    ):
        try:
            # Some versions of timm use the ``_224`` suffix for the 224×224
            # Vision Transformer.  If this fails, we try the base name.
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
    # As above, try to append the local util directory
    util_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "util")
    sys.path.append(util_dir)
    try:
        from pos_embed import interpolate_pos_embed
    except Exception as exc:
        raise ImportError(
            "Could not import util.pos_embed. Ensure the util directory from the "
            "RETFound_MAE repository is available."
        ) from exc

try:
    from util.lr_decay import param_groups_lrd
except ImportError:
    # Similarly attempt to import from local util
    try:
        from lr_decay import param_groups_lrd
    except Exception as exc:
        raise ImportError(
            "Could not import util.lr_decay. Ensure the util directory from the "
            "RETFound_MAE repository is available."
        ) from exc


def build_transforms(
    input_size: int = 224,
    color_jitter: float = 0.4,
    auto_augment: str = "rand-m9-mstd0.5-inc1",
    reprob: float = 0.25,
    remode: str = "pixel",
    recount: int = 1,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create training and evaluation transforms matching the official code.

    Parameters
    ----------
    input_size : int
        The size of the square crop fed into the network.
    color_jitter : float
        How much to jitter the colour channels during augmentation.
    auto_augment : str
        The auto‑augmentation policy; defaults to the policy used in the
        original paper.
    reprob : float
        Probability of random erasing.
    remode : str
        Random erasing mode.
    recount : int
        Random erasing count.

    Returns
    -------
    Tuple[transforms.Compose, transforms.Compose]
        A tuple containing the training transform and the evaluation
        transform.
    """
    # Mean and standard deviation values for ImageNet, used in the RETFound
    # implementation for fundus images【292752853816633†L20-L35】.
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # Build training transform using timm's helper.  This mirrors
    # `util.datasets.build_transform` in the original code【993159648397992†L18-L35】.
    train_transform = create_transform(
        input_size=input_size,
        is_training=True,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation="bicubic",
        re_prob=reprob,
        re_mode=remode,
        re_count=recount,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    # Evaluation transform: resize and centre crop, then normalise.
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    eval_transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def compute_misc_measures(
    confusion_matrix: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float]:
    """Compute accuracy, sensitivity, specificity, precision, geometric mean,
    F1‑score and Matthews correlation coefficient (MCC) from a multilabel
    confusion matrix.

    This function mirrors the behaviour in ``engine_finetune.misc_measures``.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        A 3‑D array with shape (n_classes, 2, 2), as returned by
        ``sklearn.metrics.multilabel_confusion_matrix``.

    Returns
    -------
    Tuple[float, float, float, float, float, float, float]
        Tuple containing the macro‑averaged values of accuracy, sensitivity,
        specificity, precision, geometric mean, F1‑score and MCC.
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
        acc = (tp + tn) / total
        # Avoid division by zero by adding a small epsilon.
        eps = 1e-7
        sensitivity = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        precision = tp / (tp + fp + eps)
        geometric_mean = np.sqrt(sensitivity * specificity)
        f1 = (2 * precision * sensitivity) / (precision + sensitivity + eps)
        # Matthews correlation coefficient
        mcc = ((tp * tn) - (fp * fn)) / (
            np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
        )
        accs.append(acc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)
        geometric_means.append(geometric_mean)
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
    """Plot and save a normalised confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes).
    classes : list of str
        The class names in the order corresponding to the rows/columns of ``cm``.
    output_path : str
        Path to save the resulting figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # Label ticks
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
    # Annotate cells
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


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[float, float]:
    """Run one training epoch and return loss and accuracy.

    Parameters
    ----------
    model : nn.Module
        The network to train.
    criterion : nn.Module
        Loss function (e.g. cross entropy).
    dataloader : DataLoader
        DataLoader for the training set.
    optimizer : torch.optim.Optimizer
        Optimiser to use.
    device : torch.device
        Device to run the computation on.
    scaler : torch.cuda.amp.GradScaler
        GradScaler for mixed precision training.

    Returns
    -------
    Tuple[float, float]
        A tuple of (mean_loss, accuracy) over the epoch.
    """
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    mean_loss = running_loss / total_samples
    accuracy = total_correct / total_samples
    return mean_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[dict, float]:
    """Evaluate a model on a dataset and compute various metrics.

    Parameters
    ----------
    model : nn.Module
        The network to evaluate.
    criterion : nn.Module
        Loss function (e.g. cross entropy).
    dataloader : DataLoader
        DataLoader for the evaluation split.
    device : torch.device
        Device to run the computation on.
    num_classes : int
        Number of target classes.

    Returns
    -------
    Tuple[dict, float]
        A dictionary of aggregated metrics and the macro AUC‑ROC.  The
        dictionary keys include ``loss``, ``acc`` (accuracy), ``sensitivity``,
        ``specificity``, ``precision``, ``g_mean``, ``f1``, and ``mcc``.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_probs: List[np.ndarray] = []
    all_targets: List[int] = []
    # Lists to accumulate one‑hot encoded labels for AUC/PR curves
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
        onehot = nn.functional.one_hot(labels, num_classes=num_classes)
        all_onehot.append(onehot.cpu().numpy())
    mean_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    # Concatenate accumulated tensors
    probs_cat = np.concatenate(all_probs, axis=0)
    targets_cat = np.concatenate(all_targets, axis=0)
    onehot_cat = np.concatenate(all_onehot, axis=0)
    # Compute confusion matrix and derived measures
    conf_matrix = multilabel_confusion_matrix(
        targets_cat, targets_cat, labels=list(range(num_classes))
    )
    acc, sensitivity, specificity, precision, g_mean, f1_val, mcc = (
        compute_misc_measures(conf_matrix)
    )
    # Compute AUC‑ROC and AUC‑PR; use try/except because they can fail
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


def load_pretrained_weights(model: nn.Module, weights_path: str) -> None:
    """Load pre‑trained weights into the model, interpolating position embeddings.

    The function removes the classifier head from the checkpoint (if the
    number of classes differs) and interpolates the positional embeddings
    when the input size changes.  This logic follows the official
    fine‑tuning script【685987010505304†L278-L305】.

    Parameters
    ----------
    model : nn.Module
        The ViT model into which to load the checkpoint.
    weights_path : str
        Path to the ``RETFound_cfp_weights.pth`` file.
    """
    checkpoint = torch.load(weights_path, map_location="cpu")
    if "model" in checkpoint:
        checkpoint_model = checkpoint["model"]
    else:
        # allow loading of state_dict directly
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    # Remove parameters from checkpoint whose shape doesn't match the target model
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]
    # Interpolate position embeddings when the number of patches differs
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    # Initialise the classification head weights when they were missing
    if "head.weight" in msg.missing_keys:
        nn.init.trunc_normal_(model.head.weight, std=2e-5)
    return


def save_metrics_csv(metrics: dict, output_path: str) -> None:
    """Save metric dictionary as a single‑row CSV file.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric values.
    output_path : str
        Path to the CSV file to write.  If the file already exists it will
        be overwritten.
    """
    import csv

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine‑tune RETFound for image classification"
    )
    # Data parameters
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Root folder containing train/val/test splits",
    )
    parser.add_argument(
        "--train_folder",
        type=str,
        default="train",
        help="Name of the training split subdirectory",
    )
    parser.add_argument(
        "--val_folder",
        type=str,
        default="val",
        help="Name of the validation split subdirectory",
    )
    parser.add_argument(
        "--test_folder",
        type=str,
        default="test",
        help="Name of the test split subdirectory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetune_output",
        help="Directory to save checkpoints and results",
    )
    # Model and optimisation parameters
    parser.add_argument(
        "--model",
        type=str,
        default="vit_large_patch16",
        help="Model architecture to fine‑tune (vit_large_patch16)",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        required=True,
        help="Path to RETFound foundation model weights (.pth)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (recommended 50)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size per gradient step"
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=5e-3,
        help="Base learning rate (will be scaled by batch size)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for AdamW optimiser",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.65,
        help="Layer‑wise learning rate decay factor",
    )
    parser.add_argument("--drop_path", type=float, default=0.2, help="Drop path rate")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes for the DataLoader",
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
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # Data directories
    train_dir = os.path.join(args.data_path, args.train_folder)
    val_dir = os.path.join(args.data_path, args.val_folder)
    test_dir = os.path.join(args.data_path, args.test_folder)
    # Create transforms
    train_transform, eval_transform = build_transforms()
    # Load datasets
    dataset_train = ImageFolder(train_dir, transform=train_transform)
    dataset_val = ImageFolder(val_dir, transform=eval_transform)
    dataset_test = ImageFolder(test_dir, transform=eval_transform)
    num_classes = len(dataset_train.classes)
    # Data loaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # Build model
    if args.model != "vit_large_patch16":
        raise ValueError("Only vit_large_patch16 is supported in this script.")
    model = vit_large_patch16(
        num_classes=num_classes, drop_path_rate=args.drop_path, global_pool=True
    )
    # Load foundation weights
    load_pretrained_weights(model, args.pretrained_weights)
    model.to(device)
    # Compute learning rate scaled by batch size as in the official script
    eff_batch_size = args.batch_size
    lr = args.blr * eff_batch_size / 256
    # Set up layer‑wise learning rate decay
    param_groups = param_groups_lrd(
        model,
        weight_decay=args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    # Apply lr_scale to parameter groups
    for pg in param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Containers for tracking best model
    best_auc = 0.0
    best_epoch = -1
    # Logging lists
    history = []
    start_time = time.time()
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, criterion, train_loader, optimizer, device, scaler
        )
        # Validate
        val_metrics, val_auc = evaluate(
            model, criterion, val_loader, device, num_classes
        )
        # Checkpoint if AUC improves
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            checkpoint_path = os.path.join(checkpoint_dir, "model_best.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "num_classes": num_classes,
                    "classes": dataset_train.classes,
                },
                checkpoint_path,
            )
        # Append to history
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                **val_metrics,
            }
        )
        # Print progress
        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['acc']:.4f}, "
            f"val_auc={val_metrics['auc_roc']:.4f}"
        )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(
        f"Training completed in {total_time_str}. Best validation AUC={best_auc:.4f} at epoch {best_epoch}."
    )
    # Save training history
    import json

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    # Evaluate on test set using the best checkpoint
    # Load best model
    best_ckpt = torch.load(
        os.path.join(checkpoint_dir, "model_best.pth"), map_location=device
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics, _ = evaluate(model, criterion, test_loader, device, num_classes)
    print("Test metrics:", test_metrics)
    # Save metrics to CSV
    metrics_csv_path = os.path.join(args.output_dir, "test_metrics.csv")
    save_metrics_csv(test_metrics, metrics_csv_path)
    # Compute confusion matrix and plot
    # Recompute predictions to build confusion matrix for test set
    model.eval()
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
    cm = multilabel_confusion_matrix(
        targets_cat, preds_cat, labels=list(range(num_classes))
    )
    # Convert multilabel confusion matrix to a single confusion matrix by summing
    # along the binary axes.  This yields an (n_classes, n_classes) matrix.
    # For each true label i and predicted label j we sum over matches.
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(targets_cat, preds_cat):
        conf_matrix[true, pred] += 1
    plot_path = os.path.join(args.output_dir, "confusion_matrix_test.png")
    plot_confusion_matrix(conf_matrix, dataset_train.classes, plot_path)
    print(f"Saved test confusion matrix to {plot_path}")


if __name__ == "__main__":
    main()
