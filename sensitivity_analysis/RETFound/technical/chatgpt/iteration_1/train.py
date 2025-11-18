"""
train.py
========

This script fine‑tunes the ViT‑Large backbone from the RETFound family on
a custom classification dataset of colour fundus images.  It expects
the dataset to be arranged in the following directory structure::

    data/
      train/
        class_a/  <-- images for class_a (any number of classes)
        class_b/
        ...
      val/
        class_a/
        class_b/
        ...
      test/
        ...    (used only in test.py)

The script reads the training and validation splits, prepares data
augmentations, initialises a vision transformer (ViT‑Large/16) model,
loads the RETFound foundation weights if provided, replaces the
classification head with a fresh one sized for the number of target
classes, and then runs a simple fine‑tuning loop using mixed precision.

All hyper‑parameters (batch size, number of epochs, learning rate,
weight decay, etc.) can be modified through command line arguments.
At the end of each epoch, the model is evaluated on the validation
set and the weights giving the best validation accuracy are saved
under ``best_model.pth`` in the specified output directory.

Usage example:

    python train.py \
        --data_path /path/to/data \
        --pretrained_weights ../RETFound/RETFound_CFP_weights.pth \
        --output_dir ./project/results \
        --epochs 50 --batch_size 32 --lr 1e-4

The training loop uses label smoothing when computing the
cross‑entropy loss as recommended by the RETFound authors【341698127874478†L329-L365】.

Note: This script assumes that PyTorch ≥2.0 and torchvision ≥0.15
are available in the environment and that a recent GPU (e.g., an
RTX3090) is accessible.  The RETFound pre‑trained weights file
``RETFound_CFP_weights.pth`` should correspond to the ViT‑Large
architecture; if it contains keys that do not match the default
torchvision implementation (e.g. different head names), these keys
will be ignored when loading.
"""

import argparse
import os
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    # torchvision may not be installed at authoring time, but is
    # expected to exist in the execution environment.  We import
    # conditionally so that type checkers do not complain.
    import torchvision
    from torchvision import transforms, datasets
except ImportError as exc:
    raise ImportError(
        "torchvision is required for train.py; please install it in your environment"
    ) from exc


def build_data_loaders(
    root_dir: str,
    input_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Prepare DataLoader objects for training and validation.

    Args:
        root_dir: Path to the root of the dataset containing ``train`` and ``val`` folders.
        input_size: Final spatial size (height and width) for the input images.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses used by the DataLoader.

    Returns:
        A tuple containing the training DataLoader, validation DataLoader and the list of class names.
    """
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Could not find 'train' and 'val' directories under {root_dir}."
        )

    # ImageNet statistics are commonly used as a baseline for normalisation
    # in medical imaging tasks when dataset‑specific statistics are not
    # available.  RETFound uses colour fundus photographs, which are
    # RGB images similar in range to natural images【341698127874478†L329-L365】.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Training augmentations: random crops, flips, rotations and colour
    # jitter improve robustness and mitigate over‑fitting on small
    # ophthalmic datasets【341698127874478†L329-L365】.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.14)),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Validation augmentations: deterministic resize and centre crop
    val_transforms = transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset: Dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset: Dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # The class indices are determined by the subfolder names
    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, class_names


def build_model(num_classes: int, pretrained_weights: str | None) -> nn.Module:
    """Create a ViT‑Large/16 model and load RETFound weights if provided.

    The classification head is replaced with a fresh Linear layer sized
    for ``num_classes``.  When loading the pre‑trained weights we
    remove any keys associated with the head if their shapes do not
    match the new head.  This prevents shape mismatch errors whilst
    retaining as much of the foundation model as possible.

    Args:
        num_classes: Number of target classes.
        pretrained_weights: Path to the RETFound weights file (or ``None``).

    Returns:
        A torch ``nn.Module`` ready for fine‑tuning.
    """
    # Use torchvision's ViT‑Large/16 implementation.  This ensures
    # compatibility with modern PyTorch versions.  If RETFound
    # architectures diverge significantly from this implementation,
    # additional adaptation may be required.
    model = torchvision.models.vit_l_16(weights=None)

    # Replace the classification head.  In torchvision's ViT models the
    # head is stored under model.heads.head.
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    if pretrained_weights:
        if not os.path.isfile(pretrained_weights):
            raise FileNotFoundError(
                f"Pre‑trained weights file not found: {pretrained_weights}"
            )
        state = torch.load(pretrained_weights, map_location="cpu")
        # Some checkpoints save under a 'model' or 'state_dict' key
        if isinstance(state, dict):
            # heuristically pick a sub‑dict containing weights
            if "model" in state:
                state_dict = state["model"]
            elif "state_dict" in state:
                state_dict = state["state_dict"]
            else:
                state_dict = state
        else:
            state_dict = state  # raw state dict
        # Filter out mismatched keys (e.g. classification head)
        model_state = model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered_state_dict[k] = v
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        if missing:
            # Warn the user that some keys were not found in the checkpoint
            print(
                f"Warning: {len(missing)} keys were not found in the provided weights and will be randomly initialised."
            )
        if unexpected:
            print(
                f"Warning: {len(unexpected)} keys from the checkpoint did not match the model architecture and were skipped."
            )

    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    """Run a single epoch of training.

    Returns the average loss and accuracy over the epoch.
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        # Backpropagation with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Statistics
        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        running_corrects += preds.eq(labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation set.

    Returns the average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        running_corrects += preds.eq(labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine‑tune RETFound ViT‑Large model for fundus image classification"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset root containing train/val subdirectories",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="Path to RETFound pre‑trained weights (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory where checkpoints and logs will be saved",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (max 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini‑batch size for training and validation",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Initial learning rate for the optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input image size (height and width)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for cross entropy loss",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Limit epochs to 50 to respect the specification
    if args.epochs > 50:
        print(
            f"Warning: epochs set to {args.epochs} exceeds the maximum of 50. Clipping to 50."
        )
        args.epochs = 50

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Prepare data loaders and obtain class names
    train_loader, val_loader, class_names = build_data_loaders(
        args.data_path, args.input_size, args.batch_size, args.num_workers
    )

    # Build model and load foundation weights
    model = build_model(
        num_classes=len(class_names), pretrained_weights=args.pretrained_weights
    )
    model.to(device)

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
            f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}"
        )

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "epoch": best_epoch,
                    "val_acc": best_val_acc,
                },
                checkpoint_path,
            )
            print(
                f"Saved new best model to {checkpoint_path} with val acc {best_val_acc:.4f}"
            )

    print(
        f"Training completed. Best validation accuracy: {best_val_acc:.4f} achieved at epoch {best_epoch}."
    )


if __name__ == "__main__":
    main()
