#!/usr/bin/env python3
"""
train_convnext.py
===================

This script fine‑tunes a **ConvNeXt‑Large** image classification model on a
user–provided image dataset organised into ``train`` and ``val`` folders.  The
model used here is the **ConvNeXt‑L** variant introduced by Liu et al. The
ConvNeXt family modernises classical convolutional networks by incorporating
design principles from Transformer models.  As the Hugging Face model card
explains, ConvNeXt is a pure convolutional network inspired by Vision
Transformers; it begins with a ResNet and “modernises” its design by
borrowing ideas from the Swin Transformer【230255674542041†L69-L72】.

The script expects the dataset directory to follow this structure::

    data_root/
        train/
            class_a/
                img1.jpg
                img2.jpg
                ...
            class_b/
                ...
        val/
            class_a/
                ...
            class_b/
                ...

Each class must be a folder containing the corresponding images.  This layout
matches the default directory scheme used by ``torchvision.datasets.ImageFolder``,
which loads images arranged in ``root/class_x/xxx.png`` style folders【806275724524080†L1609-L1623】.

Key features of this training script:

* **Pre‑trained ConvNeXt‑Large base:** loads the ConvNeXt‑L model
  pre‑trained on ImageNet and replaces its classification head with a new
  ``nn.Linear`` layer sized to the number of classes.  The model benefits from
  the high image classification accuracy of ConvNeXt‑L (e.g. ~87.8 % on
  ImageNet‑22k【247051686019952†L602-L631】).

* **Data augmentation:** uses common image augmentations (random resized crops,
  horizontal/vertical flips, colour jitter, rotation) to improve generalisation.
  Input images are resized to 256 × 256, randomly cropped to 224 × 224, and
  normalised using ImageNet statistics (mean and standard deviation).  For
  evaluation (validation), a deterministic resize and centre crop are applied.

* **Class imbalance handling:** optionally computes class weights from the
  training data and applies them either via a weighted loss
  (``CrossEntropyLoss`` with ``weight``) or by using a ``WeightedRandomSampler``.

* **Command‑line interface:** configurable via arguments for data directory,
  output directory, number of epochs, batch size, learning rate, weight decay,
  and whether to freeze the backbone.

* **Checkpointing:** saves the model with the best validation accuracy to the
  specified output directory as ``best_model.pth``.  Training logs are
  written to ``training_log.txt``.

Usage example::

    python train_convnext.py \
        --data-dir path/to/dataset \
        --output-dir path/to/output \
        --epochs 50 \
        --batch-size 16 \
        --lr 1e-4 \
        --freeze-base

The script uses PyTorch and Torchvision.  Install dependencies before running:

    pip install torch torchvision

Note: Although this script has been tested conceptually, it may require
adjustment depending on your computing environment (CPU/GPU, CUDA version).
"""

import argparse
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine‑tune a ConvNeXt‑Large model on a custom image dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing 'train' and 'val' subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Mini‑batch size (default: 16)."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Initial learning rate (default: 1e‑4)."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for AdamW optimiser (default: 1e‑5).",
    )
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Freeze the backbone (all layers except the final classifier).",
    )
    parser.add_argument(
        "--use-weighted-loss",
        action="store_true",
        help="Use class weights in the loss function to mitigate class imbalance.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes (default: 4).",
    )
    return parser.parse_args()


def build_transforms(
    input_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create training and validation transforms.

    Parameters
    ----------
    input_size : int
        Size to which images are resized/cropped (default: 224).

    Returns
    -------
    Tuple[transforms.Compose, transforms.Compose]
        A tuple (train_transforms, val_transforms).
    """
    # Data augmentation inspired by common ImageNet recipes.  Images are
    # randomly resized and cropped, with flips and colour jitter.  At
    # inference time, we use a deterministic resize followed by centre crop
    # consistent with the official ConvNeXt inference transforms【856698280327426†L160-L166】.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(
                input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms, val_transforms


def create_dataloaders(
    data_dir: str, batch_size: int, num_workers: int, use_weighted_sampler: bool = False
) -> Tuple[Dict[str, DataLoader], List[str], torch.Tensor]:
    """Create dataloaders for training and validation.

    Parameters
    ----------
    data_dir : str
        Root directory containing ``train`` and ``val`` subdirectories.
    batch_size : int
        Batch size for data loading.
    num_workers : int
        Number of worker processes for data loading.
    use_weighted_sampler : bool
        If True, uses a ``WeightedRandomSampler`` for the training loader to
        mitigate class imbalance.

    Returns
    -------
    Tuple[Dict[str, DataLoader], List[str], torch.Tensor]
        Dataloaders for training and validation, list of class names, and class
        weights tensor (useful if using weighted loss).
    """
    train_transforms, val_transforms = build_transforms(input_size=224)
    datasets_dict = {}
    image_datasets = {}
    class_weights = None
    for phase in ["train", "val"]:
        phase_dir = os.path.join(data_dir, phase)
        if not os.path.isdir(phase_dir):
            raise FileNotFoundError(
                f"Expected subdirectory '{phase}' in '{data_dir}', but it does not exist."
            )
        transforms_to_use = train_transforms if phase == "train" else val_transforms
        image_datasets[phase] = datasets.ImageFolder(
            phase_dir, transform=transforms_to_use
        )
    class_names = image_datasets["train"].classes
    # Compute class weights for weighted loss and sampler
    targets = [sample[1] for sample in image_datasets["train"].samples]
    num_samples_per_class = torch.bincount(torch.tensor(targets, dtype=torch.long))
    class_freq = num_samples_per_class.float()
    # Avoid division by zero in case a class has zero samples
    class_freq[class_freq == 0] = 1.0
    # Inverse frequency for weighting; multiply by mean to keep magnitude reasonable
    class_weights = (1.0 / class_freq) * (class_freq.mean())
    # WeightedRandomSampler if requested
    if use_weighted_sampler:
        sample_weights = class_weights[torch.tensor(targets)]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    val_loader = DataLoader(
        image_datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    dataloaders = {"train": train_loader, "val": val_loader}
    return dataloaders, class_names, class_weights


def build_model(num_classes: int, freeze_base: bool = False) -> nn.Module:
    """Load the pretrained ConvNeXt‑Large model and replace its classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    freeze_base : bool
        If True, freeze all parameters except the final classifier.

    Returns
    -------
    nn.Module
        The fine‑tunable ConvNeXt‑L model.
    """
    # Load pre‑trained ConvNeXt‑L model (large variant) from torchvision
    # The weights are pretrained on ImageNet (IMAGENET1K_V1).  According to
    # the Torchvision documentation, the inference transforms rescale and
    # normalise inputs【856698280327426†L160-L166】; we match those statistics in
    # our own transforms.
    weights = models.ConvNeXt_Large_Weights.DEFAULT
    model = models.convnext_large(weights=weights)
    # Replace the classifier: ConvNeXt has a 'classifier' attribute with an
    # ``nn.Linear`` as the last element.  We retrieve its input dimension and
    # create a new linear layer with ``num_classes`` outputs.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    # Optionally freeze the backbone
    if freeze_base:
        for name, param in model.named_parameters():
            # Only allow gradients for parameters in the classifier
            if "classifier" not in name:
                param.requires_grad = False
    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch.

    Returns the average loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += inputs.size(0)
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model on a validation set.

    Returns
    -------
    Tuple[float, float]
        Average loss and accuracy on the validation set.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += inputs.size(0)
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def main() -> None:
    args = parse_args()
    # Create output directory and log file
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.txt")
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dataloaders
    dataloaders, class_names, class_weights = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.use_weighted_loss,
    )
    num_classes = len(class_names)
    # Build model
    model = build_model(num_classes=num_classes, freeze_base=args.freeze_base)
    model = model.to(device)
    # Loss function; apply class weights if requested
    if args.use_weighted_loss:
        # Normalize weights to sum to number of classes
        weights = class_weights / class_weights.sum() * num_classes
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    # Optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Use a cosine annealing scheduler to gradually reduce the learning rate
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Training loop
    best_acc = 0.0
    best_epoch = 0
    best_model_wts = None
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Training started at {datetime.now().isoformat()}\n")
        log_file.write(f"Number of classes: {num_classes}\n")
        log_file.write(f"Classes: {class_names}\n\n")
        for epoch in range(args.epochs):
            start_time = time.time()
            train_loss, train_acc = train_epoch(
                model, dataloaders["train"], criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, dataloaders["val"], criterion, device)
            scheduler.step()
            epoch_time = time.time() - start_time
            # Logging
            log_msg = (
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"Time: {epoch_time/60:.2f}m\n"
            )
            print(log_msg, end="")
            log_file.write(log_msg)
            # Checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                best_model_wts = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        log_file.write(
            f"\nTraining complete. Best val acc: {best_acc:.4f} at epoch {best_epoch}\n"
        )
    # Save best model weights
    if best_model_wts is not None:
        best_model_path = os.path.join(args.output_dir, "best_model.pth")
        torch.save(best_model_wts, best_model_path)
        print(f"Best model saved to {best_model_path}")
        # Save class names alongside the model for inference.  The list of class
        # names is stored in alphabetical order corresponding to the index of
        # each output neuron.  Saving this mapping makes it possible for the
        # test script to decode predictions.
        import json  # Local import to avoid polluting global namespace.

        class_file = os.path.join(args.output_dir, "class_names.json")
        with open(class_file, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)
        print(f"Class names saved to {class_file}")
    else:
        print("No model was trained. Did you specify the correct data directories?")


if __name__ == "__main__":
    main()
