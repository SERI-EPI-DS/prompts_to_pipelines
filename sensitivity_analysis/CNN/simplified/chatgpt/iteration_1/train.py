"""
train_convnext.py
===================

This script fine‑tunes a ConvNeXt‑L model on a custom image classification
dataset.  It expects the dataset to be organised into ``train``, ``val`` and
optional ``test`` directories, each containing one subdirectory per class.  The
script uses PyTorch and the ``timm`` library to load a pre‑trained
ConvNeXt‑L backbone.  Images are resized and normalised using
``timm.data.create_transform``, which automatically applies the correct
transformations for the chosen model.  The ConvNeXt Large model expects
224×224 pixel RGB images during training and 288×288 pixel images at
inference time【587683307028815†L124-L138】【587683307028815†L264-L277】.  The
transform factory from ``timm`` ensures these requirements are met.

The script accepts a number of command line arguments to control the
training process.  You can specify the location of your dataset, where to
store outputs, the number of epochs, batch size, learning rate, etc.  During
training, it reports progress via a progress bar and logs the running
accuracy and loss on both the training and validation sets.  The best
performing model (by validation accuracy) is saved to the output
directory along with a JSON file containing the class name to index
mapping.

Usage example::

    python train_convnext.py \
        --data-dir /path/to/dataset \
        --output-dir ./results \
        --epochs 20 \
        --batch-size 32

Requirements::

    pip install timm torch torchvision tqdm pandas scikit-learn

"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


def prepare_dataloaders(
    data_dir: Path,
    model_name: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """Initialise training and validation dataloaders.

    Parameters
    ----------
    data_dir : Path
        Root directory of the dataset containing ``train`` and ``val`` folders.
    model_name : str
        Name of the ConvNeXt model variant to fine‑tune.  This controls the
        appropriate input size and normalisation statistics.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of worker processes used to load the data.

    Returns
    -------
    Tuple[DataLoader, DataLoader, Dict[int, str]]
        A tuple containing the training loader, validation loader and a
        dictionary mapping class indices to human‑readable labels.
    """
    # Create the model to resolve its data configuration.  We don't care
    # about the number of classes at this stage, so we temporarily set
    # ``num_classes=0`` (feature extraction mode).  The transforms will be
    # identical for a classification head since they depend only on the
    # backbone configuration (input_size, mean, std, etc.).
    tmp_model = timm.create_model(model_name, pretrained=True, num_classes=0)

    # Resolve the data configuration associated with this model.  The
    # configuration includes the expected input resolution (e.g. 224×224 for
    # training) and the normalisation statistics.  Using
    # ``resolve_data_config`` is recommended over hand‑crafting transforms
    # because pretrained models may employ different cropping percentages or
    # interpolation modes【670582884494607†L170-L243】.
    data_config = timm.data.resolve_data_config({}, model=tmp_model)

    # Create separate transforms for training and validation.  When
    # ``is_training=True``, ``create_transform`` applies random cropping and
    # augmentation such as random horizontal flips, whereas when
    # ``is_training=False`` it applies centre cropping.  These defaults are
    # appropriate for fine‑tuning on a dataset with relatively limited
    # augmentation needs【670582884494607†L170-L243】【587683307028815†L264-L277】.
    train_transform = timm.data.create_transform(**data_config, is_training=True)
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    # Construct ImageFolder datasets.  The ImageFolder class infers class
    # names from the subdirectory names.  It will assign integer labels in
    # alphabetical order of the subfolders.
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find 'train' and 'val' directories under {data_dir}. "
            "Ensure the dataset is organised as dataset/train/<class> and dataset/val/<class>."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Map from class index to class name for later use (saving predictions)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    return train_loader, val_loader, idx_to_class


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation set.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    dataloader : DataLoader
        Validation dataloader.
    criterion : nn.Module
        Loss function used to compute the average loss.
    device : torch.device
        Device on which tensors should be located.

    Returns
    -------
    Tuple[float, float]
        A tuple (avg_loss, accuracy) giving the mean validation loss and
        classification accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epochs: int,
    output_dir: Path,
    idx_to_class: Dict[int, str],
):
    """Main training loop.

    Trains the model for the specified number of epochs, evaluating on the
    validation set after each epoch.  The best performing model (based on
    validation accuracy) is saved to the output directory.

    Parameters
    ----------
    model : nn.Module
        Neural network to train.
    train_loader : DataLoader
        Training dataloader.
    val_loader : DataLoader
        Validation dataloader.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimiser such as AdamW.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler (OneCycleLR) used to adjust the learning rate
        during training【453849609406327†L1493-L1523】.
    device : torch.device
        CPU or GPU.
    epochs : int
        Number of full passes over the training data.
    output_dir : Path
        Directory where model checkpoints and logs will be saved.
    idx_to_class : Dict[int, str]
        Mapping from numeric class indices to class names.
    """
    best_val_acc = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save class mapping for inference
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(idx_to_class, f, indent=2)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            progress.set_postfix(
                loss=running_loss / total,
                acc=correct / total,
                lr=optimizer.param_groups[0]["lr"],
            )

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"\nEpoch {epoch} validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}"
        )

        # Save checkpoint if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "val_accuracy": val_acc,
                },
                checkpoint_path,
            )
            print(f"New best model saved to {checkpoint_path} (val_acc={val_acc:.4f})")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine‑tune a ConvNeXt‑L classifier on a custom dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing train/val/test subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save outputs (model checkpoints, logs)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="convnext_large.fb_in22k_ft_in1k",
        help="Name of the ConvNeXt variant to fine‑tune (from timm)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Prepare dataloaders
    train_loader, val_loader, idx_to_class = prepare_dataloaders(
        data_dir=data_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(idx_to_class)

    # Create model with pre‑trained weights.  We override the classifier head
    # so that it matches the number of classes in the target dataset【670582884494607†L170-L243】.
    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)

    # Transfer the model to the appropriate device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Define the OneCycleLR scheduler.  This scheduler increases the learning
    # rate up to the initial value then decays it to zero over the course of
    # training【453849609406327†L1493-L1523】.  The total number of steps is
    # computed from the number of batches per epoch multiplied by the number
    # of epochs.
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=args.learning_rate, total_steps=total_steps
    )

    # Train the model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        output_dir=output_dir,
        idx_to_class=idx_to_class,
    )


if __name__ == "__main__":
    main()
