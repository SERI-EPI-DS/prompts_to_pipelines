#!/usr/bin/env python3
"""
RETFound Fine-tuning Training Script

This script fine-tunes the RETFound_mae ViT-L foundation model for ophthalmology image classification.
It implements comprehensive training functionality including dataset preparation, model loading,
training loop with validation, and model checkpointing.

Requirements:
- Python 3.11.0
- PyTorch 2.3.1
- TorchVision 0.18.1
- PyTorch-CUDA 12.1
- Single RTX3090 with 24GB VRAM

Author: Manus AI
Date: 2025-06-25
"""

import argparse
import datetime
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Import RETFound modules
sys.path.append("RETFound")
try:
    import models_vit as models
    from util.datasets import build_dataset
    from util.lr_decay import param_groups_lrd
    from util.lr_sched import adjust_learning_rate
    from util.misc import NativeScalerWithGradNormCount as NativeScaler
    from util.pos_embed import interpolate_pos_embed
    from engine_finetune import train_one_epoch, evaluate
except ImportError as e:
    print(f"Error importing RETFound modules: {e}")
    print(
        "Please ensure the RETFound repository is available in the current directory."
    )
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class RETFoundDataset(Dataset):
    """
    Custom dataset class for RETFound fine-tuning.
    Handles image loading, preprocessing, and augmentation.
    """

    def __init__(
        self, data_path: str, split: str, transform=None, target_transform=None
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the data directory
            split: Dataset split ('train', 'val', or 'test')
            transform: Image transformations
            target_transform: Target transformations
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Use ImageFolder for automatic class detection
        self.dataset = ImageFolder(
            root=self.data_path / split,
            transform=transform,
            target_transform=target_transform,
        )

        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.samples = self.dataset.samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_args_parser():
    """
    Create argument parser for training script.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser("RETFound Fine-tuning Training", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="RETFound_mae",
        type=str,
        metavar="MODEL",
        help="Name of model to train (default: RETFound_mae)",
    )
    parser.add_argument(
        "--input_size", default=224, type=int, help="Images input size (default: 224)"
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="Learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="Base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="Layer-wise lr decay from ELECTRA/BEiT",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="Lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="Epochs to warmup LR"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using AutoAugment)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0, help="Mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="CutMix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="CutMix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Finetuning params
    parser.add_argument(
        "--finetune", default="RETFound_mae_natureCFP", help="Finetune from checkpoint"
    )
    parser.add_argument("--global_pool", action="store_true", default=True)
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    # Dataset parameters
    parser.add_argument("--data_path", default="./data", type=str, help="Dataset path")
    parser.add_argument(
        "--nb_classes",
        default=None,
        type=int,
        help="Number of the classification types (will be auto-detected if not specified)",
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./results/logs", help="Path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="Start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor)",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Training parameters
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="Number of total epochs to run (max 50 as specified)",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)",
    )
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="Number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="URL used to set up distributed training"
    )

    return parser


def build_transforms(args, is_train=True):
    """
    Build image transformations for training and validation.

    Args:
        args: Command line arguments
        is_train: Whether to build transforms for training

    Returns:
        torchvision.transforms.Compose: Image transformations
    """
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]  # ImageNet std

    if is_train:
        # Training transforms with augmentation
        transform_list = [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

        # Add random erasing if specified
        if args.reprob > 0:
            transform_list.append(
                transforms.RandomErasing(
                    p=args.reprob,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value=0,
                    inplace=False,
                )
            )
    else:
        # Validation/test transforms without augmentation
        transform_list = [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(transform_list)


def create_datasets(args):
    """
    Create training, validation, and test datasets.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Build transforms
    train_transform = build_transforms(args, is_train=True)
    val_transform = build_transforms(args, is_train=False)

    # Create datasets
    train_dataset = RETFoundDataset(args.data_path, "train", transform=train_transform)
    val_dataset = RETFoundDataset(args.data_path, "val", transform=val_transform)
    test_dataset = RETFoundDataset(args.data_path, "test", transform=val_transform)

    # Auto-detect number of classes if not specified
    if args.nb_classes is None:
        args.nb_classes = len(train_dataset.classes)
        print(f"Auto-detected {args.nb_classes} classes: {train_dataset.classes}")

    return train_dataset, val_dataset, test_dataset


def create_model(args):
    """
    Create and initialize the RETFound model.

    Args:
        args: Command line arguments

    Returns:
        torch.nn.Module: Initialized model
    """
    print(f"Creating model: {args.model}")

    # Create model
    model = models.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load pre-trained weights
    if args.finetune and not args.eval:
        print(f"Loading pre-trained weights from: {args.finetune}")

        try:
            from huggingface_hub import hf_hub_download

            # Download checkpoint from HuggingFace
            checkpoint_path = hf_hub_download(
                repo_id=f"YukunZhou/{args.finetune}",
                filename=f"{args.finetune}.pth",
            )

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print(f"Loaded pre-trained checkpoint from: {checkpoint_path}")

            # Extract model state dict
            if args.model == "RETFound_mae":
                checkpoint_model = checkpoint["teacher"]
            else:
                checkpoint_model = checkpoint["model"]

            # Remove mismatched keys
            state_dict = model.state_dict()
            for k in ["head.weight", "head.bias"]:
                if (
                    k in checkpoint_model
                    and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # Interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # Load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(f"Load pre-trained checkpoint: {msg}")

        except Exception as e:
            print(f"Error loading pre-trained weights: {e}")
            print("Continuing with random initialization...")

    return model


def setup_optimizer_and_scheduler(model, args, data_loader_train):
    """
    Setup optimizer and learning rate scheduler.

    Args:
        model: The model to optimize
        args: Command line arguments
        data_loader_train: Training data loader

    Returns:
        Tuple of (optimizer, loss_scaler)
    """
    # Calculate effective batch size
    eff_batch_size = args.batch_size * args.accum_iter

    # Calculate learning rate
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print(f"Base lr: {args.blr:.2e}")
    print(f"Actual lr: {args.lr:.2e}")
    print(f"Accumulate grad iterations: {args.accum_iter}")
    print(f"Effective batch size: {eff_batch_size}")

    # Build optimizer with layer-wise lr decay
    param_groups = param_groups_lrd(
        model,
        args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    # Create loss scaler
    loss_scaler = NativeScaler()

    return optimizer, loss_scaler


def train_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    max_norm=None,
    mixup_fn=None,
    log_writer=None,
    args=None,
):
    """
    Train for one epoch.

    Args:
        model: The model to train
        criterion: Loss function
        data_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch
        loss_scaler: Loss scaler for mixed precision
        max_norm: Gradient clipping norm
        mixup_fn: Mixup function
        log_writer: Tensorboard writer
        args: Command line arguments

    Returns:
        Dict with training statistics
    """
    model.train(True)
    metric_logger = {}

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        # Adjust learning rate
        if data_iter_step % args.accum_iter == 0:
            adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply mixup if specified
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= args.accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % args.accum_iter == 0,
        )

        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        # Logging
        if data_iter_step % 50 == 0:
            print(
                f"Epoch: [{epoch}][{data_iter_step}/{len(data_loader)}] "
                f"Loss: {loss_value:.4f} "
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )

    return {"loss": loss_value}


def validate(model, criterion, data_loader, device, args):
    """
    Validate the model.

    Args:
        model: The model to validate
        criterion: Loss function
        data_loader: Validation data loader
        device: Device to use
        args: Command line arguments

    Returns:
        Dict with validation statistics
    """
    model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)

            # Get predictions
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            total_loss += loss.item() * samples.size(0)
            num_samples += samples.size(0)

    # Calculate metrics
    avg_loss = total_loss / num_samples
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(
        all_targets, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_preds,
        "targets": all_targets,
    }


def save_checkpoint(model, optimizer, loss_scaler, epoch, args, is_best=False):
    """
    Save model checkpoint.

    Args:
        model: The model to save
        optimizer: Optimizer state
        loss_scaler: Loss scaler state
        epoch: Current epoch
        args: Command line arguments
        is_best: Whether this is the best checkpoint
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_scaler": loss_scaler.state_dict(),
        "epoch": epoch,
        "args": args,
    }

    # Save regular checkpoint
    checkpoint_path = output_dir / f"checkpoint-{epoch:04d}.pth"
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint
    if is_best:
        best_path = output_dir / "checkpoint-best.pth"
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint to {best_path}")


def main(args):
    """
    Main training function.

    Args:
        args: Command line arguments
    """
    print(f"Job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"Arguments: {args}")

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(args)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Create model
    model = create_model(args)
    model.to(device)

    # Count parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params (M): {n_parameters / 1.e6:.2f}")

    # Setup optimizer and scheduler
    optimizer, loss_scaler = setup_optimizer_and_scheduler(model, args, train_loader)

    # Setup loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    # Setup tensorboard logging
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # Train for one epoch
        train_stats = train_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            None,
            log_writer,
            args,
        )

        # Validate
        val_stats = validate(model, criterion, val_loader, device, args)

        print(f"Training Loss: {train_stats['loss']:.4f}")
        print(f"Validation Loss: {val_stats['loss']:.4f}")
        print(f"Validation Accuracy: {val_stats['accuracy']:.4f}")
        print(f"Validation F1: {val_stats['f1']:.4f}")

        # Log to tensorboard
        if log_writer:
            log_writer.add_scalar("train/loss", train_stats["loss"], epoch)
            log_writer.add_scalar("val/loss", val_stats["loss"], epoch)
            log_writer.add_scalar("val/accuracy", val_stats["accuracy"], epoch)
            log_writer.add_scalar("val/f1", val_stats["f1"], epoch)
            log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Save checkpoint
        is_best = val_stats["accuracy"] > best_acc
        if is_best:
            best_acc = val_stats["accuracy"]
            best_epoch = epoch

        save_checkpoint(model, optimizer, loss_scaler, epoch, args, is_best)

        # Save training log
        log_stats = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
            "val_precision": val_stats["precision"],
            "val_recall": val_stats["recall"],
            "val_f1": val_stats["f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "n_parameters": n_parameters,
        }

        if args.output_dir:
            with open(Path(args.output_dir) / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print(f"\nTraining completed in {total_time_str}")
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch + 1}")

    if log_writer:
        log_writer.close()


if __name__ == "__main__":
    import math

    args = get_args_parser()
    args = args.parse_args()

    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
