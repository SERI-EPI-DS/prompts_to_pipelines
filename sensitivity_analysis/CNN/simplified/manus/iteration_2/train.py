#!/usr/bin/env python3
"""
ConvNext-L Fine-tuning Script for Ophthalmology Image Classification
Author: AI Research Assistant
Description: State-of-the-art training script for fine-tuning ConvNext-L on medical imaging datasets
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from timm.scheduler import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import AverageMeter, accuracy
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConvNextTrainer:
    """
    ConvNext-L trainer class with state-of-the-art techniques for medical image classification
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Create output directories
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize model, data loaders, optimizer, scheduler
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        # Training tracking
        self.best_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def setup_data_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Setup data transforms with medical imaging best practices
        """
        # Training transforms with augmentation
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Validation transforms (no augmentation)
        val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return train_transform, val_transform

    def setup_data_loaders(self):
        """
        Setup data loaders with class balancing for medical datasets
        """
        train_transform, val_transform = self.setup_data_transforms()

        # Load datasets
        train_dataset = ImageFolder(
            root=os.path.join(self.config["data_path"], "train"),
            transform=train_transform,
        )

        val_dataset = ImageFolder(
            root=os.path.join(self.config["data_path"], "val"), transform=val_transform
        )

        # Calculate class weights for imbalanced datasets
        train_targets = [train_dataset.targets[i] for i in range(len(train_dataset))]
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(train_targets), y=train_targets
        )

        # Create weighted sampler
        sample_weights = [class_weights[t] for t in train_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

        self.num_classes = len(train_dataset.classes)
        self.class_names = train_dataset.classes

        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Class names: {self.class_names}")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        # Save class weights and names
        class_info = {
            "class_names": self.class_names,
            "class_weights": class_weights.tolist(),
            "num_classes": self.num_classes,
        }

        with open(self.output_dir / "class_info.json", "w") as f:
            json.dump(class_info, f, indent=2)

    def setup_model(self):
        """
        Setup ConvNext-L model with pre-trained weights
        """
        # Load ConvNext-L with ImageNet-22K -> ImageNet-1K pre-trained weights
        self.model = timm.create_model(
            "convnext_large_in22k",
            pretrained=True,
            num_classes=self.num_classes,
            drop_path_rate=self.config["drop_path_rate"],
        )

        # Move model to device
        self.model = self.model.to(self.device)

        # Use DataParallel for multi-GPU training
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        logger.info(
            f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def setup_optimizer_and_scheduler(self):
        """
        Setup optimizer and learning rate scheduler
        """
        # Adam optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            betas=(0.9, 0.999),
        )

        # Cosine annealing scheduler with warmup
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.config["epochs"],
            lr_min=self.config["learning_rate"] * 0.01,
            warmup_t=self.config["warmup_epochs"],
            warmup_lr_init=self.config["learning_rate"] * 0.1,
            warmup_prefix=True,
        )

        # Label smoothing cross entropy for better generalization
        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=self.config["label_smoothing"]
        )

        logger.info("Optimizer and scheduler setup complete")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch
        """
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Calculate accuracy
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{losses.avg:.4f}",
                    "Acc": f"{top1.avg:.2f}%",
                    "LR": f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                }
            )

        return losses.avg, top1.avg

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model
        """
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                # Calculate accuracy
                acc1 = accuracy(output, target, topk=(1,))[0]
                losses.update(loss.item(), data.size(0))
                top1.update(acc1.item(), data.size(0))

                # Store predictions for detailed analysis
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Generate classification report
        report = classification_report(
            all_targets, all_preds, target_names=self.class_names, output_dict=True
        )

        return losses.avg, top1.avg, report

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_acc": self.best_acc,
            "config": self.config,
            "class_names": self.class_names,
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(state, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(state, best_path)
            logger.info(f"New best model saved with accuracy: {self.best_acc:.2f}%")

    def plot_training_curves(self):
        """
        Plot and save training curves
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(self.train_losses, label="Train Loss")
        ax1.plot(self.val_losses, label="Validation Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(self.train_accs, label="Train Accuracy")
        ax2.plot(self.val_accs, label="Validation Accuracy")
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "training_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def train(self):
        """
        Main training loop
        """
        logger.info("Starting training...")
        start_time = time.time()

        for epoch in range(self.config["epochs"]):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)

            # Validation
            val_loss, val_acc, val_report = self.validate()

            # Update scheduler
            self.scheduler.step(epoch)

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # Check if best model
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc

            # Save checkpoint
            if (epoch + 1) % self.config["save_freq"] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Best Val Acc: {self.best_acc:.2f}%")
            logger.info("-" * 50)

            # Save detailed validation report for best epoch
            if is_best:
                with open(self.output_dir / "best_validation_report.json", "w") as f:
                    json.dump(val_report, f, indent=2)

        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        logger.info(f"Best validation accuracy: {self.best_acc:.2f}%")

        # Plot training curves
        self.plot_training_curves()

        # Save final training log
        training_log = {
            "config": self.config,
            "best_accuracy": self.best_acc,
            "total_training_time": total_time,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
        }

        with open(self.output_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="ConvNext-L Fine-tuning for Ophthalmology"
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset directory (should contain train/val/test folders)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for results and checkpoints",
    )

    # Model arguments
    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.2,
        help="Drop path rate for regularization",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Number of warmup epochs"
    )

    # System arguments
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--save_freq", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Main training function
    """
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create config dictionary
    config = {
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "drop_path_rate": args.drop_path_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "warmup_epochs": args.warmup_epochs,
        "num_workers": args.num_workers,
        "save_freq": args.save_freq,
        "seed": args.seed,
    }

    # Validate data path
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")

    required_dirs = ["train", "val"]
    for dir_name in required_dirs:
        dir_path = os.path.join(args.data_path, dir_name)
        if not os.path.exists(dir_path):
            raise ValueError(f"Required directory does not exist: {dir_path}")

    # Initialize trainer
    trainer = ConvNextTrainer(config)

    # Setup training components
    trainer.setup_data_loaders()
    trainer.setup_model()
    trainer.setup_optimizer_and_scheduler()

    # Start training
    trainer.train()

    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()
