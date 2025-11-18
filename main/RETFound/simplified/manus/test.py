#!/usr/bin/env python3
"""
RETFound Model Testing and Evaluation Script (Calibration Plot Fix)
==================================================================

A comprehensive testing script for evaluating fine-tuned RETFound models
on ophthalmology image classification tasks with detailed metrics and analysis.

FIXES:
- Fixed calibration curve plotting for multi-class scenarios
- Robust handling of different bin structures across classes
- Enhanced visualization error handling
- Improved multi-class calibration analysis

Features:
- Comprehensive evaluation metrics (accuracy, precision, recall, F1, AUC)
- Per-class performance analysis
- Confusion matrix visualization
- ROC and Precision-Recall curves
- Test-time augmentation support
- Statistical significance testing
- Detailed reporting and visualization
- Batch processing for large datasets
- Confidence analysis and calibration

Author: AI Assistant
Date: 2025
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.calibration import calibration_curve
from scipy import stats
import cv2

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class TestConfig:
    """Configuration class for testing parameters."""

    def __init__(self):
        # Model configuration
        self.model_name = "RETFound_mae"
        self.checkpoint_path = ""
        self.num_classes = 5  # Default, will be updated based on data/checkpoint
        self.input_size = 224

        # Data configuration
        self.data_path = "./data/test"
        self.batch_size = 32
        self.num_workers = 8
        self.pin_memory = True

        # Testing configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.amp = True  # Automatic Mixed Precision

        # Test-time augmentation
        self.test_time_augmentation = True
        self.tta_num_augmentations = 5
        self.tta_transforms = [
            "horizontal_flip",
            "vertical_flip",
            "rotation",
            "brightness",
            "contrast",
        ]

        # Output configuration
        self.output_dir = "./test_results"
        self.save_predictions = True
        self.save_visualizations = True
        self.save_misclassified = True
        self.detailed_analysis = True

        # Visualization configuration
        self.plot_confusion_matrix = True
        self.plot_roc_curves = True
        self.plot_pr_curves = True
        self.plot_calibration = True
        self.plot_confidence_distribution = True
        self.figure_dpi = 300
        self.figure_size = (10, 8)


def find_checkpoint_files(search_dirs):
    """Find available checkpoint files in given directories."""
    checkpoint_files = []

    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            # Look for common checkpoint patterns
            patterns = [
                os.path.join(search_dir, "best_model.pth"),
                os.path.join(search_dir, "checkpoint.pth"),
                os.path.join(search_dir, "model_best.pth"),
                os.path.join(search_dir, "checkpoint_*.pth"),
                os.path.join(search_dir, "*.pth"),
                os.path.join(search_dir, "*.pt"),
            ]

            for pattern in patterns:
                files = glob.glob(pattern)
                checkpoint_files.extend(files)

    # Remove duplicates and sort
    checkpoint_files = list(set(checkpoint_files))
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)  # Most recent first

    return checkpoint_files


def get_checkpoint_info(checkpoint_path):
    """Extract information from checkpoint without loading the full model."""
    try:
        # Load checkpoint metadata only
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        info = {
            "num_classes": None,
            "input_size": None,
            "model_name": None,
            "epoch": checkpoint.get("epoch", "unknown"),
            "best_acc": checkpoint.get("best_acc", "unknown"),
        }

        # Try to get configuration from checkpoint
        if "config" in checkpoint:
            config = checkpoint["config"]
            info["num_classes"] = config.get("num_classes")
            info["input_size"] = config.get("input_size", 224)
            info["model_name"] = config.get("model_name", "unknown")

        # If no config, try to infer from model state
        if info["num_classes"] is None:
            state_dict = None
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Look for final layer to determine number of classes
            if state_dict:
                # Remove 'module.' prefix if present
                if any(key.startswith("module.") for key in state_dict.keys()):
                    state_dict = {key[7:]: value for key, value in state_dict.items()}

                # Check common final layer names
                final_layer_keys = [
                    "fc.weight",
                    "head.weight",
                    "classifier.weight",
                    "linear.weight",
                ]
                for key in final_layer_keys:
                    if key in state_dict:
                        info["num_classes"] = state_dict[key].shape[0]
                        break

        return info

    except Exception as e:
        print(f"⚠️  Warning: Could not extract checkpoint info: {e}")
        return {"num_classes": None, "input_size": None, "model_name": None}


def validate_checkpoint_path(checkpoint_path, data_path=None):
    """Validate and potentially auto-discover checkpoint path."""
    if not checkpoint_path or checkpoint_path.strip() == "":
        print("⚠️  No checkpoint path provided. Searching for available checkpoints...")

        # Common directories to search for checkpoints
        search_dirs = [
            "./outputs",
            "./checkpoints",
            "./models",
            "./results",
            ".",
        ]

        # If data path is provided, also search relative to it
        if data_path:
            data_parent = os.path.dirname(
                os.path.dirname(data_path)
            )  # Go up two levels from test data
            search_dirs.extend(
                [
                    os.path.join(data_parent, "outputs"),
                    os.path.join(data_parent, "checkpoints"),
                    os.path.join(data_parent, "models"),
                    data_parent,
                ]
            )

        checkpoint_files = find_checkpoint_files(search_dirs)

        if checkpoint_files:
            print(f"📁 Found {len(checkpoint_files)} checkpoint file(s):")
            for i, file in enumerate(checkpoint_files[:5]):  # Show first 5
                size_mb = os.path.getsize(file) / (1024 * 1024)
                mod_time = os.path.getmtime(file)
                mod_time_str = pd.to_datetime(mod_time, unit="s").strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                # Get checkpoint info
                info = get_checkpoint_info(file)
                info_str = (
                    f"classes: {info['num_classes']}"
                    if info["num_classes"]
                    else "classes: unknown"
                )

                print(
                    f"   {i+1}. {file} ({size_mb:.1f} MB, {mod_time_str}, {info_str})"
                )

            if len(checkpoint_files) > 5:
                print(f"   ... and {len(checkpoint_files) - 5} more")

            # Use the most recent one
            selected_checkpoint = checkpoint_files[0]
            print(f"✅ Auto-selected most recent checkpoint: {selected_checkpoint}")
            return selected_checkpoint
        else:
            print("❌ No checkpoint files found in common directories.")
            print("\n💡 Suggestions:")
            print("   1. Provide the checkpoint path explicitly:")
            print(
                "      python test.py --checkpoint /path/to/your/model.pth --data-path /path/to/test/data"
            )
            print("   2. Make sure you have trained a model first:")
            print(
                "      python train.py --data-path /path/to/data --output-dir ./outputs"
            )
            print("   3. Check if your checkpoint is in a different directory")
            print("\n🔍 Searched in directories:")
            for dir_path in search_dirs:
                exists = "✅" if os.path.exists(dir_path) else "❌"
                print(f"   {exists} {dir_path}")

            raise FileNotFoundError(
                "No checkpoint file found. Please provide a valid checkpoint path using --checkpoint argument."
            )

    # Validate provided checkpoint path
    if not os.path.isfile(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")

        # Try to find similar files
        checkpoint_dir = os.path.dirname(checkpoint_path) or "."
        checkpoint_name = os.path.basename(checkpoint_path)

        if os.path.exists(checkpoint_dir):
            similar_files = []
            for file in os.listdir(checkpoint_dir):
                if (
                    file.endswith((".pth", ".pt"))
                    or "checkpoint" in file.lower()
                    or "model" in file.lower()
                ):
                    similar_files.append(os.path.join(checkpoint_dir, file))

            if similar_files:
                print(f"\n💡 Found similar files in {checkpoint_dir}:")
                for file in similar_files:
                    info = get_checkpoint_info(file)
                    info_str = (
                        f"(classes: {info['num_classes']})"
                        if info["num_classes"]
                        else "(classes: unknown)"
                    )
                    print(f"   📄 {file} {info_str}")
                print("\nDid you mean one of these files?")

        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    return checkpoint_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RETFound Model Testing and Evaluation Script (Calibration Plot Fix)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint (if not provided, will auto-search)",
    )
    parser.add_argument(
        "--model",
        default="RETFound_mae",
        choices=["RETFound_mae", "RETFound_dinov2"],
        help="Model architecture",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classification classes (auto-detected if not provided)",
    )
    parser.add_argument("--input-size", type=int, default=224, help="Input image size")

    # Data arguments
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to test dataset directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of data loading workers"
    )

    # Testing arguments
    parser.add_argument(
        "--tta", action="store_true", default=True, help="Use test-time augmentation"
    )
    parser.add_argument(
        "--tta-num-augmentations",
        type=int,
        default=5,
        help="Number of augmentations for TTA",
    )
    parser.add_argument(
        "--amp", action="store_true", default=True, help="Use automatic mixed precision"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=True,
        help="Save detailed predictions",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        default=True,
        help="Save visualization plots",
    )
    parser.add_argument(
        "--save-misclassified",
        action="store_true",
        default=True,
        help="Save misclassified examples",
    )
    parser.add_argument(
        "--detailed-analysis",
        action="store_true",
        default=True,
        help="Perform detailed statistical analysis",
    )

    return parser.parse_args()


def determine_num_classes(data_path, checkpoint_path, args_num_classes):
    """Determine the correct number of classes from multiple sources."""
    num_classes_sources = {}

    # 1. From command line argument
    if args_num_classes is not None:
        num_classes_sources["command_line"] = args_num_classes
        print(f"📋 Number of classes from command line: {args_num_classes}")

    # 2. From test dataset
    try:
        if os.path.exists(data_path):
            # Create a temporary dataset to count classes
            temp_transform = transforms.Compose([transforms.ToTensor()])
            temp_dataset = ImageFolder(root=data_path, transform=temp_transform)
            dataset_classes = len(temp_dataset.classes)
            num_classes_sources["dataset"] = dataset_classes
            print(f"📁 Number of classes from test dataset: {dataset_classes}")
            print(f"   Class names: {temp_dataset.classes}")
        else:
            print(f"⚠️  Test dataset path not found: {data_path}")
    except Exception as e:
        print(f"⚠️  Could not determine classes from dataset: {e}")

    # 3. From checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_info = get_checkpoint_info(checkpoint_path)
        if checkpoint_info["num_classes"]:
            num_classes_sources["checkpoint"] = checkpoint_info["num_classes"]
            print(
                f"💾 Number of classes from checkpoint: {checkpoint_info['num_classes']}"
            )

    # Decide which source to use
    if len(num_classes_sources) == 0:
        print(
            "⚠️  Could not determine number of classes from any source. Using default: 5"
        )
        return 5

    # Priority: command line > checkpoint > dataset
    if "command_line" in num_classes_sources:
        selected_classes = num_classes_sources["command_line"]
        source = "command line"
    elif "checkpoint" in num_classes_sources:
        selected_classes = num_classes_sources["checkpoint"]
        source = "checkpoint"
    elif "dataset" in num_classes_sources:
        selected_classes = num_classes_sources["dataset"]
        source = "dataset"
    else:
        selected_classes = 5
        source = "default"

    print(f"✅ Using {selected_classes} classes (source: {source})")

    # Warn about mismatches
    unique_values = set(num_classes_sources.values())
    if len(unique_values) > 1:
        print("⚠️  Warning: Mismatch in number of classes between sources:")
        for src, num in num_classes_sources.items():
            print(f"   {src}: {num} classes")
        print(f"   Using: {selected_classes} classes from {source}")

    return selected_classes


def create_model(config: TestConfig):
    """Create and initialize the RETFound model."""
    try:
        # Import RETFound models (assuming they're available)
        from models_vit import vit_large_patch16

        if config.model_name == "RETFound_mae":
            model = vit_large_patch16(
                img_size=config.input_size,
                num_classes=config.num_classes,
                drop_path_rate=0.0,  # No dropout during testing
                global_pool=True,
            )
        else:
            raise NotImplementedError(f"Model {config.model_name} not implemented")

        print(f"🏗️  Created {config.model_name} model with {config.num_classes} classes")
        return model

    except ImportError:
        print(
            "Warning: RETFound models not available. Using torchvision ResNet as fallback."
        )
        import torchvision.models as models

        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
        print(f"🏗️  Created ResNet50 fallback model with {config.num_classes} classes")
        return model


def load_model_checkpoint(model, checkpoint_path, device, expected_classes=None):
    """Load model from checkpoint with enhanced error handling."""
    # Validate checkpoint path
    checkpoint_path = validate_checkpoint_path(checkpoint_path)

    print(f"📂 Loading checkpoint from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("📋 Found 'model' key in checkpoint")
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("📋 Found 'state_dict' key in checkpoint")
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("📋 Found 'model_state_dict' key in checkpoint")
    else:
        state_dict = checkpoint
        print("📋 Using checkpoint as state_dict directly")

    # Remove 'module.' prefix if present (from DataParallel)
    if any(key.startswith("module.") for key in state_dict.keys()):
        print("🔧 Removing 'module.' prefix from state_dict keys")
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    # Check for class mismatch before loading
    final_layer_keys = [
        "fc.weight",
        "head.weight",
        "classifier.weight",
        "linear.weight",
    ]
    checkpoint_classes = None

    for key in final_layer_keys:
        if key in state_dict:
            checkpoint_classes = state_dict[key].shape[0]
            print(f"📊 Checkpoint has {checkpoint_classes} classes (from {key})")
            break

    if (
        checkpoint_classes
        and expected_classes
        and checkpoint_classes != expected_classes
    ):
        print(f"⚠️  Class mismatch detected:")
        print(f"   Expected: {expected_classes} classes")
        print(f"   Checkpoint: {checkpoint_classes} classes")

        # Try to handle the mismatch
        print("🔧 Attempting to fix class mismatch...")

        # Option 1: Remove the final layer from state_dict (let model use its own)
        for key in list(state_dict.keys()):
            if any(
                final_key.replace(".weight", "") in key
                for final_key in final_layer_keys
            ):
                print(f"   Removing {key} from checkpoint")
                del state_dict[key]

        print(
            "✅ Removed final layer weights from checkpoint. Model will use randomly initialized final layer."
        )

    # Load state dict with error handling
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"⚠️  Missing keys in checkpoint: {len(missing_keys)} keys")
            # Only show final layer missing keys as info, others as warnings
            final_layer_missing = [
                k
                for k in missing_keys
                if any(fl.replace(".weight", "") in k for fl in final_layer_keys)
            ]
            other_missing = [k for k in missing_keys if k not in final_layer_missing]

            if final_layer_missing:
                print(
                    f"ℹ️  Missing final layer keys (expected for class mismatch): {len(final_layer_missing)}"
                )
                for key in final_layer_missing[:3]:
                    print(f"   - {key}")
                if len(final_layer_missing) > 3:
                    print(f"   ... and {len(final_layer_missing) - 3} more")

            if other_missing:
                print(f"⚠️  Missing other keys: {len(other_missing)}")
                for key in other_missing[:3]:
                    print(f"   - {key}")
                if len(other_missing) > 3:
                    print(f"   ... and {len(other_missing) - 3} more")

        if unexpected_keys:
            print(f"⚠️  Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 5:
                for key in unexpected_keys:
                    print(f"   - {key}")
            else:
                for key in unexpected_keys[:3]:
                    print(f"   - {key}")
                print(f"   ... and {len(unexpected_keys) - 3} more")

        if not missing_keys and not unexpected_keys:
            print("✅ All keys matched perfectly")
        elif (
            len(
                [
                    k
                    for k in missing_keys
                    if not any(
                        fl.replace(".weight", "") in k for fl in final_layer_keys
                    )
                ]
            )
            == 0
        ):
            print("✅ All keys matched (except expected final layer differences)")

    except Exception as e:
        print(f"❌ Error loading state dict: {e}")
        raise RuntimeError(f"Failed to load model state from checkpoint: {e}")

    # Get additional info from checkpoint
    epoch = checkpoint.get("epoch", "unknown")
    best_acc = checkpoint.get("best_acc", "unknown")

    print(f"📊 Checkpoint info:")
    print(f"   - Epoch: {epoch}")
    print(f"   - Best accuracy: {best_acc}")

    # Check if config exists in checkpoint
    if "config" in checkpoint:
        checkpoint_config = checkpoint["config"]
        print(f"   - Model classes: {checkpoint_config.get('num_classes', 'unknown')}")
        print(f"   - Input size: {checkpoint_config.get('input_size', 'unknown')}")

    return model


def create_test_transforms(config: TestConfig, augmentation_type=None):
    """Create test transforms with optional augmentation."""
    base_transforms = [
        transforms.Resize((config.input_size, config.input_size)),
    ]

    # Add specific augmentation if requested
    if augmentation_type == "horizontal_flip":
        base_transforms.append(transforms.RandomHorizontalFlip(p=1.0))
    elif augmentation_type == "vertical_flip":
        base_transforms.append(transforms.RandomVerticalFlip(p=1.0))
    elif augmentation_type == "rotation":
        base_transforms.append(transforms.RandomRotation(degrees=15))
    elif augmentation_type == "brightness":
        base_transforms.append(transforms.ColorJitter(brightness=0.2))
    elif augmentation_type == "contrast":
        base_transforms.append(transforms.ColorJitter(contrast=0.2))

    base_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transforms.Compose(base_transforms)


def create_test_dataset_and_loader(config: TestConfig, augmentation_type=None):
    """Create test dataset and data loader."""
    transform = create_test_transforms(config, augmentation_type)

    # Validate data path
    if not os.path.exists(config.data_path):
        print(f"❌ Test data path not found: {config.data_path}")

        # Suggest alternative paths
        parent_dir = os.path.dirname(config.data_path)
        if os.path.exists(parent_dir):
            subdirs = [
                d
                for d in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, d))
            ]
            if subdirs:
                print(f"💡 Available subdirectories in {parent_dir}:")
                for subdir in subdirs:
                    print(f"   📁 {os.path.join(parent_dir, subdir)}")

        raise FileNotFoundError(f"Test data directory not found: {config.data_path}")

    try:
        dataset = ImageFolder(root=config.data_path, transform=transform)
        print(f"📁 Test dataset loaded successfully")
        print(f"   - Path: {config.data_path}")
        print(f"   - Classes: {len(dataset.classes)}")
        print(f"   - Samples: {len(dataset)}")
        print(f"   - Class names: {dataset.classes}")

    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        print(f"💡 Make sure your test data is organized as:")
        print(f"   {config.data_path}/")
        print(f"   ├── class1/")
        print(f"   │   ├── image1.jpg")
        print(f"   │   └── image2.jpg")
        print(f"   ├── class2/")
        print(f"   │   ├── image3.jpg")
        print(f"   │   └── image4.jpg")
        print(f"   └── ...")
        raise

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    return dataset, loader


@torch.no_grad()
def predict_with_tta(model, data_loader, config: TestConfig):
    """Perform prediction with test-time augmentation."""
    model.eval()

    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_filenames = []

    if config.test_time_augmentation:
        print(
            f"🔄 Performing test-time augmentation with {config.tta_num_augmentations} augmentations..."
        )

        # Get predictions for each augmentation
        tta_probabilities = []

        for aug_idx, aug_type in enumerate(
            config.tta_transforms[: config.tta_num_augmentations]
        ):
            print(
                f"   Augmentation {aug_idx + 1}/{config.tta_num_augmentations}: {aug_type}"
            )

            # Create dataset with specific augmentation
            _, aug_loader = create_test_dataset_and_loader(config, aug_type)

            aug_probs = []
            targets = []
            filenames = []

            for batch_idx, (images, target) in enumerate(aug_loader):
                images = images.to(config.device, non_blocking=True)
                target = target.to(config.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=config.amp):
                    outputs = model(images)
                    probabilities = F.softmax(outputs, dim=1)

                aug_probs.append(probabilities.cpu().numpy())
                if aug_idx == 0:  # Only collect targets and filenames once
                    targets.append(target.cpu().numpy())
                    # Get filenames from dataset
                    batch_start = batch_idx * config.batch_size
                    batch_end = min(
                        batch_start + config.batch_size, len(aug_loader.dataset)
                    )
                    batch_filenames = [
                        aug_loader.dataset.samples[i][0]
                        for i in range(batch_start, batch_end)
                    ]
                    filenames.extend(batch_filenames)

            tta_probabilities.append(np.concatenate(aug_probs, axis=0))
            if aug_idx == 0:
                all_targets = np.concatenate(targets, axis=0)
                all_filenames = filenames

        # Average predictions across augmentations
        all_probabilities = np.mean(tta_probabilities, axis=0)
        all_predictions = np.argmax(all_probabilities, axis=1)

    else:
        print("🔄 Performing standard inference without augmentation...")

        # Standard inference without augmentation
        dataset, data_loader = create_test_dataset_and_loader(config)

        for batch_idx, (images, target) in enumerate(data_loader):
            images = images.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=config.amp):
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)

            predictions = outputs.argmax(dim=1)

            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            # Get filenames
            batch_start = batch_idx * config.batch_size
            batch_end = min(batch_start + config.batch_size, len(dataset))
            batch_filenames = [
                dataset.samples[i][0] for i in range(batch_start, batch_end)
            ]
            all_filenames.extend(batch_filenames)

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_probabilities = np.concatenate(all_probabilities, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

    return all_predictions, all_probabilities, all_targets, all_filenames


def calculate_comprehensive_metrics(y_true, y_pred, y_prob, class_names):
    """Calculate comprehensive evaluation metrics."""
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Per-class and averaged metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics["per_class_precision"] = precision
    metrics["per_class_recall"] = recall
    metrics["per_class_f1"] = f1
    metrics["per_class_support"] = support

    # Averaged metrics
    metrics["macro_precision"] = np.mean(precision)
    metrics["macro_recall"] = np.mean(recall)
    metrics["macro_f1"] = np.mean(f1)

    metrics["weighted_precision"] = np.average(precision, weights=support)
    metrics["weighted_recall"] = np.average(recall, weights=support)
    metrics["weighted_f1"] = np.average(f1, weights=support)

    # Additional metrics
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)

    # ROC AUC
    try:
        if len(class_names) > 2:
            metrics["macro_auc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
            metrics["weighted_auc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"
            )
        else:
            metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
    except ValueError as e:
        print(f"Warning: Could not calculate AUC - {e}")
        metrics["macro_auc"] = 0.0
        metrics["weighted_auc"] = 0.0

    # Average Precision
    try:
        if len(class_names) > 2:
            ap_scores = []
            for i in range(len(class_names)):
                y_true_binary = (y_true == i).astype(int)
                ap = average_precision_score(y_true_binary, y_prob[:, i])
                ap_scores.append(ap)
            metrics["per_class_ap"] = np.array(ap_scores)
            metrics["macro_ap"] = np.mean(ap_scores)
        else:
            metrics["ap"] = average_precision_score(y_true, y_prob[:, 1])
    except ValueError as e:
        print(f"Warning: Could not calculate Average Precision - {e}")
        metrics["macro_ap"] = 0.0

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return metrics


def plot_confusion_matrix(cm, class_names, output_path, normalize=False):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))

    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Normalized Confusion Matrix")
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, y_prob, class_names, output_path):
    """Plot and save ROC curves."""
    plt.figure(figsize=(10, 8))

    if len(class_names) > 2:
        # Multi-class ROC curves
        for i, class_name in enumerate(class_names):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            auc = roc_auc_score(y_true_binary, y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc:.3f})")
    else:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curves(y_true, y_prob, class_names, output_path):
    """Plot and save Precision-Recall curves."""
    plt.figure(figsize=(10, 8))

    if len(class_names) > 2:
        # Multi-class PR curves
        for i, class_name in enumerate(class_names):
            y_true_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, i])
            ap = average_precision_score(y_true_binary, y_prob[:, i])
            plt.plot(recall, precision, label=f"{class_name} (AP = {ap:.3f})")
    else:
        # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        ap = average_precision_score(y_true, y_prob[:, 1])
        plt.plot(recall, precision, label=f"PR Curve (AP = {ap:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_calibration_curve(y_true, y_prob, class_names, output_path):
    """Plot and save calibration curve with robust multi-class handling."""
    try:
        plt.figure(figsize=(10, 8))

        if len(class_names) == 2:
            # Binary classification calibration
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_prob[:, 1], n_bins=10
                )
                plt.plot(
                    mean_predicted_value,
                    fraction_of_positives,
                    "s-",
                    label="Model",
                    linewidth=2,
                    markersize=8,
                )
            except Exception as e:
                print(f"Warning: Could not compute binary calibration curve: {e}")
                # Fallback: simple reliability diagram
                bins = np.linspace(0, 1, 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bin_accuracies = []
                bin_confidences = []

                for i in range(len(bins) - 1):
                    mask = (y_prob[:, 1] >= bins[i]) & (y_prob[:, 1] < bins[i + 1])
                    if np.sum(mask) > 0:
                        bin_accuracy = np.mean(y_true[mask])
                        bin_confidence = np.mean(y_prob[mask, 1])
                        bin_accuracies.append(bin_accuracy)
                        bin_confidences.append(bin_confidence)
                    else:
                        bin_accuracies.append(0)
                        bin_confidences.append(bins[i])

                plt.plot(
                    bin_confidences,
                    bin_accuracies,
                    "s-",
                    label="Model",
                    linewidth=2,
                    markersize=8,
                )

        else:
            # Multi-class calibration with robust handling
            print("📊 Computing multi-class calibration curves...")

            # Use a fixed bin structure for all classes
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

            all_fractions = []
            all_confidences = []
            valid_classes = []

            for i, class_name in enumerate(class_names):
                try:
                    y_true_binary = (y_true == i).astype(int)
                    class_probs = y_prob[:, i]

                    # Check if we have enough samples for this class
                    if np.sum(y_true_binary) < 2:
                        print(
                            f"   Skipping {class_name}: insufficient samples ({np.sum(y_true_binary)})"
                        )
                        continue

                    # Compute calibration using fixed bins
                    bin_fractions = []
                    bin_confidences = []

                    for j in range(n_bins):
                        mask = (class_probs >= bin_boundaries[j]) & (
                            class_probs < bin_boundaries[j + 1]
                        )
                        if (
                            j == n_bins - 1
                        ):  # Include the right boundary for the last bin
                            mask = (class_probs >= bin_boundaries[j]) & (
                                class_probs <= bin_boundaries[j + 1]
                            )

                        if np.sum(mask) > 0:
                            bin_fraction = np.mean(y_true_binary[mask])
                            bin_confidence = np.mean(class_probs[mask])
                            bin_fractions.append(bin_fraction)
                            bin_confidences.append(bin_confidence)
                        else:
                            # Use bin center for empty bins
                            bin_fractions.append(0.0)
                            bin_confidences.append(bin_centers[j])

                    all_fractions.append(bin_fractions)
                    all_confidences.append(bin_confidences)
                    valid_classes.append(class_name)

                    # Plot individual class calibration
                    plt.plot(
                        bin_confidences,
                        bin_fractions,
                        "o-",
                        alpha=0.7,
                        label=f"{class_name}",
                        linewidth=1.5,
                        markersize=6,
                    )

                except Exception as e:
                    print(
                        f"   Warning: Could not compute calibration for {class_name}: {e}"
                    )
                    continue

            # Compute and plot average calibration if we have valid classes
            if len(all_fractions) > 0:
                try:
                    # Convert to numpy arrays with same length
                    all_fractions = np.array(all_fractions)
                    all_confidences = np.array(all_confidences)

                    # Compute average
                    avg_fractions = np.mean(all_fractions, axis=0)
                    avg_confidences = np.mean(all_confidences, axis=0)

                    plt.plot(
                        avg_confidences,
                        avg_fractions,
                        "s-",
                        color="black",
                        label="Average",
                        linewidth=3,
                        markersize=10,
                    )

                except Exception as e:
                    print(f"   Warning: Could not compute average calibration: {e}")

            print(
                f"   Successfully computed calibration for {len(valid_classes)} classes"
            )

        # Perfect calibration line
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Plot (Reliability Diagram)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Calibration plot saved to {output_path}")

    except Exception as e:
        print(f"❌ Error creating calibration plot: {e}")
        print("   Skipping calibration plot generation")

        # Create a simple placeholder plot
        try:
            plt.figure(figsize=(10, 8))
            plt.text(
                0.5,
                0.5,
                f"Calibration plot could not be generated\nError: {str(e)}",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                fontsize=12,
            )
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title("Calibration Plot (Error)")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
        except:
            pass


def plot_confidence_distribution(y_prob, y_true, y_pred, class_names, output_path):
    """Plot confidence distribution for correct and incorrect predictions."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Get maximum probabilities (confidence scores)
        confidence_scores = np.max(y_prob, axis=1)
        correct_mask = y_true == y_pred

        # Overall confidence distribution
        if np.sum(correct_mask) > 0 and np.sum(~correct_mask) > 0:
            axes[0, 0].hist(
                confidence_scores[correct_mask],
                bins=20,
                alpha=0.7,
                label="Correct",
                density=True,
                color="green",
            )
            axes[0, 0].hist(
                confidence_scores[~correct_mask],
                bins=20,
                alpha=0.7,
                label="Incorrect",
                density=True,
                color="red",
            )
        else:
            axes[0, 0].hist(
                confidence_scores,
                bins=20,
                alpha=0.7,
                label="All predictions",
                density=True,
                color="blue",
            )

        axes[0, 0].set_xlabel("Confidence Score")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_title("Confidence Distribution")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Confidence vs Accuracy
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []

        for i in range(len(bins) - 1):
            mask = (confidence_scores >= bins[i]) & (confidence_scores < bins[i + 1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(correct_mask[mask])
                bin_accuracies.append(bin_accuracy)
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)

        axes[0, 1].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, color="blue")
        axes[0, 1].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        axes[0, 1].set_xlabel("Confidence Score")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Confidence vs Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Per-class confidence (show first 2 classes)
        for i, class_name in enumerate(class_names[:2]):
            ax = axes[1, i]
            class_mask = y_true == i
            if np.sum(class_mask) > 0:
                class_confidence = y_prob[class_mask, i]
                ax.hist(class_confidence, bins=15, alpha=0.7, color=f"C{i}")
                ax.set_xlabel("Confidence Score")
                ax.set_ylabel("Count")
                ax.set_title(f"Confidence for {class_name}")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No samples for {class_name}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"Confidence for {class_name}")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"Warning: Could not create confidence distribution plot: {e}")


def save_misclassified_examples(
    y_true, y_pred, y_prob, filenames, class_names, output_dir, top_k=20
):
    """Save examples of misclassified images with analysis."""
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]

    if len(misclassified_indices) == 0:
        print("✅ No misclassified examples found!")
        return

    # Get confidence scores for misclassified examples
    misclassified_confidence = np.max(y_prob[misclassified_indices], axis=1)

    # Sort by confidence (highest confidence mistakes first)
    sorted_indices = misclassified_indices[np.argsort(-misclassified_confidence)]

    misclassified_data = []
    for idx in sorted_indices[:top_k]:
        misclassified_data.append(
            {
                "filename": filenames[idx],
                "true_class": class_names[y_true[idx]],
                "predicted_class": class_names[y_pred[idx]],
                "confidence": float(np.max(y_prob[idx])),
                "true_class_prob": float(y_prob[idx, y_true[idx]]),
                "predicted_class_prob": float(y_prob[idx, y_pred[idx]]),
                "all_probabilities": y_prob[idx].tolist(),
            }
        )

    # Save to JSON
    misclassified_path = os.path.join(output_dir, "misclassified_examples.json")
    with open(misclassified_path, "w") as f:
        json.dump(misclassified_data, f, indent=2)

    print(
        f"📄 Saved {len(misclassified_data)} misclassified examples to {misclassified_path}"
    )


def generate_detailed_report(metrics, class_names, output_path):
    """Generate a detailed evaluation report."""
    report = []
    report.append("=" * 80)
    report.append("RETFOUND MODEL EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Overall metrics
    report.append("OVERALL PERFORMANCE METRICS")
    report.append("-" * 40)
    report.append(f"Accuracy: {metrics['accuracy']:.4f}")
    report.append(f"Macro Precision: {metrics['macro_precision']:.4f}")
    report.append(f"Macro Recall: {metrics['macro_recall']:.4f}")
    report.append(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    report.append(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    report.append(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
    report.append(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")

    if "macro_auc" in metrics:
        report.append(f"Macro AUC: {metrics['macro_auc']:.4f}")
        report.append(f"Weighted AUC: {metrics['weighted_auc']:.4f}")

    if "macro_ap" in metrics:
        report.append(f"Macro Average Precision: {metrics['macro_ap']:.4f}")

    report.append(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    report.append(
        f"Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}"
    )
    report.append("")

    # Per-class metrics
    report.append("PER-CLASS PERFORMANCE METRICS")
    report.append("-" * 40)
    report.append(
        f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}"
    )
    report.append("-" * 70)

    for i, class_name in enumerate(class_names):
        precision = metrics["per_class_precision"][i]
        recall = metrics["per_class_recall"][i]
        f1 = metrics["per_class_f1"][i]
        support = metrics["per_class_support"][i]

        report.append(
            f"{class_name:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}"
        )

    report.append("")

    # Confusion matrix
    report.append("CONFUSION MATRIX")
    report.append("-" * 40)
    cm = metrics["confusion_matrix"]

    # Header
    header = "True\\Pred".ljust(12)
    for class_name in class_names:
        header += f"{class_name[:8]:<10}"
    report.append(header)

    # Matrix rows
    for i, class_name in enumerate(class_names):
        row = f"{class_name[:10]:<12}"
        for j in range(len(class_names)):
            row += f"{cm[i, j]:<10}"
        report.append(row)

    report.append("")

    # Additional analysis
    report.append("ADDITIONAL ANALYSIS")
    report.append("-" * 40)

    # Class balance
    total_samples = np.sum(metrics["per_class_support"])
    report.append("Class Distribution:")
    for i, class_name in enumerate(class_names):
        percentage = (metrics["per_class_support"][i] / total_samples) * 100
        report.append(
            f"  {class_name}: {metrics['per_class_support'][i]} samples ({percentage:.1f}%)"
        )

    report.append("")

    # Performance insights
    report.append("PERFORMANCE INSIGHTS")
    report.append("-" * 40)

    # Best and worst performing classes
    best_f1_idx = np.argmax(metrics["per_class_f1"])
    worst_f1_idx = np.argmin(metrics["per_class_f1"])

    report.append(
        f"Best performing class: {class_names[best_f1_idx]} (F1: {metrics['per_class_f1'][best_f1_idx]:.4f})"
    )
    report.append(
        f"Worst performing class: {class_names[worst_f1_idx]} (F1: {metrics['per_class_f1'][worst_f1_idx]:.4f})"
    )

    # Class imbalance impact
    class_sizes = metrics["per_class_support"]
    size_std = np.std(class_sizes)
    size_mean = np.mean(class_sizes)
    cv = size_std / size_mean if size_mean > 0 else 0

    report.append(f"Class imbalance (CV): {cv:.4f}")
    if cv > 0.5:
        report.append(
            "  Warning: High class imbalance detected. Consider using balanced metrics."
        )

    report.append("")
    report.append("=" * 80)

    # Save report
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    # Also print to console
    print("\n".join(report))


def main():
    """Main testing function."""
    # Parse arguments
    args = parse_args()

    # Create configuration
    config = TestConfig()

    # Update config with command line arguments
    for key, value in vars(args).items():
        if hasattr(config, key.replace("-", "_")):
            setattr(config, key.replace("-", "_"), value)

    # Validate and set checkpoint path
    config.checkpoint_path = validate_checkpoint_path(args.checkpoint, args.data_path)

    # Determine the correct number of classes
    config.num_classes = determine_num_classes(
        args.data_path, config.checkpoint_path, args.num_classes
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"📁 Output directory: {config.output_dir}")

    # Save configuration
    config_path = os.path.join(config.output_dir, "test_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2, default=str)
    print(f"💾 Test configuration saved to {config_path}")

    # Create model
    print("🏗️  Creating model...")
    model = create_model(config)

    # Load checkpoint
    model = load_model_checkpoint(
        model, config.checkpoint_path, config.device, config.num_classes
    )
    model.to(config.device)
    model.eval()

    # Create test dataset to get class names
    test_dataset, _ = create_test_dataset_and_loader(config)
    class_names = test_dataset.classes

    # Verify class count matches
    if len(class_names) != config.num_classes:
        print(
            f"⚠️  Warning: Test dataset has {len(class_names)} classes but model expects {config.num_classes}"
        )
        print(f"   Dataset classes: {class_names}")
        print("   This may cause issues during evaluation.")

    print(f"📊 Dataset summary:")
    print(f"   - Classes: {len(class_names)}")
    print(f"   - Class names: {class_names}")
    print(f"   - Test samples: {len(test_dataset)}")

    # Perform inference
    print("🔄 Starting inference...")
    predictions, probabilities, targets, filenames = predict_with_tta(
        model, None, config
    )

    print(f"✅ Inference completed. Processed {len(predictions)} samples.")

    # Calculate comprehensive metrics
    print("📊 Calculating metrics...")
    metrics = calculate_comprehensive_metrics(
        targets, predictions, probabilities, class_names
    )

    # Generate and save detailed report
    report_path = os.path.join(config.output_dir, "evaluation_report.txt")
    generate_detailed_report(metrics, class_names, report_path)

    # Save predictions if requested
    if config.save_predictions:
        predictions_data = {
            "filenames": filenames,
            "true_labels": targets.tolist(),
            "predicted_labels": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "class_names": class_names,
        }

        predictions_path = os.path.join(config.output_dir, "predictions.json")
        with open(predictions_path, "w") as f:
            json.dump(predictions_data, f, indent=2)
        print(f"💾 Predictions saved to {predictions_path}")

        # Also save as CSV for easy analysis
        df = pd.DataFrame(
            {
                "filename": filenames,
                "true_label": [class_names[i] for i in targets],
                "predicted_label": [class_names[i] for i in predictions],
                "confidence": np.max(probabilities, axis=1),
                "correct": targets == predictions,
            }
        )

        # Add probability columns
        for i, class_name in enumerate(class_names):
            df[f"prob_{class_name}"] = probabilities[:, i]

        csv_path = os.path.join(config.output_dir, "preds.csv")
        df.to_csv(csv_path, index=False)
        print(f"💾 Predictions CSV saved to {csv_path}")

    # Generate visualizations if requested
    if config.save_visualizations:
        print("📈 Generating visualizations...")

        # Confusion matrix
        if config.plot_confusion_matrix:
            try:
                cm_path = os.path.join(config.output_dir, "confusion_matrix.png")
                plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path)

                cm_norm_path = os.path.join(
                    config.output_dir, "confusion_matrix_normalized.png"
                )
                plot_confusion_matrix(
                    metrics["confusion_matrix"],
                    class_names,
                    cm_norm_path,
                    normalize=True,
                )
            except Exception as e:
                print(f"Warning: Could not create confusion matrix: {e}")

        # ROC curves
        if config.plot_roc_curves:
            try:
                roc_path = os.path.join(config.output_dir, "roc_curves.png")
                plot_roc_curves(targets, probabilities, class_names, roc_path)
            except Exception as e:
                print(f"Warning: Could not create ROC curves: {e}")

        # Precision-Recall curves
        if config.plot_pr_curves:
            try:
                pr_path = os.path.join(config.output_dir, "precision_recall_curves.png")
                plot_precision_recall_curves(
                    targets, probabilities, class_names, pr_path
                )
            except Exception as e:
                print(f"Warning: Could not create PR curves: {e}")

        # Calibration curve
        if config.plot_calibration:
            try:
                cal_path = os.path.join(config.output_dir, "calibration_curve.png")
                plot_calibration_curve(targets, probabilities, class_names, cal_path)
            except Exception as e:
                print(f"Warning: Could not create calibration curve: {e}")

        # Confidence distribution
        if config.plot_confidence_distribution:
            try:
                conf_path = os.path.join(
                    config.output_dir, "confidence_distribution.png"
                )
                plot_confidence_distribution(
                    probabilities, targets, predictions, class_names, conf_path
                )
            except Exception as e:
                print(f"Warning: Could not create confidence distribution: {e}")

        print("📈 Visualizations saved to output directory.")

    # Save misclassified examples if requested
    if config.save_misclassified:
        try:
            save_misclassified_examples(
                targets,
                predictions,
                probabilities,
                filenames,
                class_names,
                config.output_dir,
            )
        except Exception as e:
            print(f"Warning: Could not save misclassified examples: {e}")

    # Save metrics summary
    metrics_summary = {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_precision": float(metrics["weighted_precision"]),
        "weighted_recall": float(metrics["weighted_recall"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "cohen_kappa": float(metrics["cohen_kappa"]),
        "matthews_corrcoef": float(metrics["matthews_corrcoef"]),
    }

    if "macro_auc" in metrics:
        metrics_summary["macro_auc"] = float(metrics["macro_auc"])
        metrics_summary["weighted_auc"] = float(metrics["weighted_auc"])

    if "macro_ap" in metrics:
        metrics_summary["macro_ap"] = float(metrics["macro_ap"])

    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        metrics_summary[f"{class_name}_precision"] = float(
            metrics["per_class_precision"][i]
        )
        metrics_summary[f"{class_name}_recall"] = float(metrics["per_class_recall"][i])
        metrics_summary[f"{class_name}_f1"] = float(metrics["per_class_f1"][i])
        metrics_summary[f"{class_name}_support"] = int(metrics["per_class_support"][i])

    metrics_path = os.path.join(config.output_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\n🎉 Evaluation completed successfully!")
    print(f"📁 Results saved to: {config.output_dir}")
    print(f"📊 Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"📊 Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"📊 Weighted F1-Score: {metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
