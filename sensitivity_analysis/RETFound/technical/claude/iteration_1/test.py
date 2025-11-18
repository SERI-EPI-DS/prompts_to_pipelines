"""
Testing script for fine-tuned RETFound model
Evaluates on test set and saves results to CSV
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Add RETFound path to system
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "RETFound"))
from models_vit import VisionTransformer
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_args():
    parser = argparse.ArgumentParser("RETFound Testing Script")

    # Paths
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data directory containing test folder",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save results"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )

    # Model parameters
    parser.add_argument("--input_size", default=224, type=int, help="Image input size")
    parser.add_argument(
        "--nb_classes", type=int, required=True, help="Number of classes"
    )

    # Other parameters
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument(
        "--num_workers", default=8, type=int, help="Number of data loading workers"
    )
    parser.add_argument(
        "--pin_mem", action="store_true", default=True, help="Pin memory"
    )

    return parser.parse_args()


class TestDataset(Dataset):
    """Custom dataset that returns image path along with image and label"""

    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        path = self.dataset.imgs[idx][0]
        # Get relative path from test folder
        rel_path = os.path.relpath(path, os.path.dirname(os.path.dirname(path)))
        return image, label, rel_path


def build_transform(args):
    """Build test transformation"""
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    transform = transforms.Compose(
        [
            transforms.Resize(int(args.input_size * 256 / 224), Image.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transform


def create_model(args):
    """Create ViT-L model"""
    print(f"Creating model: ViT-Large-patch16")

    model = VisionTransformer(
        img_size=args.input_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        num_classes=args.nb_classes,
        global_pool=True,
        drop_path_rate=0.0,  # No drop path during testing
    )

    return model


@torch.no_grad()
def evaluate(model, data_loader, args):
    """Evaluate model and collect predictions"""
    model.eval()

    all_paths = []
    all_predictions = []
    all_scores = []
    all_labels = []

    for images, labels, paths in data_loader:
        images = images.to(args.device, non_blocking=True)

        # Forward pass
        outputs = model(images)

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)

        # Get predictions
        _, preds = outputs.max(1)

        # Store results
        all_paths.extend(paths)
        all_predictions.extend(preds.cpu().numpy())
        all_scores.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    return all_paths, all_predictions, all_scores, all_labels


def calculate_metrics(predictions, labels, num_classes):
    """Calculate classification metrics"""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, labels=range(num_classes)
    )

    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }


def save_results(
    all_paths, all_predictions, all_scores, all_labels, class_names, metrics, args
):
    """Save results to CSV and metrics to JSON"""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create DataFrame with results
    results_data = {
        "filename": all_paths,
        "true_label": [class_names[label] for label in all_labels],
        "predicted_label": [class_names[pred] for pred in all_predictions],
        "true_label_idx": all_labels,
        "predicted_label_idx": all_predictions,
    }

    # Add probability scores for each class
    for i, class_name in enumerate(class_names):
        results_data[f"prob_{class_name}"] = [scores[i] for scores in all_scores]

    # Create and save DataFrame
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(args.output_dir, "test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Save metrics
    metrics_dict = {
        "overall_accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class_metrics": {},
    }

    for i, class_name in enumerate(class_names):
        metrics_dict["per_class_metrics"][class_name] = {
            "precision": float(metrics["precision"][i]),
            "recall": float(metrics["recall"][i]),
            "f1": float(metrics["f1"][i]),
            "support": int(metrics["support"][i]),
        }

    # Save confusion matrix
    metrics_dict["confusion_matrix"] = metrics["confusion_matrix"].tolist()

    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Print summary
    print(f"\nTest Results Summary:")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")

    print(f"\nPer-Class Results:")
    for i, class_name in enumerate(class_names):
        print(
            f"{class_name}: "
            f"Precision={metrics['precision'][i]:.4f}, "
            f"Recall={metrics['recall'][i]:.4f}, "
            f"F1={metrics['f1'][i]:.4f}, "
            f"Support={metrics['support'][i]}"
        )


def main(args):
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Build dataset
    print("Building test dataset...")
    transform = build_transform(args)
    test_dataset = TestDataset(
        os.path.join(args.data_path, "test"), transform=transform
    )

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    # Create model
    model = create_model(args)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device)

    # Evaluate
    print("Starting evaluation...")
    all_paths, all_predictions, all_scores, all_labels = evaluate(
        model, test_loader, args
    )

    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_labels, args.nb_classes)

    # Save results
    save_results(
        all_paths,
        all_predictions,
        all_scores,
        all_labels,
        test_dataset.classes,
        metrics,
        args,
    )

    print("\nTesting completed!")


if __name__ == "__main__":
    args = get_args()
    main(args)
