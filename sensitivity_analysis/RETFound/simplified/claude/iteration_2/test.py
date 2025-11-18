"""
Test fine-tuned RETFound model and generate comprehensive evaluation
Author: AI Research Assistant
Updated to fix model loading issues
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F

from timm import create_model
from timm.models.layers import trunc_normal_
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings("ignore")


# Custom Vision Transformer for RETFound (compatible with MAE checkpoints)
class RETFoundViT(nn.Module):
    """Custom ViT implementation compatible with RETFound checkpoints"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        global_pool=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Try to use timm's create_model first, fall back to custom if needed
        try:
            from timm.models.vision_transformer import VisionTransformer

            # Create base model
            self.base_model = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=0,  # We'll add our own head
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                global_pool="avg" if global_pool else "",
            )
        except:
            # Fallback implementation
            self.patch_embed = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, embed_dim)
            )
            self.pos_drop = nn.Dropout(p=drop_rate)

            self.blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=int(mlp_ratio * embed_dim),
                        dropout=drop_rate,
                        activation="gelu",
                        batch_first=True,
                    )
                    for _ in range(depth)
                ]
            )

            self.norm = norm_layer(embed_dim)
            self.base_model = None

        # Head
        self.global_pool = global_pool
        self.fc_norm = norm_layer(embed_dim) if global_pool else nn.Identity()
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize cls_token and pos_embed
        if hasattr(self, "pos_embed"):
            trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, "cls_token"):
            trunc_normal_(self.cls_token, std=0.02)

        # Initialize head
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        if self.base_model is not None:
            # Use timm's implementation
            x = self.base_model.forward_features(x)
            if self.global_pool:
                x = x[:, 1:].mean(dim=1)  # Global average pooling (excluding cls token)
        else:
            # Custom implementation
            B = x.shape[0]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)

            if self.global_pool:
                x = x[:, 1:].mean(dim=1)  # Global average pooling
            else:
                x = x[:, 0]  # Use cls token

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc_norm(x)
        x = self.head(x)
        return x


def get_transforms(input_size=224):
    """Get test transforms"""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(checkpoint_path, device="cuda"):
    """Load fine-tuned model with improved compatibility"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract configuration
    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
        model_name = model_config.get("model_name", "RETFound_mae")
        input_size = model_config.get("input_size", 224)
        num_classes = model_config.get("num_classes", len(checkpoint["class_names"]))
    else:
        # Fallback for older checkpoints
        model_name = "RETFound_mae"
        input_size = 224
        num_classes = len(checkpoint["class_names"])

    # Model configurations
    model_configs = {
        "RETFound_mae": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4,
            "patch_size": 16,
            "timm_name": "vit_large_patch16_224",
        },
        "RETFound_dinov2": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4,
            "patch_size": 14,
            "timm_name": "vit_large_patch14_224",
        },
    }

    config = model_configs.get(model_name, model_configs["RETFound_mae"])

    # Try to create model with timm first
    try:
        model = create_model(
            config["timm_name"],
            pretrained=False,
            num_classes=num_classes,
            img_size=input_size,
        )
        print(f"Created model using timm: {config['timm_name']}")
    except Exception as e:
        print(f"Could not create model with timm: {e}")
        print("Using custom RETFound ViT implementation...")

        # Fallback to custom implementation
        model = RETFoundViT(
            img_size=input_size,
            patch_size=config["patch_size"],
            num_classes=num_classes,
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            global_pool=True,
        )

    # Load weights
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Warning: Error loading weights directly: {e}")
        print("Attempting to load with flexibility...")

        # Try loading with more flexibility
        model_dict = model.state_dict()
        pretrained_dict = checkpoint["model_state_dict"]

        # Filter out incompatible keys
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }

        # Update the model dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")

    model = model.to(device)
    model.eval()

    return model, checkpoint["class_names"], input_size


def test_model(model, test_loader, device):
    """Test model and collect predictions"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    all_features = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Get model outputs
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def calculate_metrics(y_true, y_pred, y_probs, class_names):
    """Calculate comprehensive metrics"""
    num_classes = len(class_names)
    print(f"bugcheck-1: {num_classes}")

    # Basic metrics
    accuracy = np.mean(y_true == y_pred)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    per_class_metrics = []
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    print("bugcheck:", range(num_classes))
    print(f"bugcheck-2: {y_true_bin.shape}")

    for i, class_name in enumerate(class_names):

        # Bugfix for binary case, where label_binarize returns array of shape (n_samples, 1) instead of (n_samples, 2)
        if num_classes == 2:
            if i == 0:
                class_true = np.array([x == 0 for x in y_true_bin])
            else:
                class_true = y_true_bin
        else:
            class_true = y_true_bin[:, i]

        class_probs = y_probs[:, i]

        # Handle cases with no positive samples
        if np.sum(class_true) > 0:
            # ROC curve
            fpr, tpr, _ = roc_curve(class_true, class_probs)
            roc_auc = auc(fpr, tpr)

            # PR curve
            precision, recall, _ = precision_recall_curve(class_true, class_probs)
            ap_score = average_precision_score(class_true, class_probs)
        else:
            fpr, tpr = [0, 1], [0, 1]
            roc_auc = 0.5
            precision, recall = [1, 0], [0, 1]
            ap_score = 0.0

        per_class_metrics.append(
            {
                "class_name": class_name,
                "roc_auc": roc_auc,
                "ap_score": ap_score,
                "fpr": fpr,
                "tpr": tpr,
                "precision": precision,
                "recall": recall,
            }
        )

    # Overall AUC
    try:
        overall_auc = roc_auc_score(y_true_bin, y_probs, average="macro")
    except:
        overall_auc = 0.5

    return {
        "accuracy": accuracy,
        "overall_auc": overall_auc,
        "classification_report": report,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics,
    }


def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))

    # Normalize confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proportion"},
    )

    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Also save raw confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Confusion Matrix (Raw Counts)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_raw.png"), dpi=300)
    plt.close()


def plot_roc_curves(per_class_metrics, output_dir):
    """Plot ROC curves for all classes"""
    plt.figure(figsize=(10, 8))

    for metrics in per_class_metrics:
        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            label=f"{metrics['class_name']} (AUC = {metrics['roc_auc']:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.close()


def plot_pr_curves(per_class_metrics, output_dir):
    """Plot Precision-Recall curves for all classes"""
    plt.figure(figsize=(10, 8))

    for metrics in per_class_metrics:
        plt.plot(
            metrics["recall"],
            metrics["precision"],
            label=f"{metrics['class_name']} (AP = {metrics['ap_score']:.3f})",
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=300)
    plt.close()


def plot_class_distribution(y_true, class_names, output_dir):
    """Plot class distribution in test set"""
    plt.figure(figsize=(10, 6))

    unique, counts = np.unique(y_true, return_counts=True)

    # Handle missing classes
    all_counts = np.zeros(len(class_names), dtype=int)
    for i, count in zip(unique, counts):
        all_counts[i] = count

    bars = plt.bar(range(len(class_names)), all_counts)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title("Test Set Class Distribution")

    # Add count labels on bars
    for bar, count in zip(bars, all_counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
    plt.close()


def generate_error_analysis(
    y_true, y_pred, y_probs, class_names, test_dataset, output_dir
):
    """Analyze misclassified samples"""
    errors = []

    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors.append(
                {
                    "index": i,
                    "true_class": class_names[y_true[i]],
                    "predicted_class": class_names[y_pred[i]],
                    "true_prob": y_probs[i, y_true[i]],
                    "pred_prob": y_probs[i, y_pred[i]],
                    "confidence": np.max(y_probs[i]),
                }
            )

    # Save error analysis
    error_df = pd.DataFrame(errors)
    if len(errors) > 0:
        error_df = error_df.sort_values("confidence", ascending=False)
        error_df.to_csv(os.path.join(output_dir, "error_analysis.csv"), index=False)

        # Plot confidence distribution for errors
        plt.figure(figsize=(10, 6))
        plt.hist(error_df["confidence"], bins=20, edgecolor="black")
        plt.xlabel("Prediction Confidence")
        plt.ylabel("Number of Errors")
        plt.title("Confidence Distribution of Misclassified Samples")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "error_confidence_dist.png"), dpi=300)
        plt.close()
    else:
        # Create empty error analysis file
        pd.DataFrame(
            columns=[
                "index",
                "true_class",
                "predicted_class",
                "true_prob",
                "pred_prob",
                "confidence",
            ]
        ).to_csv(os.path.join(output_dir, "error_analysis.csv"), index=False)

    return error_df if len(errors) > 0 else pd.DataFrame()


def save_predictions(y_true, y_pred, y_probs, class_names, image_paths, output_dir):
    """Save detailed predictions for each sample"""
    predictions = []

    for i in range(len(y_true)):
        pred_dict = {
            "image_path": image_paths[i],
            "true_class": class_names[y_true[i]],
            "predicted_class": class_names[y_pred[i]],
            "correct": y_true[i] == y_pred[i],
        }

        # Add probability for each class
        for j, class_name in enumerate(class_names):
            pred_dict[f"prob_{class_name}"] = float(y_probs[i, j])

        predictions.append(pred_dict)

    # Save as CSV
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    return pred_df


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned RETFound model")

    # Required arguments
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to test dataset folder"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Output directory for results",
    )

    # Optional arguments
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    try:
        model, class_names, input_size = load_model(args.checkpoint, device)
        print(f"Model loaded successfully with input size: {input_size}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Create test dataset and loader
    test_dataset = datasets.ImageFolder(
        args.test_data, transform=get_transforms(input_size)
    )

    # Verify class names match
    if test_dataset.classes != class_names:
        print(
            f"Warning: Test dataset classes {test_dataset.classes} don't match "
            f"training classes {class_names}"
        )
        print("Proceeding with training class order...")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Get image paths
    image_paths = [item[0] for item in test_dataset.samples]

    # Test model
    print("\nTesting model...")
    y_pred, y_true, y_probs = test_model(model, test_loader, device)

    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_probs, class_names)

    # Print results
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Overall AUC: {metrics['overall_auc']:.4f}")
    print("\nPer-Class Results:")

    for class_name in class_names:
        if class_name in metrics["classification_report"]:
            class_metrics = metrics["classification_report"][class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  F1-Score: {class_metrics['f1-score']:.4f}")
            print(f"  Support: {class_metrics['support']}")

    # Save detailed results
    results = {
        "test_accuracy": float(metrics["accuracy"]),
        "test_auc": float(metrics["overall_auc"]),
        "classification_report": metrics["classification_report"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "class_names": class_names,
        "test_samples": len(y_true),
        "model_checkpoint": args.checkpoint,
        "test_data": args.test_data,
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, args.output_dir)
    plot_roc_curves(metrics["per_class_metrics"], args.output_dir)
    plot_pr_curves(metrics["per_class_metrics"], args.output_dir)
    plot_class_distribution(y_true, class_names, args.output_dir)

    # Error analysis
    print("Performing error analysis...")
    error_df = generate_error_analysis(
        y_true, y_pred, y_probs, class_names, test_dataset, args.output_dir
    )

    # Save predictions
    print("Saving predictions...")
    pred_df = save_predictions(
        y_true, y_pred, y_probs, class_names, image_paths, args.output_dir
    )

    # Generate final report
    report_path = os.path.join(args.output_dir, "test_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("RETFOUND MODEL TEST REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Checkpoint: {args.checkpoint}\n")
        f.write(f"Test Dataset: {args.test_data}\n")
        f.write(f"Total Test Samples: {len(y_true)}\n\n")

        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro AUC: {metrics['overall_auc']:.4f}\n\n")

        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        for i, class_name in enumerate(class_names):
            cm = metrics["confusion_matrix"]
            if class_name in metrics["classification_report"]:
                class_report = metrics["classification_report"][class_name]
                class_auc = metrics["per_class_metrics"][i]["roc_auc"]

                f.write(f"\n{class_name}:\n")
                f.write(f"  Samples: {class_report['support']}\n")
                f.write(f"  Precision: {class_report['precision']:.4f}\n")
                f.write(f"  Recall: {class_report['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_report['f1-score']:.4f}\n")
                f.write(f"  AUC: {class_auc:.4f}\n")
                if i < len(cm):
                    f.write(f"  True Positives: {cm[i, i]}\n")
                    f.write(f"  False Positives: {cm[:, i].sum() - cm[i, i]}\n")
                    f.write(f"  False Negatives: {cm[i, :].sum() - cm[i, i]}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("ERROR ANALYSIS SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Errors: {len(error_df)}\n")
        f.write(f"Error Rate: {len(error_df)/len(y_true):.4f}\n")

        if len(error_df) > 0:
            f.write(f"Average Error Confidence: {error_df['confidence'].mean():.4f}\n")
            f.write(f"Min Error Confidence: {error_df['confidence'].min():.4f}\n")
            f.write(f"Max Error Confidence: {error_df['confidence'].max():.4f}\n")

    print(f"\nAll results saved to: {args.output_dir}")
    print(f"Test report saved to: {report_path}")


if __name__ == "__main__":
    main()
