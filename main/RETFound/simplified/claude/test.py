"""
RETFound Testing and Inference Script
Author: AI Assistant
Description: Comprehensive testing and evaluation for fine-tuned RETFound models
Updated to handle model size detection and JSON serialization issues
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class RETFoundTester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_paths()
        self.setup_output_dirs()
        self.load_class_mapping()

    def setup_paths(self):
        """Setup paths and add RETFound to Python path"""
        # Try to find RETFound directory
        current_dir = Path.cwd()
        possible_paths = [
            current_dir / "RETFound_MAE",
            current_dir.parent / "RETFound_MAE",
            current_dir / ".." / "RETFound_MAE",
            Path(self.config.get("retfound_path", "RETFound_MAE")),
        ]

        self.retfound_path = None
        for path in possible_paths:
            if path.exists() and (path / "models_vit.py").exists():
                self.retfound_path = path.resolve()
                break

        if self.retfound_path is None:
            raise RuntimeError(
                "Could not find RETFound_MAE directory. Please ensure it's in the current "
                "directory or specify the path with --retfound_path"
            )

        # Add to Python path
        sys.path.insert(0, str(self.retfound_path))
        print(f"Found RETFound at: {self.retfound_path}")

    def setup_output_dirs(self):
        """Create output directories"""
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for subdir in ["predictions", "visualizations", "metrics"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)

    def load_class_mapping(self):
        """Load class names and mapping"""
        class_map_path = Path(self.config["checkpoint_dir"]) / "class_mapping.json"
        if class_map_path.exists():
            with open(class_map_path, "r") as f:
                self.class_mapping = json.load(f)
            self.idx_to_class = {int(v): k for k, v in self.class_mapping.items()}
            self.num_classes = len(self.class_mapping)
        else:
            # Infer from test directory
            test_dir = Path(self.config["test_data_path"])
            classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
            self.class_mapping = {name: idx for idx, name in enumerate(classes)}
            self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
            self.num_classes = len(classes)

        print(f"Loaded {self.num_classes} classes: {list(self.class_mapping.keys())}")

    def detect_model_size(self, checkpoint_path):
        """Detect the model size from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Get state dict
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Check embedding dimension from various possible keys
        embed_dim = None
        for key in ["cls_token", "pos_embed", "patch_embed.proj.weight"]:
            if key in state_dict:
                if key == "cls_token":
                    embed_dim = state_dict[key].shape[-1]
                elif key == "pos_embed":
                    embed_dim = state_dict[key].shape[-1]
                elif key == "patch_embed.proj.weight":
                    embed_dim = state_dict[key].shape[0]
                break

        print(f"Detected embedding dimension: {embed_dim}")

        # Determine model type based on embedding dimension
        if embed_dim == 768:
            return "base", embed_dim
        elif embed_dim == 1024:
            return "large", embed_dim
        elif embed_dim == 1280:
            return "huge", embed_dim
        else:
            print(f"Warning: Unknown embedding dimension {embed_dim}, assuming large")
            return "large", 1024

    def create_model(self):
        """Create and load the model"""
        print("Creating model...")

        # Detect model size
        model_size, embed_dim = self.detect_model_size(self.config["checkpoint_path"])
        print(f"Model size: {model_size}, embedding dimension: {embed_dim}")

        # Import models_vit
        try:
            import models_vit

            # Check if RETFound_mae is available
            if hasattr(models_vit, "RETFound_mae"):
                print("Using RETFound_mae function")
                # RETFound_mae should create the appropriate model
                model = models_vit.RETFound_mae(
                    num_classes=self.num_classes,
                    global_pool=True,
                )
            else:
                # Try to find the right function based on model size
                if model_size == "large":
                    possible_names = [
                        "vit_large_patch16",
                        "vit_large_patch16_224",
                        "vit_large",
                    ]
                elif model_size == "base":
                    possible_names = [
                        "vit_base_patch16",
                        "vit_base_patch16_224",
                        "vit_base",
                    ]
                else:
                    possible_names = ["vit_huge_patch16", "vit_huge_patch14"]

                model_fn = None
                for name in possible_names:
                    if hasattr(models_vit, name):
                        model_fn = getattr(models_vit, name)
                        print(f"Using model function: {name}")
                        break

                if model_fn is None:
                    # Last resort: manually create VisionTransformer
                    print("Creating VisionTransformer manually")
                    from models_vit import VisionTransformer

                    # Determine depth based on model size
                    if model_size == "base":
                        depth = 12
                        num_heads = 12
                        mlp_ratio = 4
                    elif model_size == "large":
                        depth = 24
                        num_heads = 16
                        mlp_ratio = 4
                    else:  # huge
                        depth = 32
                        num_heads = 16
                        mlp_ratio = 4

                    model = VisionTransformer(
                        img_size=224,
                        patch_size=16,
                        in_chans=3,
                        num_classes=self.num_classes,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        norm_layer=nn.LayerNorm,
                        global_pool=True,
                    )
                else:
                    # Create model with detected parameters
                    try:
                        model = model_fn(
                            num_classes=self.num_classes,
                            drop_path_rate=0.2,
                            global_pool=True,
                        )
                    except TypeError:
                        model = model_fn(num_classes=self.num_classes)

        except Exception as e:
            print(f"Error creating model: {e}")
            raise

        # Load checkpoint
        checkpoint_path = Path(self.config["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove prefix if necessary
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Try to import position embedding interpolation
        try:
            from util.pos_embed import interpolate_pos_embed

            interpolate_pos_embed(model, state_dict)
        except ImportError:
            print("Warning: Could not import interpolate_pos_embed")
        except Exception as e:
            print(f"Warning: Position embedding interpolation failed: {e}")

        # Load weights
        msg = model.load_state_dict(state_dict, strict=False)

        if len(msg.missing_keys) > 0:
            print(f"Missing keys ({len(msg.missing_keys)}): {msg.missing_keys[:5]}...")
        if len(msg.unexpected_keys) > 0:
            print(
                f"Unexpected keys ({len(msg.unexpected_keys)}): {msg.unexpected_keys[:5]}..."
            )

        # Verify that most parameters were loaded
        model_params = sum(p.numel() for p in model.parameters())
        loaded_params = sum(
            p.numel()
            for n, p in model.named_parameters()
            if any(n in k for k in state_dict.keys())
        )
        print(f"Model parameters: {model_params:,}")
        print(
            f"Loaded parameters: {loaded_params:,} ({loaded_params/model_params*100:.1f}%)"
        )

        if loaded_params / model_params < 0.5:
            print("WARNING: Less than 50% of parameters were loaded!")
            print("This may indicate a model architecture mismatch.")

        model = model.to(self.device)
        model.eval()

        print("Model loaded successfully!")
        return model

    def create_dataloader(self):
        """Create test dataloader with augmentation options"""
        # Define transforms
        if self.config.get("use_tta", False):
            # Test-time augmentation
            self.tta_transforms = [
                transforms.Compose(
                    [
                        transforms.Resize(
                            (self.config["input_size"], self.config["input_size"])
                        ),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                ),
                # Horizontal flip
                transforms.Compose(
                    [
                        transforms.Resize(
                            (self.config["input_size"], self.config["input_size"])
                        ),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                ),
                # Different scales
                transforms.Compose(
                    [
                        transforms.Resize(
                            (
                                int(self.config["input_size"] * 1.1),
                                int(self.config["input_size"] * 1.1),
                            )
                        ),
                        transforms.CenterCrop(self.config["input_size"]),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                ),
            ]
        else:
            # Standard transform
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (self.config["input_size"], self.config["input_size"])
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        # Create dataset
        self.test_dataset = datasets.ImageFolder(
            root=self.config["test_data_path"],
            transform=self.transform if not self.config.get("use_tta", False) else None,
        )

        # Create dataloader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

        print(f"Created dataloader with {len(self.test_dataset)} images")

        return self.test_loader

    def predict_with_tta(self, model, image_path):
        """Predict with test-time augmentation"""
        image = Image.open(image_path).convert("RGB")
        predictions = []

        with torch.no_grad():
            for transform in self.tta_transforms:
                img_tensor = transform(image).unsqueeze(0).to(self.device)
                output = model(img_tensor)
                prob = F.softmax(output, dim=1)
                predictions.append(prob.cpu().numpy())

        # Average predictions
        avg_prediction = np.mean(predictions, axis=0)
        return avg_prediction[0]

    def evaluate(self):
        """Run comprehensive evaluation"""
        try:
            model = self.create_model()
        except Exception as e:
            print(f"Failed to create model: {e}")
            print("\nTroubleshooting tips:")
            print("1. Ensure you're using the correct checkpoint file")
            print("2. Check that the model type matches the checkpoint")
            print("3. Verify the RETFound repository is properly installed")
            raise

        dataloader = self.create_dataloader()

        all_preds = []
        all_labels = []
        all_probs = []
        all_paths = []

        print("\nRunning inference...")
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)

                # Get file paths
                batch_start = batch_idx * self.config["batch_size"]
                batch_end = min(batch_start + images.size(0), len(self.test_dataset))
                batch_paths = [
                    self.test_dataset.samples[i][0]
                    for i in range(batch_start, batch_end)
                ]

                if self.config.get("use_tta", False):
                    # TTA evaluation
                    batch_probs = []
                    for img_path in batch_paths:
                        prob = self.predict_with_tta(model, img_path)
                        batch_probs.append(prob)
                    batch_probs = np.array(batch_probs)
                    preds = np.argmax(batch_probs, axis=1)
                else:
                    # Standard evaluation
                    outputs = model(images)
                    batch_probs = F.softmax(outputs, dim=1).cpu().numpy()
                    preds = np.argmax(batch_probs, axis=1)

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(batch_probs)
                all_paths.extend(batch_paths)

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        self.calculate_metrics(all_labels, all_preds, all_probs)

        # Save predictions
        self.save_predictions(all_paths, all_labels, all_preds, all_probs)

        # Generate visualizations
        self.create_visualizations(all_labels, all_preds, all_probs)

        print(f"\nEvaluation complete! Results saved to: {self.output_dir}")

    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics"""
        metrics = {}

        # Basic metrics
        report = classification_report(
            y_true,
            y_pred,
            target_names=list(self.class_mapping.keys()),
            output_dict=True,
        )

        metrics["classification_report"] = report
        metrics["accuracy"] = float(report["accuracy"])

        # Per-class metrics
        for class_name in self.class_mapping.keys():
            class_idx = self.class_mapping[class_name]
            metrics[f"{class_name}_precision"] = float(report[class_name]["precision"])
            metrics[f"{class_name}_recall"] = float(report[class_name]["recall"])
            metrics[f"{class_name}_f1"] = float(report[class_name]["f1-score"])

        # Calculate AUC if binary or compute multi-class AUC
        if self.num_classes == 2:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            metrics["auc_pr"] = float(average_precision_score(y_true, y_prob[:, 1]))
        else:
            # Multi-class AUC
            try:
                metrics["auc_roc_macro"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
                metrics["auc_roc_weighted"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
                )
            except:
                print("Could not calculate multi-class AUC")

        # Save metrics with custom encoder
        metrics_path = self.output_dir / "metrics" / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)

        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print("\nPer-class Performance:")
        for class_name in self.class_mapping.keys():
            print(
                f"{class_name:>15}: "
                f"Precision={metrics[f'{class_name}_precision']:.3f}, "
                f"Recall={metrics[f'{class_name}_recall']:.3f}, "
                f"F1={metrics[f'{class_name}_f1']:.3f}"
            )

        if "auc_roc" in metrics:
            print(f"\nAUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"AUC-PR: {metrics['auc_pr']:.4f}")
        elif "auc_roc_macro" in metrics:
            print(f"\nAUC-ROC (Macro): {metrics['auc_roc_macro']:.4f}")
            print(f"AUC-ROC (Weighted): {metrics['auc_roc_weighted']:.4f}")

        return metrics

    def save_predictions(self, paths, y_true, y_pred, y_prob):
        """Save detailed predictions"""
        results = []

        for i, path in enumerate(paths):
            result = {
                "image_path": str(path),
                "true_label": self.idx_to_class[int(y_true[i])],
                "predicted_label": self.idx_to_class[int(y_pred[i])],
                "correct": bool(y_true[i] == y_pred[i]),
                "confidence": float(y_prob[i][y_pred[i]]),
            }

            # Add probabilities for each class
            for class_name, class_idx in self.class_mapping.items():
                result[f"prob_{class_name}"] = float(y_prob[i][class_idx])

            results.append(result)

        # Save as JSON with custom encoder
        json_path = self.output_dir / "predictions" / "predictions.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        # Save as CSV for easier analysis
        df = pd.DataFrame(results)
        csv_path = self.output_dir / "predictions" / "predictions.csv"
        df.to_csv(csv_path, index=False)

        # Save misclassified cases
        misclassified = df[~df["correct"]]
        if len(misclassified) > 0:
            misclass_path = self.output_dir / "predictions" / "misclassified.csv"
            misclassified.to_csv(misclass_path, index=False)
            print(f"\nFound {len(misclassified)} misclassified cases")

    def create_visualizations(self, y_true, y_pred, y_prob):
        """Create comprehensive visualizations"""
        # Set style
        plt.style.use(
            "seaborn-v0_8-darkgrid"
            if "seaborn-v0_8-darkgrid" in plt.style.available
            else "seaborn-darkgrid"
        )

        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(self.class_mapping.keys()),
            yticklabels=list(self.class_mapping.keys()),
        )
        plt.title("Confusion Matrix", fontsize=16)
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "visualizations" / "confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 2. Normalized Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=list(self.class_mapping.keys()),
            yticklabels=list(self.class_mapping.keys()),
            vmin=0,
            vmax=1,
        )
        plt.title("Normalized Confusion Matrix", fontsize=16)
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "visualizations" / "confusion_matrix_normalized.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 3. ROC Curves (for binary classification)
        if self.num_classes == 2:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            auc = roc_auc_score(y_true, y_prob[:, 1])
            plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {auc:.3f})")
            plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title("ROC Curve", fontsize=16)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                self.output_dir / "visualizations" / "roc_curve.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Precision-Recall Curve
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            ap = average_precision_score(y_true, y_prob[:, 1])
            plt.plot(recall, precision, linewidth=2, label=f"PR curve (AP = {ap:.3f})")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("Recall", fontsize=12)
            plt.ylabel("Precision", fontsize=12)
            plt.title("Precision-Recall Curve", fontsize=16)
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                self.output_dir / "visualizations" / "pr_curve.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # 4. Class-wise performance bar plot
        report = classification_report(
            y_true,
            y_pred,
            target_names=list(self.class_mapping.keys()),
            output_dict=True,
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Precision and Recall
        classes = list(self.class_mapping.keys())
        precisions = [report[c]["precision"] for c in classes]
        recalls = [report[c]["recall"] for c in classes]

        x = np.arange(len(classes))
        width = 0.35

        bars1 = ax1.bar(
            x - width / 2,
            precisions,
            width,
            label="Precision",
            alpha=0.8,
            color="skyblue",
        )
        bars2 = ax1.bar(
            x + width / 2, recalls, width, label="Recall", alpha=0.8, color="lightcoral"
        )

        ax1.set_xlabel("Classes", fontsize=12)
        ax1.set_ylabel("Score", fontsize=12)
        ax1.set_title("Class-wise Precision and Recall", fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(
            classes,
            rotation=45 if len(classes) > 3 else 0,
            ha="right" if len(classes) > 3 else "center",
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.set_ylim(0, 1.1)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        # F1 scores
        f1_scores = [report[c]["f1-score"] for c in classes]
        bars3 = ax2.bar(x, f1_scores, alpha=0.8, color="lightgreen")
        ax2.set_xlabel("Classes", fontsize=12)
        ax2.set_ylabel("F1 Score", fontsize=12)
        ax2.set_title("Class-wise F1 Scores", fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            classes,
            rotation=45 if len(classes) > 3 else 0,
            ha="right" if len(classes) > 3 else "center",
        )
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_ylim(0, 1.1)

        # Add value labels on bars
        for bar, score in zip(bars3, f1_scores):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "visualizations" / "class_performance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Visualizations saved successfully!")

    def generate_report(self):
        """Generate a comprehensive HTML report"""
        from datetime import datetime

        # Load metrics
        metrics_path = self.output_dir / "metrics" / "evaluation_metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RETFound Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .metric {{ margin: 10px 0; }}
                .metric-value {{ font-weight: bold; color: #2196F3; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .warning {{ color: #ff9800; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>RETFound Model Evaluation Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Model Information</h2>
            <div class="metric">Model Type: <span class="metric-value">{self.config['model_type']}</span></div>
            <div class="metric">Checkpoint: <span class="metric-value">{self.config['checkpoint_path']}</span></div>
            <div class="metric">Number of Classes: <span class="metric-value">{self.num_classes}</span></div>
            <div class="metric">Test Set Size: <span class="metric-value">{len(self.test_dataset)} images</span></div>
            
            <h2>Overall Performance</h2>
            <div class="metric">Accuracy: <span class="metric-value">{metrics['accuracy']:.4f}</span></div>
        """

        if "auc_roc" in metrics:
            html_content += f"""
            <div class="metric">AUC-ROC: <span class="metric-value">{metrics['auc_roc']:.4f}</span></div>
            <div class="metric">AUC-PR: <span class="metric-value">{metrics['auc_pr']:.4f}</span></div>
            """

        html_content += """
            <h2>Confusion Matrix</h2>
            <img src="visualizations/confusion_matrix.png" alt="Confusion Matrix">
            
            <h2>Normalized Confusion Matrix</h2>
            <img src="visualizations/confusion_matrix_normalized.png" alt="Normalized Confusion Matrix">
            
            <h2>Class-wise Performance</h2>
            <img src="visualizations/class_performance.png" alt="Class Performance">
        """

        if self.num_classes == 2:
            html_content += """
            <h2>ROC Curve</h2>
            <img src="visualizations/roc_curve.png" alt="ROC Curve">
            
            <h2>Precision-Recall Curve</h2>
            <img src="visualizations/pr_curve.png" alt="Precision-Recall Curve">
            """

        html_content += """
        </body>
        </html>
        """

        report_path = self.output_dir / "evaluation_report.html"
        with open(report_path, "w") as f:
            f.write(html_content)

        print(f"HTML report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned RETFound model")

    # Required arguments
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test_data_path", type=str, required=True, help="Path to test data folder"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save results"
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="RETFound_mae",
        choices=["RETFound_mae", "RETFound_dinov2"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing training outputs (for class mapping)",
    )
    parser.add_argument(
        "--retfound_path", type=str, default=None, help="Path to RETFound_MAE directory"
    )

    # Inference arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--use_tta", action="store_true", help="Use test-time augmentation"
    )

    args = parser.parse_args()

    # Set checkpoint directory if not provided
    if args.checkpoint_dir is None:
        checkpoint_path = Path(args.checkpoint_path)
        # Try to find the parent directory with class_mapping.json
        possible_dirs = [
            checkpoint_path.parent,
            checkpoint_path.parent.parent,
            checkpoint_path.parent.parent.parent,
        ]

        for dir_path in possible_dirs:
            if (dir_path / "class_mapping.json").exists():
                args.checkpoint_dir = str(dir_path)
                break

        if args.checkpoint_dir is None:
            print(
                "Warning: Could not find class_mapping.json, will infer classes from test data"
            )
            args.checkpoint_dir = str(checkpoint_path.parent)

    config = vars(args)

    # Run testing
    try:
        tester = RETFoundTester(config)
        tester.evaluate()
        tester.generate_report()
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print("\nPlease check:")
        print("1. The checkpoint file exists and is valid")
        print("2. The test data directory has the correct structure")
        print("3. RETFound_MAE is properly installed")
        print("4. All required dependencies are installed")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
