import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import argparse
import os
import json
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class SwinClassifier(nn.Module):
    """Wrapper class for Swin Transformer with proper classification head"""

    def __init__(
        self, model_name, num_classes, pretrained=True, dropout=0.1, img_size=224
    ):
        super(SwinClassifier, self).__init__()

        # Create base model without classifier
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # No classifier
            drop_rate=dropout,
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone(dummy_input)
            if isinstance(features, (list, tuple)):
                feature_dim = features[0].shape[1]
            else:
                # Handle 4D output by global averaging
                if features.dim() == 4:
                    feature_dim = features.shape[1]
                else:
                    feature_dim = features.shape[-1]

        # Global average pooling for 4D features
        self.global_pool = (
            nn.AdaptiveAvgPool2d((1, 1)) if features.dim() == 4 else nn.Identity()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        print(
            f"Model: {model_name}, Feature dim: {feature_dim}, Num classes: {num_classes}"
        )

    def forward(self, x):
        features = self.backbone(x)

        # Handle different feature shapes
        if features.dim() == 4:  # [batch, channels, height, width]
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)
        elif features.dim() == 3:  # [batch, seq_len, features]
            features = features.mean(dim=1)  # Average over sequence

        return self.classifier(features)


def create_test_transform(img_size=224):
    """Create test transforms"""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(model_path, num_classes, device, img_size=224):
    """Load trained model with proper architecture"""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Get model info from checkpoint
    if "model_name" in checkpoint:
        model_name = checkpoint["model_name"]
    else:
        # Try to get from model_info.json
        try:
            with open(
                os.path.join(os.path.dirname(model_path), "model_info.json"), "r"
            ) as f:
                model_info = json.load(f)
                model_name = model_info["model_name"]
        except:
            model_name = "swin_base_patch4_window7_224"  # Default fallback

    # Load class names
    if "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
    else:
        # Try to load from separate file
        try:
            with open(
                os.path.join(os.path.dirname(model_path), "class_names.json"), "r"
            ) as f:
                class_names = json.load(f)
        except:
            class_names = [str(i) for i in range(num_classes)]

    print(f"Loading model: {model_name}")

    # Load training args for dropout rate
    dropout = 0.1
    try:
        with open(
            os.path.join(os.path.dirname(model_path), "training_args.json"), "r"
        ) as f:
            training_args = json.load(f)
            dropout = training_args.get("dropout", 0.1)
    except:
        pass

    # Create model with the same architecture as during training
    model = SwinClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,  # We're loading trained weights
        dropout=dropout,
        img_size=img_size,
    )

    # Load trained weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, class_names


def evaluate_model(model, test_loader, device, class_names, output_dir):
    """Comprehensive model evaluation"""
    all_predictions = []
    all_labels = []
    all_probabilities = []

    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calculate metrics
    accuracy = np.mean(all_predictions == all_labels)

    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels, all_predictions, target_names=class_names, digits=4
        )
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {class_accuracy[i]:.4f}")

    # ROC Curve and AUC for multi-class (one-vs-rest)
    if len(class_names) > 2:
        # Multi-class ROC
        fpr = {}
        tpr = {}
        roc_auc = {}

        plt.figure(figsize=(10, 8))
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(
                (all_labels == i).astype(int), all_probabilities[:, i]
            )
            roc_auc[i] = roc_auc_score(
                (all_labels == i).astype(int), all_probabilities[:, i]
            )
            plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.4f})")

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-class ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "roc_curves.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"\nMacro AUC: {np.mean(list(roc_auc.values())):.4f}")
    else:
        # Binary classification
        auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "roc_curve.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"AUC: {auc_score:.4f}")

    # Save detailed results
    results = {
        "accuracy": float(accuracy),
        "predictions": all_predictions.tolist(),
        "labels": all_labels.tolist(),
        "probabilities": all_probabilities.tolist(),
        "class_names": class_names,
        "classification_report": classification_report(
            all_labels, all_predictions, target_names=class_names, output_dict=True
        ),
    }

    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions as CSV
    df_results = pd.DataFrame(
        {
            "true_label": all_labels,
            "predicted_label": all_predictions,
            "true_class": [class_names[i] for i in all_labels],
            "predicted_class": [class_names[i] for i in all_predictions],
        }
    )

    # Add probabilities for each class
    for i, class_name in enumerate(class_names):
        df_results[f"prob_{class_name}"] = all_probabilities[:, i]

    df_results.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    return accuracy, results


def main():
    parser = argparse.ArgumentParser(
        description="Test Swin Transformer Ophthalmology Classifier"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to test data directory"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create test dataset and loader
    test_transform = create_test_transform(args.img_size)
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")

    # Load model - use the same number of classes as the test dataset
    model, class_names = load_model(
        args.model_path, len(test_dataset.classes), device, args.img_size
    )
    print(f"Model loaded from: {args.model_path}")

    # Verify class names match
    if set(class_names) != set(test_dataset.classes):
        print(
            f"Warning: Class names in model ({class_names}) don't match test dataset classes ({test_dataset.classes})"
        )
        print("Using test dataset classes for evaluation")
        class_names = test_dataset.classes

    # Evaluate model
    accuracy, results = evaluate_model(
        model, test_loader, device, class_names, args.output_dir
    )

    print(f"\nEvaluation completed!")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
