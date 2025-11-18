import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# Define the same classes and functions from the training script
class FundusImageDataset(datasets.ImageFolder):
    """Custom dataset for fundus images with additional augmentations"""

    def __init__(self, root, transform=None):
        super().__init__(root, transform)


def get_transforms(image_size=224, is_training=True):
    """Get data transforms appropriate for fundus images"""
    if is_training:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                ),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transform


class ConvNeXtClassifier(nn.Module):
    """ConvNeXt-L with custom classification head"""

    def __init__(self, num_classes, dropout_rate=0.2):
        super().__init__()
        # Load pretrained ConvNeXt-L
        self.base_model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)

        # Get the in_features of the final layer
        in_features = self.base_model.classifier[2].in_features

        # Replace the classifier with a custom head
        self.base_model.classifier[2] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)


def compute_metrics(y_true, y_pred, y_prob, class_names):
    """Compute comprehensive metrics"""
    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ROC AUC scores
    if len(class_names) == 2:
        # Binary classification
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        roc_auc_dict = {"overall": roc_auc}
    else:
        # Multi-class classification
        roc_auc_dict = {}
        for i, class_name in enumerate(class_names):
            y_true_binary = (y_true == i).astype(int)
            if np.sum(y_true_binary) > 0:  # Check if class exists in test set
                try:
                    roc_auc_dict[class_name] = roc_auc_score(
                        y_true_binary, y_prob[:, i]
                    )
                except:
                    roc_auc_dict[class_name] = "N/A"

        # Macro and weighted average
        valid_scores = [v for v in roc_auc_dict.values() if isinstance(v, (int, float))]
        if valid_scores:
            roc_auc_dict["macro_avg"] = np.mean(valid_scores)
            weights = []
            weighted_scores = []
            for i, class_name in enumerate(class_names):
                count = np.sum(y_true == i)
                if (
                    count > 0
                    and class_name in roc_auc_dict
                    and isinstance(roc_auc_dict[class_name], (int, float))
                ):
                    weights.append(count)
                    weighted_scores.append(roc_auc_dict[class_name])

            if weights:
                roc_auc_dict["weighted_avg"] = np.average(
                    weighted_scores, weights=weights
                )

    return report, cm, roc_auc_dict


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))

    # Calculate percentages
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with both counts and percentages
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, y_prob, class_names, output_path):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(12, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    if len(class_names) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})", linewidth=2)
    else:
        # Multi-class classification
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            y_true_binary = (y_true == i).astype(int)
            if np.sum(y_true_binary) > 0:
                try:
                    fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
                    auc = roc_auc_score(y_true_binary, y_prob[:, i])
                    plt.plot(
                        fpr,
                        tpr,
                        color=color,
                        label=f"{class_name} (AUC = {auc:.3f})",
                        linewidth=2,
                    )
                except:
                    print(f"Could not compute ROC curve for class {class_name}")

    plt.plot([0, 1], [0, 1], "k--", label="Random classifier", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def test_model(model, dataloader, device, class_names):
    """Test the model and collect predictions"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Testing")):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

            # Get file paths - correctly accessing batch samples
            batch_size = len(labels)
            start_idx = batch_idx * dataloader.batch_size
            for i in range(batch_size):
                idx = start_idx + i
                if idx < len(dataloader.dataset.samples):
                    path = dataloader.dataset.samples[idx][0]
                    all_paths.append(path)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_paths


def main():
    parser = argparse.ArgumentParser(description="Test ConvNeXt-L on Fundus Images")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./test_results", help="Output directory"
    )
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument(
        "--tta_transforms", type=int, default=5, help="Number of TTA transforms"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    print(f'Validation accuracy: {checkpoint["val_acc"]:.2f}%')

    # Load class mapping
    checkpoint_dir = os.path.dirname(args.checkpoint)
    class_mapping_path = os.path.join(checkpoint_dir, "class_mapping.json")

    if not os.path.exists(class_mapping_path):
        print(f"Warning: class_mapping.json not found in {checkpoint_dir}")
        print("Attempting to infer from test dataset...")
        # We'll get it from the test dataset
        temp_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "test"))
        idx_to_class = {
            i: class_name for i, class_name in enumerate(temp_dataset.classes)
        }
        class_names = temp_dataset.classes
        num_classes = len(class_names)
    else:
        with open(class_mapping_path, "r") as f:
            class_mapping = json.load(f)
        idx_to_class = {int(k): v for k, v in class_mapping["idx_to_class"].items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

    # Model
    model = ConvNeXtClassifier(num_classes=num_classes)

    # Handle DataParallel checkpoints
    state_dict = checkpoint["model_state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        # Remove 'module.' prefix
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Data transform
    test_transform = get_transforms(args.image_size, is_training=False)

    # Test dataset
    test_dataset = FundusImageDataset(
        os.path.join(args.data_dir, "test"), transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Number of test samples: {len(test_dataset)}")

    # Test the model
    if args.tta:
        # Test-time augmentation
        print(f"Using test-time augmentation with {args.tta_transforms} transforms...")

        # Define TTA transforms
        tta_transforms = [test_transform]  # Original transform

        # Add horizontal flip
        tta_transforms.append(
            transforms.Compose(
                [
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        )

        # Add vertical flip
        tta_transforms.append(
            transforms.Compose(
                [
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.RandomVerticalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        )

        # Add rotations if requested
        if args.tta_transforms > 3:
            for angle in [90, 270]:
                tta_transforms.append(
                    transforms.Compose(
                        [
                            transforms.Resize((args.image_size, args.image_size)),
                            transforms.RandomRotation(degrees=(angle, angle)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    )
                )

        # Limit to requested number of transforms
        tta_transforms = tta_transforms[: args.tta_transforms]

        all_probs_tta = []
        for i, tta_transform in enumerate(tta_transforms):
            print(f"TTA transform {i+1}/{len(tta_transforms)}...")
            test_dataset.transform = tta_transform
            _, _, probs, _ = test_model(model, test_loader, device, class_names)
            all_probs_tta.append(probs)

        # Average predictions
        y_prob = np.mean(all_probs_tta, axis=0)
        y_pred = np.argmax(y_prob, axis=1)

        # Get true labels and paths with original transform
        test_dataset.transform = test_transform
        y_true, _, _, paths = test_model(model, test_loader, device, class_names)
    else:
        # Standard testing
        y_true, y_pred, y_prob, paths = test_model(
            model, test_loader, device, class_names
        )

    # Compute metrics
    report, cm, roc_auc_dict = compute_metrics(y_true, y_pred, y_prob, class_names)

    # Overall accuracy
    overall_acc = np.sum(y_true == y_pred) / len(y_true) * 100
    print(f"\nOverall Accuracy: {overall_acc:.2f}%")

    # Save results
    results = {
        "overall_accuracy": overall_acc,
        "classification_report": report,
        "roc_auc_scores": roc_auc_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "num_test_samples": len(y_true),
        "checkpoint_path": args.checkpoint,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_val_acc": checkpoint["val_acc"],
        "test_time_augmentation": args.tta,
        "tta_transforms": args.tta_transforms if args.tta else 0,
    }

    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save detailed classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(args.output_dir, "classification_report.csv"))

    # Save per-sample predictions
    predictions_df = pd.DataFrame(
        {
            "image_path": paths,
            "true_label": [class_names[i] for i in y_true],
            "predicted_label": [class_names[i] for i in y_pred],
            "correct": y_true == y_pred,
            "confidence": np.max(y_prob, axis=1),
        }
    )

    # Add probability columns
    for i, class_name in enumerate(class_names):
        predictions_df[f"prob_{class_name}"] = y_prob[:, i]

    predictions_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    # Save misclassified samples
    misclassified_df = predictions_df[predictions_df["correct"] == False].copy()
    misclassified_df.to_csv(
        os.path.join(args.output_dir, "misclassified.csv"), index=False
    )
    print(f"Number of misclassified samples: {len(misclassified_df)}")

    # Plot confusion matrix
    plot_confusion_matrix(
        cm, class_names, os.path.join(args.output_dir, "confusion_matrix.png")
    )

    # Plot ROC curves
    plot_roc_curves(
        y_true, y_prob, class_names, os.path.join(args.output_dir, "roc_curves.png")
    )

    # Print summary
    print("\nPer-class Performance:")
    print("-" * 80)
    print(
        f'{"Class":20s} | {"Precision":>9s} | {"Recall":>9s} | {"F1-Score":>9s} | {"AUC":>9s} | {"Support":>7s}'
    )
    print("-" * 80)

    for class_name in class_names:
        if class_name in report:
            precision = report[class_name]["precision"]
            recall = report[class_name]["recall"]
            f1 = report[class_name]["f1-score"]
            support = int(report[class_name]["support"])
            auc = roc_auc_dict.get(class_name, "N/A")
            if isinstance(auc, float):
                auc_str = f"{auc:.3f}"
            else:
                auc_str = str(auc)

            print(
                f"{class_name:20s} | {precision:9.3f} | {recall:9.3f} | {f1:9.3f} | {auc_str:>9s} | {support:7d}"
            )

    print("-" * 80)

    # Print weighted averages
    if "weighted avg" in report:
        wavg = report["weighted avg"]
        wauc = roc_auc_dict.get("weighted_avg", "N/A")
        if isinstance(wauc, float):
            wauc_str = f"{wauc:.3f}"
        else:
            wauc_str = str(wauc)
        print(
            f'{"Weighted Average":20s} | {wavg["precision"]:9.3f} | {wavg["recall"]:9.3f} | '
            f'{wavg["f1-score"]:9.3f} | {wauc_str:>9s} | {int(wavg["support"]):7d}'
        )

    print("\nResults saved to:", args.output_dir)


if __name__ == "__main__":
    main()
