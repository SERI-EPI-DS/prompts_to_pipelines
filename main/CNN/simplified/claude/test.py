import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast
from timm import create_model
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class ConvNeXtClassifier(nn.Module):
    """ConvNeXt-Large classifier."""

    def __init__(self, num_classes, dropout_rate=0.2):
        super().__init__()
        self.model = create_model(
            "convnext_large",
            pretrained=False,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )

    def forward(self, x):
        return self.model(x)


def get_test_transforms(input_size=384):
    """Get test-time transforms."""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def test_time_augmentation(model, image, device, n_augmentations=5):
    """Apply test-time augmentation for more robust predictions."""
    tta_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]
    )

    predictions = []

    # Original prediction
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        predictions.append(torch.softmax(output, dim=1).cpu())

    # Augmented predictions
    for _ in range(n_augmentations - 1):
        aug_image = tta_transforms(image)
        with torch.no_grad():
            output = model(aug_image.unsqueeze(0).to(device))
            predictions.append(torch.softmax(output, dim=1).cpu())

    # Average predictions
    return torch.stack(predictions).mean(dim=0)


def compute_confidence_intervals(accuracies, confidence=0.95):
    """Compute confidence intervals for accuracy."""
    mean_acc = np.mean(accuracies)
    std_err = stats.sem(accuracies)
    interval = std_err * stats.t.ppf((1 + confidence) / 2, len(accuracies) - 1)
    return mean_acc, mean_acc - interval, mean_acc + interval


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix."""
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
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curves(y_true, y_scores, class_names, output_path):
    """Plot ROC curves for multiclass classification."""
    n_classes = len(class_names)

    # Compute ROC curve for each class
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score_binary = y_scores[:, i]

        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
        auc = roc_auc_score(y_true_binary, y_score_binary)

        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report(results, output_dir):
    """Generate comprehensive test report."""
    report_path = os.path.join(output_dir, "test_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ConvNeXt-L Fundus Image Classification Test Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Test Dataset: {results['test_dataset']}\n")
        f.write(f"Number of Test Samples: {results['num_samples']}\n")
        f.write(f"Number of Classes: {results['num_classes']}\n\n")

        f.write("Overall Performance:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.2f}%\n")
        f.write(
            f"95% Confidence Interval: [{results['ci_lower']:.2f}%, {results['ci_upper']:.2f}%]\n\n"
        )

        f.write("Classification Report:\n")
        f.write("-" * 40 + "\n")
        f.write(results["classification_report"] + "\n")

        if "tta_accuracy" in results:
            f.write(
                f"\nTest-Time Augmentation Accuracy: {results['tta_accuracy']:.2f}%\n"
            )

        f.write("\nPer-Class AUC Scores:\n")
        f.write("-" * 40 + "\n")
        for class_name, auc in results["auc_scores"].items():
            f.write(f"{class_name}: {auc:.3f}\n")

    print(f"Test report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test ConvNeXt-L fundus image classifier"
    )
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Path to test dataset"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save results"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")
    parser.add_argument(
        "--use_tta", action="store_true", help="Use test-time augmentation"
    )
    parser.add_argument(
        "--tta_samples", type=int, default=5, help="Number of TTA samples"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # Create model and load weights
    model = ConvNeXtClassifier(
        num_classes, dropout_rate=checkpoint["args"].get("dropout", 0.2)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from: {args.model_path}")
    print(f"Number of classes: {num_classes}")

    # Create test dataset and dataloader
    test_transform = get_test_transforms(args.input_size)
    test_dataset = datasets.ImageFolder(args.test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Test the model
    all_preds = []
    all_targets = []
    all_scores = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            if args.use_tta:
                # Test-time augmentation
                batch_preds = []
                for i in range(inputs.size(0)):
                    tta_pred = test_time_augmentation(
                        model, inputs[i].cpu(), device, args.tta_samples
                    )
                    batch_preds.append(tta_pred)
                outputs = torch.cat(batch_preds, dim=0)
            else:
                with autocast():
                    outputs = model(inputs)
                outputs = torch.softmax(outputs, dim=1).cpu()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(targets.cpu().numpy())
            all_scores.extend(outputs.numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)

    # Calculate metrics
    accuracy = 100.0 * np.mean(all_preds == all_targets)

    # Confidence intervals
    sample_accuracies = [
        100.0 * np.mean(all_preds[i : i + 100] == all_targets[i : i + 100])
        for i in range(0, len(all_preds), 100)
        if i + 100 <= len(all_preds)
    ]
    if len(sample_accuracies) > 1:
        mean_acc, ci_lower, ci_upper = compute_confidence_intervals(sample_accuracies)
    else:
        mean_acc, ci_lower, ci_upper = accuracy, accuracy, accuracy

    # Classification report
    report = classification_report(
        all_targets, all_preds, target_names=class_names, digits=3
    )

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(
        cm, class_names, os.path.join(args.output_dir, "confusion_matrix.png")
    )

    # ROC curves and AUC scores
    plot_roc_curves(
        all_targets,
        all_scores,
        class_names,
        os.path.join(args.output_dir, "roc_curves.png"),
    )

    # Calculate per-class AUC
    auc_scores = {}
    for i, class_name in enumerate(class_names):
        y_true_binary = (all_targets == i).astype(int)
        y_score_binary = all_scores[:, i]
        auc = roc_auc_score(y_true_binary, y_score_binary)
        auc_scores[class_name] = auc

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "image_path": [
                test_dataset.samples[i][0] for i in range(len(test_dataset))
            ],
            "true_label": [class_names[t] for t in all_targets],
            "predicted_label": [class_names[p] for p in all_preds],
            "confidence": np.max(all_scores, axis=1),
        }
    )
    predictions_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    # Generate results dictionary
    results = {
        "test_dataset": args.test_dir,
        "num_samples": len(test_dataset),
        "num_classes": num_classes,
        "accuracy": accuracy,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "classification_report": report,
        "auc_scores": auc_scores,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    if args.use_tta:
        results["tta_accuracy"] = accuracy
        results["tta_samples"] = args.tta_samples

    # Save results
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Generate report
    generate_report(results, args.output_dir)

    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
