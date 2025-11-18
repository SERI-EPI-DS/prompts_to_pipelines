import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torchvision import transforms, datasets
import timm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


class TestTimeAugmentation:
    """Test-time augmentation for improved predictions"""

    def __init__(self, input_size=384):
        self.input_size = input_size
        self.base_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_transforms(self):
        """Get multiple transforms for TTA"""
        transforms_list = []

        # Original
        transforms_list.append(
            transforms.Compose(
                [
                    transforms.Resize(int(self.input_size * 1.1)),
                    transforms.CenterCrop(self.input_size),
                    self.base_transform,
                ]
            )
        )

        # Horizontal flip
        transforms_list.append(
            transforms.Compose(
                [
                    transforms.Resize(int(self.input_size * 1.1)),
                    transforms.CenterCrop(self.input_size),
                    transforms.RandomHorizontalFlip(p=1.0),
                    self.base_transform,
                ]
            )
        )

        # Vertical flip
        transforms_list.append(
            transforms.Compose(
                [
                    transforms.Resize(int(self.input_size * 1.1)),
                    transforms.CenterCrop(self.input_size),
                    transforms.RandomVerticalFlip(p=1.0),
                    self.base_transform,
                ]
            )
        )

        # Different crops
        for scale in [0.9, 1.0, 1.1]:
            transforms_list.append(
                transforms.Compose(
                    [
                        transforms.Resize(int(self.input_size * scale * 1.1)),
                        transforms.CenterCrop(int(self.input_size * scale)),
                        transforms.Resize(self.input_size),
                        self.base_transform,
                    ]
                )
            )

        return transforms_list


class ConvNextTester:
    def __init__(self, model_path, device="cuda"):
        self.device = device

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.args = checkpoint["args"]
        self.class_mapping = checkpoint["class_mapping"]
        self.num_classes = len(self.class_mapping)

        # Initialize model
        self.model = timm.create_model(
            self.args.model_name,
            pretrained=False,
            num_classes=self.num_classes,
            drop_rate=0.0,  # No dropout during testing
            drop_path_rate=0.0,
        ).to(device)

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Initialize TTA
        self.tta = TestTimeAugmentation(self.args.input_size)

    def predict_single_image(self, image_path, use_tta=True):
        """Predict single image with optional TTA"""
        image = Image.open(image_path).convert("RGB")

        if use_tta:
            predictions = []
            transforms_list = self.tta.get_transforms()

            for transform in transforms_list:
                img_tensor = transform(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    with autocast():
                        output = self.model(img_tensor)
                        probs = torch.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())

            # Average predictions
            avg_probs = np.mean(predictions, axis=0)[0]
            std_probs = np.std(predictions, axis=0)[0]
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(int(self.args.input_size * 1.1)),
                    transforms.CenterCrop(self.args.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            img_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                with autocast():
                    output = self.model(img_tensor)
                    avg_probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    std_probs = np.zeros_like(avg_probs)

        # Get prediction
        pred_idx = np.argmax(avg_probs)
        pred_class = self.class_mapping[str(pred_idx)]
        confidence = avg_probs[pred_idx]
        uncertainty = std_probs[pred_idx]

        results = {
            "predicted_class": pred_class,
            "confidence": float(confidence),
            "uncertainty": float(uncertainty),
            "probabilities": {
                self.class_mapping[str(i)]: float(avg_probs[i])
                for i in range(self.num_classes)
            },
        }

        return results

    def test_dataset(self, test_dir, use_tta=True, save_predictions=True):
        """Test on entire dataset"""
        # Create dataset
        transform = transforms.Compose(
            [
                transforms.Resize(int(self.args.input_size * 1.1)),
                transforms.CenterCrop(self.args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1 if use_tta else self.args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        all_preds = []
        all_probs = []
        all_targets = []
        all_paths = []

        # Get all image paths in order
        image_paths = [s[0] for s in test_dataset.samples]

        if use_tta:
            # TTA mode - process one image at a time
            for idx, (inputs, targets) in enumerate(
                tqdm(test_loader, desc="Testing with TTA")
            ):
                image_path = image_paths[idx]

                # Use TTA prediction
                results = self.predict_single_image(image_path, use_tta=True)
                pred_idx = list(self.class_mapping.keys())[
                    list(self.class_mapping.values()).index(results["predicted_class"])
                ]
                probs = [
                    results["probabilities"][self.class_mapping[str(i)]]
                    for i in range(self.num_classes)
                ]

                all_preds.append(int(pred_idx))
                all_probs.append(probs)
                all_targets.extend(targets.numpy())
                all_paths.append(image_path)
        else:
            # Non-TTA mode - can process in batches
            batch_idx = 0
            with torch.no_grad():
                for inputs, targets in tqdm(test_loader, desc="Testing"):
                    batch_size = inputs.size(0)
                    inputs = inputs.to(self.device)

                    with autocast():
                        outputs = self.model(inputs)
                        probs = torch.softmax(outputs, dim=1)

                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_targets.extend(targets.numpy())

                    # Add corresponding image paths for this batch
                    start_idx = batch_idx * test_loader.batch_size
                    end_idx = start_idx + batch_size
                    all_paths.extend(image_paths[start_idx:end_idx])

                    batch_idx += 1

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)

        # Calculate metrics
        accuracy = np.mean(all_preds == all_targets) * 100

        # Classification report
        class_names = [test_dataset.classes[i] for i in range(self.num_classes)]
        report = classification_report(
            all_targets, all_preds, target_names=class_names, output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)

        # ROC-AUC for multi-class
        y_true_binary = label_binarize(
            all_targets, classes=list(range(self.num_classes))
        )
        try:
            auc_scores = {}
            for i, class_name in enumerate(class_names):
                if (
                    len(np.unique(y_true_binary[:, i])) > 1
                ):  # Check if both classes are present
                    auc_scores[class_name] = roc_auc_score(
                        y_true_binary[:, i], all_probs[:, i]
                    )
                else:
                    auc_scores[class_name] = np.nan

            # Calculate macro AUC excluding NaN values
            valid_aucs = [v for v in auc_scores.values() if not np.isnan(v)]
            macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
        except Exception as e:
            print(f"Warning: Could not calculate AUC scores: {e}")
            auc_scores = {}
            macro_auc = 0.0

        results = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "auc_scores": auc_scores,
            "macro_auc": macro_auc,
            "num_samples": len(all_targets),
        }

        # Save predictions if requested
        if save_predictions:
            # Create probability columns dictionary
            prob_columns = {
                f"prob_{class_name}": all_probs[:, i]
                for i, class_name in enumerate(class_names)
            }

            predictions_df = pd.DataFrame(
                {
                    "image_path": all_paths,
                    "true_label": [class_names[t] for t in all_targets],
                    "predicted_label": [class_names[p] for p in all_preds],
                    "confidence": [
                        all_probs[i, all_preds[i]] for i in range(len(all_preds))
                    ],
                    **prob_columns,
                }
            )

            predictions_df["correct"] = (
                predictions_df["true_label"] == predictions_df["predicted_label"]
            )

            return results, predictions_df

        return results, None


def visualize_results(results, output_dir, class_names):
    """Create visualization plots"""
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = np.array(results["confusion_matrix"])

    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create mask for text color
    mask = cm_normalized > 0.5

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )

    # Add percentage annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = cm_normalized[i, j] * 100
            text_color = "white" if mask[i, j] else "black"
            plt.text(
                j + 0.5,
                i + 0.7,
                f"({percentage:.1f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Per-class metrics
    report = results["classification_report"]
    metrics_data = []
    for class_name in class_names:
        if class_name in report:
            metrics_data.append(
                {
                    "Class": class_name,
                    "Precision": report[class_name]["precision"],
                    "Recall": report[class_name]["recall"],
                    "F1-Score": report[class_name]["f1-score"],
                    "Support": report[class_name]["support"],
                }
            )

    metrics_df = pd.DataFrame(metrics_data)

    # Plot per-class metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # F1-scores
    ax1.bar(metrics_df["Class"], metrics_df["F1-Score"])
    ax1.set_xlabel("Class")
    ax1.set_ylabel("F1-Score")
    ax1.set_title("F1-Score by Class")
    ax1.set_ylim(0, 1)
    for i, v in enumerate(metrics_df["F1-Score"]):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Precision and Recall
    x = np.arange(len(metrics_df))
    width = 0.35
    ax2.bar(x - width / 2, metrics_df["Precision"], width, label="Precision", alpha=0.8)
    ax2.bar(x + width / 2, metrics_df["Recall"], width, label="Recall", alpha=0.8)
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision and Recall by Class")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_df["Class"])
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_metrics.png"), dpi=300)
    plt.close()

    # Save metrics table
    metrics_df.to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)

    # ROC curves if available
    if results.get("auc_scores"):
        plt.figure(figsize=(10, 8))
        for class_name, auc in results["auc_scores"].items():
            if not np.isnan(auc):
                plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
                break

        # Add AUC scores to legend
        for class_name, auc in results["auc_scores"].items():
            if not np.isnan(auc):
                label = f"{class_name} (AUC = {auc:.3f})"
            else:
                label = f"{class_name} (AUC = N/A)"

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f'ROC Curves (Macro-AUC = {results["macro_auc"]:.3f})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test ConvNeXt-L for Fundus Image Classification"
    )

    # Required arguments
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Path to test dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory to save test results",
    )

    # Test options
    parser.add_argument(
        "--use_tta", action="store_true", help="Use test-time augmentation"
    )
    parser.add_argument(
        "--single_image",
        type=str,
        default=None,
        help="Path to single image for prediction",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize tester
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        tester = ConvNextTester(args.model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if args.single_image:
        # Single image prediction
        print(f"Predicting single image: {args.single_image}")
        results = tester.predict_single_image(args.single_image, use_tta=args.use_tta)

        print("\nPrediction Results:")
        print(f"Predicted Class: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.4f}")
        if args.use_tta:
            print(f"Uncertainty: {results['uncertainty']:.4f}")
        print("\nClass Probabilities:")
        for class_name, prob in sorted(
            results["probabilities"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {class_name}: {prob:.4f}")

        # Save results
        with open(os.path.join(args.output_dir, "single_prediction.json"), "w") as f:
            json.dump(results, f, indent=4)
    else:
        # Full dataset testing
        print(f"Testing on dataset: {args.test_dir}")
        print(f"Test-time augmentation: {'Enabled' if args.use_tta else 'Disabled'}")

        results, predictions_df = tester.test_dataset(
            args.test_dir, use_tta=args.use_tta, save_predictions=True
        )

        # Print summary
        print(f"\nTest Results:")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print(f"Macro AUC: {results['macro_auc']:.4f}")
        print(f"Number of samples: {results['num_samples']}")

        # Save results
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        # Save predictions
        if predictions_df is not None:
            predictions_df.to_csv(
                os.path.join(args.output_dir, "predictions.csv"), index=False
            )

            # Save misclassified images
            misclassified = predictions_df[~predictions_df["correct"]]
            misclassified.to_csv(
                os.path.join(args.output_dir, "misclassified.csv"), index=False
            )
            print(
                f"\nMisclassified samples: {len(misclassified)}/{len(predictions_df)}"
            )

            # Print top misclassified pairs
            if len(misclassified) > 0:
                print("\nTop misclassification patterns:")
                misclass_patterns = misclassified.groupby(
                    ["true_label", "predicted_label"]
                ).size()
                misclass_patterns = misclass_patterns.sort_values(ascending=False).head(
                    10
                )
                for (true_label, pred_label), count in misclass_patterns.items():
                    print(f"  {true_label} -> {pred_label}: {count} samples")

        # Create visualizations
        class_names = list(tester.class_mapping.values())
        visualize_results(results, args.output_dir, class_names)

        # Print detailed classification report
        print("\nClassification Report:")
        report_df = pd.DataFrame(results["classification_report"]).transpose()
        print(report_df.round(3))

        print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
