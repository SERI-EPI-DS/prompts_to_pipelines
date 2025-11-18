import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def get_transforms(image_size):
    """Get test time augmentation transforms"""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


def test_time_augmentation(model, image, num_augmentations=5):
    """Apply test time augmentation for more robust predictions"""
    model.eval()
    predictions = []

    # Original image
    with torch.no_grad():
        output = model(image.unsqueeze(0).cuda())
        predictions.append(torch.softmax(output, dim=1))

    # Augmented versions
    augmentation_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomRotation(degrees=10),
        transforms.RandomRotation(degrees=-10),
    ]

    for aug_transform in augmentation_transforms[: num_augmentations - 1]:
        aug_image = aug_transform(image)
        with torch.no_grad():
            output = model(aug_image.unsqueeze(0).cuda())
            predictions.append(torch.softmax(output, dim=1))

    # Average predictions
    avg_prediction = torch.stack(predictions).mean(dim=0)
    return avg_prediction


def evaluate_model(model, test_loader, class_names, use_tta=False):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.cuda(), labels.cuda()

            if use_tta and len(images) == 1:
                # Apply TTA for single image batches
                outputs = test_time_augmentation(model, images[0])
            else:
                outputs = model(images)
                outputs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    # Calculate per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    # Calculate weighted averages
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    # Calculate AUC if binary classification
    auc_score = None
    if len(class_names) == 2:
        auc_score = roc_auc_score(all_labels, all_probs[:, 1])

    return {
        "predictions": all_preds,
        "probabilities": all_probs,
        "labels": all_labels,
        "accuracy": float(accuracy),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "class_metrics": class_metrics,
        "auc": float(auc_score) if auc_score is not None else None,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=True):
    """Plot confusion matrix with normalization option"""
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count" if not normalize else "Proportion"},
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(y_true, y_probs, class_names, save_path):
    """Plot ROC curves for each class"""
    n_classes = len(class_names)

    # Compute ROC curve for each class
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        # Create binary labels for current class
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]

        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_prediction_report(results, class_names, output_dir):
    """Generate comprehensive prediction report"""
    report = classification_report(
        results["labels"],
        results["predictions"],
        target_names=class_names,
        output_dict=True,
    )

    # Convert to DataFrame for better visualization
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_dir, "classification_report.csv"))

    # Create summary statistics
    summary = {
        "Total Samples": int(len(results["labels"])),
        "Overall Accuracy": float(results["accuracy"]),
        "Weighted Precision": float(results["weighted_precision"]),
        "Weighted Recall": float(results["weighted_recall"]),
        "Weighted F1-Score": float(results["weighted_f1"]),
    }

    if results["auc"] is not None:
        summary["AUC"] = float(results["auc"])

    # Add per-class statistics
    for class_name, metrics in results["class_metrics"].items():
        summary[f"{class_name}_precision"] = float(metrics["precision"])
        summary[f"{class_name}_recall"] = float(metrics["recall"])
        summary[f"{class_name}_f1"] = float(metrics["f1"])
        summary[f"{class_name}_support"] = int(metrics["support"])

    # Convert all numpy types to native Python types
    summary = convert_numpy_types(summary)

    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return summary


def predict_single_image(model, image_path, transform, class_names):
    """Predict class for a single image"""
    from PIL import Image

    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()

    # Get top-k predictions
    top_k = min(5, len(class_names))
    probs, indices = probabilities[0].topk(top_k)

    predictions = {}
    for i in range(top_k):
        predictions[class_names[indices[i].item()]] = float(probs[i].item())

    return predicted_class, confidence_score, predictions


def get_model_name_from_image_size(image_size):
    """Determine the appropriate model name based on image size"""
    if image_size == 192:
        return "swinv2_base_window12_192_22k"
    elif image_size == 256:
        return "swinv2_base_window16_256"
    elif image_size == 384:
        return "swinv2_base_window12to24_192to384_22kft1k"
    else:
        print(f"Warning: Non-standard image size {image_size}. Using flexible variant.")
        return "swinv2_base_window12to24_192to384_22kft1k"


def main(args):
    # Load model configuration
    checkpoint = torch.load(args.model_path, map_location="cuda")
    config = checkpoint.get("config", {})
    class_names = checkpoint.get("class_names", [])

    # Get image size from checkpoint or use default
    image_size = checkpoint.get("image_size", config.get("image_size", 192))
    print(f"Using image size: {image_size}x{image_size}")

    if not class_names:
        # Get class names from test dataset
        test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "test"))
        class_names = test_dataset.classes

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create model with the appropriate architecture
    num_classes = len(class_names)
    model_name = get_model_name_from_image_size(image_size)

    print(f"Creating model: {model_name}")
    model = timm.create_model(
        model_name, pretrained=False, num_classes=num_classes, img_size=image_size
    )

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Single image prediction
    if args.image_path:
        transform = get_transforms(image_size)
        predicted_class, confidence, all_predictions = predict_single_image(
            model, args.image_path, transform, class_names
        )

        print(f"\nSingle Image Prediction:")
        print(f"Image: {args.image_path}")
        print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.4f})")
        print(f"\nTop-5 Predictions:")
        for class_name, prob in all_predictions.items():
            print(f"  {class_name}: {prob:.4f}")

        # Save prediction results
        single_prediction_result = {
            "image_path": args.image_path,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "top_5_predictions": all_predictions,
        }

        with open(
            os.path.join(args.output_dir, "single_image_prediction.json"), "w"
        ) as f:
            json.dump(single_prediction_result, f, indent=4)

        return

    # Test dataset evaluation
    transform = get_transforms(image_size)
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"\nEvaluating on test set...")
    print(f"Test samples: {len(test_dataset)}")

    # Evaluate model
    results = evaluate_model(model, test_loader, class_names, use_tta=args.use_tta)

    # Generate reports and visualizations
    print(f"\nGenerating evaluation reports...")

    # Classification report
    summary = generate_prediction_report(results, class_names, args.output_dir)

    # Confusion matrices
    plot_confusion_matrix(
        results["labels"],
        results["predictions"],
        class_names,
        os.path.join(args.output_dir, "confusion_matrix.png"),
        normalize=False,
    )

    plot_confusion_matrix(
        results["labels"],
        results["predictions"],
        class_names,
        os.path.join(args.output_dir, "confusion_matrix_normalized.png"),
        normalize=True,
    )

    # ROC curves
    if len(class_names) <= 10:  # Only plot ROC curves for reasonable number of classes
        plot_roc_curves(
            results["labels"],
            results["probabilities"],
            class_names,
            os.path.join(args.output_dir, "roc_curves.png"),
        )

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "true_label": results["labels"],
            "predicted_label": results["predictions"],
            "true_class": [class_names[i] for i in results["labels"]],
            "predicted_class": [class_names[i] for i in results["predictions"]],
        }
    )

    # Add probability columns
    for i, class_name in enumerate(class_names):
        predictions_df[f"prob_{class_name}"] = results["probabilities"][:, i]

    predictions_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    # Print summary
    print(f"\nEvaluation Results:")
    print(f"Overall Accuracy: {summary['Overall Accuracy']:.4f}")
    print(f"Weighted Precision: {summary['Weighted Precision']:.4f}")
    print(f"Weighted Recall: {summary['Weighted Recall']:.4f}")
    print(f"Weighted F1-Score: {summary['Weighted F1-Score']:.4f}")
    if summary.get("AUC"):
        print(f"AUC: {summary['AUC']:.4f}")

    print(f"\nPer-class metrics:")
    for class_name in class_names:
        print(f"\n{class_name}:")
        print(f"  Precision: {summary[f'{class_name}_precision']:.4f}")
        print(f"  Recall: {summary[f'{class_name}_recall']:.4f}")
        print(f"  F1-Score: {summary[f'{class_name}_f1']:.4f}")
        print(f"  Support: {summary[f'{class_name}_support']}")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Swin-V2-B model for fundus image classification"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data_dir", type=str, help="Path to dataset directory containing test folder"
    )
    parser.add_argument(
        "--image_path", type=str, help="Path to single image for prediction"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory to save test results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument(
        "--use_tta", action="store_true", help="Use test time augmentation"
    )

    args = parser.parse_args()

    # Validation
    if not args.image_path and not args.data_dir:
        parser.error("Either --image_path or --data_dir must be specified")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Testing will be slower on CPU.")
    else:
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")

    main(args)
