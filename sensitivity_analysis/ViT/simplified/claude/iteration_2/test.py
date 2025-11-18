import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def get_transforms(input_size=384):
    """Get test-time transforms"""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_tta_transforms(input_size=384):
    """Test-time augmentation transforms"""
    return [
        transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    ]


class TTADataset(torch.utils.data.Dataset):
    """Dataset wrapper for test-time augmentation"""

    def __init__(self, dataset, transforms_list):
        self.dataset = dataset
        self.transforms_list = transforms_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset.dataset.samples[idx]
        img = transforms.functional.pil_loader(img)

        augmented_images = []
        for transform in self.transforms_list:
            augmented_images.append(transform(img))

        return torch.stack(augmented_images), label


def test_model(model, dataloader, device, num_classes, use_tta=False, input_size=384):
    """Test model with optional test-time augmentation"""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing"):
            if use_tta:
                # inputs shape: (batch_size, num_augmentations, channels, height, width)
                batch_size = inputs.size(0)
                num_augs = inputs.size(1)

                # Reshape for batch processing
                inputs = inputs.view(-1, inputs.size(2), inputs.size(3), inputs.size(4))
                inputs = inputs.to(device)

                with autocast():
                    outputs = model(inputs)

                # Reshape back and average predictions
                outputs = outputs.view(batch_size, num_augs, -1)
                probs = torch.softmax(outputs, dim=2).mean(
                    dim=1
                )  # Average over augmentations
            else:
                inputs = inputs.to(device)
                with autocast():
                    outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

            _, predicted = probs.max(1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_targets)


def plot_roc_curves(y_true, y_probs, class_names, save_path):
    """Plot ROC curves for multi-class classification"""
    n_classes = len(class_names)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(
        np.eye(n_classes)[y_true].ravel(), y_probs.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot
    plt.figure(figsize=(10, 8))

    # Plot micro-average
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    # Plot macro-average
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.3f})',
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    # Plot ROC curves for each class
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    for i, color in enumerate(colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return roc_auc


def save_predictions(image_paths, predictions, probabilities, class_names, save_path):
    """Save detailed predictions to CSV"""
    results = []

    for path, pred, probs in zip(image_paths, predictions, probabilities):
        result = {
            "image_path": path,
            "predicted_class": class_names[pred],
            "confidence": probs[pred],
        }
        # Add probabilities for each class
        for i, class_name in enumerate(class_names):
            result[f"prob_{class_name}"] = probs[i]

        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Saved predictions for {len(df)} images to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Test Swin-V2-B on fundus images")
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Path to test dataset"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save test results"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--use_tta", action="store_true", help="Use test-time augmentation"
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Use EMA model if available"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # Get model configuration from checkpoint
    model_name = checkpoint.get("model_name", "swinv2_base_window8_256")
    saved_input_size = checkpoint.get("input_size", args.input_size)

    # Use the saved input size if different from specified
    if saved_input_size != args.input_size:
        print(
            f"Note: Using input size {saved_input_size} from checkpoint (instead of {args.input_size})"
        )
        args.input_size = saved_input_size

    # Create model
    print(f"Loading model: {model_name}")
    try:
        if (
            model_name == "swinv2_base_window8"
            or model_name == "swinv2_base_window8_256"
        ):
            model = timm.create_model(
                "swinv2_base_window8_256", pretrained=False, num_classes=num_classes
            )
        else:
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes,
                img_size=args.input_size,
            )
    except Exception as e:
        print(f"Failed to create {model_name}: {e}")
        print("Using swinv2_base_window8_256")
        model = timm.create_model(
            "swinv2_base_window8_256", pretrained=False, num_classes=num_classes
        )

    # Load weights
    if args.use_ema and "model_ema_state_dict" in checkpoint:
        print("Loading EMA model weights")
        # Extract the actual model state dict from EMA
        ema_state_dict = checkpoint["model_ema_state_dict"]
        if "module" in ema_state_dict:
            model.load_state_dict(ema_state_dict["module"])
        else:
            model.load_state_dict(ema_state_dict)
    else:
        print("Loading standard model weights")
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    # Create test dataset
    if args.use_tta:
        print("Using test-time augmentation")
        base_dataset = ImageFolder(args.test_dir)
        test_dataset = TTADataset(base_dataset, get_tta_transforms(args.input_size))
        # Get image paths from base dataset
        image_paths = [base_dataset.samples[i][0] for i in range(len(base_dataset))]
    else:
        test_transform = get_transforms(args.input_size)
        test_dataset = ImageFolder(args.test_dir, transform=test_transform)
        image_paths = [test_dataset.samples[i][0] for i in range(len(test_dataset))]

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Test model
    print(f"\nTesting model on {len(test_dataset)} images...")
    predictions, probabilities, targets = test_model(
        model,
        test_loader,
        device,
        num_classes,
        use_tta=args.use_tta,
        input_size=args.input_size,
    )

    # Calculate metrics
    accuracy = np.mean(predictions == targets)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # Classification report
    report = classification_report(
        targets, predictions, target_names=class_names, output_dict=True, digits=4
    )
    print("\nClassification Report:")
    print(
        classification_report(targets, predictions, target_names=class_names, digits=4)
    )

    # Save classification report
    with open(os.path.join(args.output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(targets, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
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
        os.path.join(args.output_dir, "confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot ROC curves and get AUC scores
    roc_auc_scores = plot_roc_curves(
        targets,
        probabilities,
        class_names,
        os.path.join(args.output_dir, "roc_curves.png"),
    )

    # Calculate overall AUC scores
    if num_classes == 2:
        # Binary classification
        overall_auc = roc_auc_score(targets, probabilities[:, 1])
        print(f"\nBinary ROC AUC Score: {overall_auc:.4f}")
    else:
        # Multi-class classification
        try:
            # One-vs-Rest AUC
            ovr_auc = roc_auc_score(
                targets, probabilities, multi_class="ovr", average="macro"
            )
            print(f"\nMulti-class ROC AUC Score (macro): {ovr_auc:.4f}")

            # Weighted average AUC
            ovr_auc_weighted = roc_auc_score(
                targets, probabilities, multi_class="ovr", average="weighted"
            )
            print(f"Multi-class ROC AUC Score (weighted): {ovr_auc_weighted:.4f}")
        except Exception as e:
            print(f"\nNote: Could not calculate multi-class AUC: {e}")
            ovr_auc = None
            ovr_auc_weighted = None

    # Save predictions
    save_predictions(
        image_paths,
        predictions,
        probabilities,
        class_names,
        os.path.join(args.output_dir, "predictions.csv"),
    )

    # Per-class accuracy
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = targets == i
        if class_mask.sum() > 0:
            class_acc = (predictions[class_mask] == i).mean()
            per_class_acc[class_name] = float(class_acc)
            print(f"{class_name}: {class_acc:.4f} ({class_mask.sum()} samples)")

    # Save summary
    summary = {
        "test_accuracy": float(accuracy),
        "total_images": len(test_dataset),
        "class_names": class_names,
        "per_class_accuracy": per_class_acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "use_tta": args.use_tta,
        "use_ema": args.use_ema,
        "model_name": model_name,
        "input_size": args.input_size,
    }

    if num_classes == 2:
        summary["binary_auc"] = float(overall_auc)
    else:
        if ovr_auc is not None:
            summary["multiclass_auc_macro"] = float(ovr_auc)
        if ovr_auc_weighted is not None:
            summary["multiclass_auc_weighted"] = float(ovr_auc_weighted)
        summary["per_class_auc"] = {
            class_names[i]: float(roc_auc_scores[i]) for i in range(num_classes)
        }

    with open(os.path.join(args.output_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nTest results saved to: {args.output_dir}")
    print("Files generated:")
    print("- classification_report.json: Detailed per-class metrics")
    print("- confusion_matrix.png: Visual confusion matrix")
    print("- roc_curves.png: ROC curves for each class")
    print("- predictions.csv: Individual predictions for each image")
    print("- test_summary.json: Overall test summary")


if __name__ == "__main__":
    main()
