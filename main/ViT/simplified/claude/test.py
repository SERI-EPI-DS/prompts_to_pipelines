import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.cuda.amp import autocast
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
import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Available Swin-V2 models with their expected input sizes
SWIN_MODELS = {
    "swinv2_tiny_window8_256": 256,
    "swinv2_small_window8_256": 256,
    "swinv2_base_window8_256": 256,
    "swinv2_base_window12_192": 192,
    "swinv2_base_window16_256": 256,
    "swinv2_base_window24_384": 384,
    "swinv2_large_window12_192": 192,
    "swinv2_large_window16_256": 256,
    "swinv2_large_window24_384": 384,
}


def detect_model_config(checkpoint_path):
    """Detect model configuration from checkpoint or config file"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Try to get from checkpoint
    model_name = checkpoint.get("model_name")
    input_size = checkpoint.get("input_size")

    # If not in checkpoint, try to load from config file in the same directory
    if model_name is None or input_size is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                model_name = config.get("model_name", "swinv2_base_window16_256")
                input_size = config.get("input_size")
        else:
            # Default fallback
            print("Warning: Could not find model configuration. Using defaults.")
            model_name = "swinv2_base_window16_256"
            input_size = None

    # Auto-detect input size from model name if not specified
    if input_size is None and model_name in SWIN_MODELS:
        input_size = SWIN_MODELS[model_name]
        print(f"Auto-detected input size {input_size} for model {model_name}")
    elif input_size is None:
        # Final fallback
        input_size = 256
        print(f"Warning: Could not determine input size. Using default: {input_size}")

    return checkpoint, model_name, input_size


def create_test_transform(input_size=256):
    """Create transform pipeline for testing"""
    return transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.143)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def test_model(model, test_loader, device, num_classes):
    """Test the model and collect predictions"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_paths = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs = inputs.to(device)

                with autocast():
                    outputs = model(inputs)
                    probabilities = torch.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                # Get file paths from the current batch
                batch_start_idx = batch_idx * test_loader.batch_size
                for i in range(len(targets)):
                    idx = batch_start_idx + i
                    if idx < len(test_loader.dataset.samples):
                        all_paths.append(test_loader.dataset.samples[idx][0])

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                # Skip this batch but continue testing
                continue

    return (
        np.array(all_predictions),
        np.array(all_targets),
        np.array(all_probabilities),
        all_paths,
    )


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, y_probs, class_names, output_path):
    """Plot ROC curves for each class"""
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    plt.figure(figsize=(10, 8))

    for i in range(len(class_names)):
        if np.sum(y_true_bin[:, i]) > 0:  # Only plot if class has samples
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves for All Classes", fontsize=16)
    plt.legend(loc="lower right", bbox_to_anchor=(1.3, 0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_gradcam_heatmap(model, input_tensor, pred_class, device):
    """Generate Grad-CAM heatmap for visualization"""
    model.eval()

    # Register hooks
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # For Swin Transformer, we'll use the last stage
    target_layer = model.layers[-1]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    model.zero_grad()
    output = model(input_tensor)

    # Backward pass
    output[0, pred_class].backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Process gradients and activations
    if len(gradients) > 0 and len(activations) > 0:
        grad = gradients[0].cpu().data.numpy()[0]
        act = activations[0].cpu().data.numpy()[0]

        # Global average pooling
        weights = np.mean(grad, axis=(1, 2))

        # Weighted combination
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]

        # Apply ReLU
        cam = np.maximum(cam, 0)

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam
    else:
        return None


def save_gradcam_samples(
    model,
    test_dataset,
    predictions,
    targets,
    class_names,
    output_dir,
    device,
    num_samples=20,
    input_size=256,
):
    """Save sample predictions with Grad-CAM visualizations"""
    gradcam_dir = os.path.join(output_dir, "gradcam_samples")
    os.makedirs(gradcam_dir, exist_ok=True)

    # Get indices for correct and incorrect predictions
    correct_mask = predictions == targets
    incorrect_mask = ~correct_mask

    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]

    # Sample indices
    n_correct = min(len(correct_indices), num_samples // 2)
    n_incorrect = min(len(incorrect_indices), num_samples // 2)

    if n_correct > 0:
        correct_samples = np.random.choice(correct_indices, n_correct, replace=False)
    else:
        correct_samples = []

    if n_incorrect > 0:
        incorrect_samples = np.random.choice(
            incorrect_indices, n_incorrect, replace=False
        )
    else:
        incorrect_samples = []

    sample_indices = list(correct_samples) + list(incorrect_samples)

    # Transform for model input
    transform = create_test_transform(input_size)

    for idx in sample_indices:
        try:
            # Load image
            img_path, _ = test_dataset.samples[idx]
            original_img = Image.open(img_path).convert("RGB")

            # Prepare for model
            input_tensor = transform(original_img).unsqueeze(0).to(device)

            # Generate Grad-CAM
            pred_class = predictions[idx]
            heatmap = generate_gradcam_heatmap(model, input_tensor, pred_class, device)

            if heatmap is not None:
                # Resize heatmap to match original image size
                heatmap_resized = cv2.resize(
                    heatmap, (original_img.width, original_img.height)
                )

                # Create colored heatmap
                heatmap_colored = cv2.applyColorMap(
                    np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
                )
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

                # Overlay on original image
                original_array = np.array(original_img)
                overlay = cv2.addWeighted(original_array, 0.7, heatmap_colored, 0.3, 0)

                # Save visualization
                status = "correct" if correct_mask[idx] else "incorrect"
                true_class = class_names[targets[idx]]
                pred_class_name = class_names[pred_class]

                filename = (
                    f"{status}_true_{true_class}_pred_{pred_class_name}_idx{idx}.jpg"
                )
                save_path = os.path.join(gradcam_dir, filename)

                # Create figure with original and grad-cam
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                ax1.imshow(original_img)
                ax1.set_title(f"Original Image\nTrue: {true_class}")
                ax1.axis("off")

                ax2.imshow(overlay)
                ax2.set_title(f"Grad-CAM\nPredicted: {pred_class_name}")
                ax2.axis("off")

                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()

                print(f"Saved Grad-CAM for sample {idx}")

        except Exception as e:
            print(f"Error generating Grad-CAM for sample {idx}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Test Swin-V2 for Fundus Image Classification"
    )
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Path to test dataset directory"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Output directory for results",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--save_gradcam", action="store_true", help="Save Grad-CAM visualizations"
    )
    parser.add_argument(
        "--gradcam_samples",
        type=int,
        default=20,
        help="Number of Grad-CAM samples to save",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint and detect configuration
    print("Loading model checkpoint...")
    checkpoint, model_name, input_size = detect_model_config(args.model_path)

    print(f"Detected configuration:")
    print(f"  - Model: {model_name}")
    print(f"  - Input size: {input_size}x{input_size}")

    # Get class names
    if "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
    else:
        # Try to load from the model directory
        model_dir = os.path.dirname(args.model_path)
        class_names_path = os.path.join(model_dir, "class_names.json")
        if os.path.exists(class_names_path):
            with open(class_names_path, "r") as f:
                class_names = json.load(f)
        else:
            # Get from test dataset
            test_dataset_temp = datasets.ImageFolder(args.test_dir)
            class_names = test_dataset_temp.classes
            print(
                "Warning: Class names not found in checkpoint. Using test dataset classes."
            )

    num_classes = len(class_names)

    # Initialize model
    print(f"Initializing {model_name} model with {num_classes} classes...")
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative approach...")
        # Try with default model if specific model fails
        model = timm.create_model(
            "swinv2_base_window16_256", pretrained=False, num_classes=num_classes
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model = model.to(device)
        model.eval()
        input_size = 256
        print(f"Loaded model with default configuration (input size: {input_size})")

    # Create test dataset and loader
    print(f"\nPreparing test dataset with input size {input_size}x{input_size}...")
    test_transform = create_test_transform(input_size)
    test_dataset = datasets.ImageFolder(args.test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Testing on {len(test_dataset)} images across {num_classes} classes")

    # Test the model
    predictions, targets, probabilities, paths = test_model(
        model, test_loader, device, num_classes
    )

    if len(predictions) == 0:
        print(
            "Error: No predictions were generated. Please check your test dataset and model."
        )
        return

    # Calculate metrics
    accuracy = (predictions == targets).mean() * 100

    # Generate classification report
    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Generate confusion matrix
    cm = confusion_matrix(targets, predictions)

    # Save results
    results = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "num_test_samples": len(test_dataset),
        "num_predictions": len(predictions),
        "model_checkpoint": args.model_path,
        "model_name": model_name,
        "input_size": input_size,
        "args": vars(args),
    }

    # Save JSON results
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save detailed predictions
    predictions_df = pd.DataFrame(
        {
            "file_path": paths[: len(predictions)],  # Ensure same length
            "true_label": targets,
            "true_class": [class_names[t] for t in targets],
            "predicted_label": predictions,
            "predicted_class": [class_names[p] for p in predictions],
            "correct": predictions == targets,
            "confidence": probabilities.max(axis=1),
        }
    )

    # Add probability columns for each class
    for i, class_name in enumerate(class_names):
        predictions_df[f"prob_{class_name}"] = probabilities[:, i]

    predictions_df.to_csv(
        os.path.join(args.output_dir, "detailed_predictions.csv"), index=False
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        cm, class_names, os.path.join(args.output_dir, "confusion_matrix.png")
    )

    # Plot ROC curves
    if num_classes > 2:  # Multi-class
        plot_roc_curves(
            targets,
            probabilities,
            class_names,
            os.path.join(args.output_dir, "roc_curves.png"),
        )

    # Generate per-class metrics plot
    plt.figure(figsize=(12, 6))

    # Extract per-class metrics
    precision = [report[class_name]["precision"] for class_name in class_names]
    recall = [report[class_name]["recall"] for class_name in class_names]
    f1_score = [report[class_name]["f1-score"] for class_name in class_names]
    support = [report[class_name]["support"] for class_name in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    plt.bar(x - width, precision, width, label="Precision", alpha=0.8)
    plt.bar(x, recall, width, label="Recall", alpha=0.8)
    plt.bar(x + width, f1_score, width, label="F1-Score", alpha=0.8)

    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Per-Class Performance Metrics", fontsize=16)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, "per_class_metrics.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save Grad-CAM visualizations if requested
    if args.save_gradcam:
        print("\nGenerating Grad-CAM visualizations...")
        save_gradcam_samples(
            model,
            test_dataset,
            predictions,
            targets,
            class_names,
            args.output_dir,
            device,
            args.gradcam_samples,
            input_size,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Input Size: {input_size}x{input_size}")
    print(f"Test Samples: {len(predictions)}/{len(test_dataset)}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f'Macro Avg Precision: {report["macro avg"]["precision"]:.3f}')
    print(f'Macro Avg Recall: {report["macro avg"]["recall"]:.3f}')
    print(f'Macro Avg F1-Score: {report["macro avg"]["f1-score"]:.3f}')
    print("\nPer-Class Results:")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        print(
            f"{class_name:20s} | Precision: {precision[i]:.3f} | Recall: {recall[i]:.3f} | "
            f"F1: {f1_score[i]:.3f} | Support: {support[i]}"
        )
    print("-" * 60)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
