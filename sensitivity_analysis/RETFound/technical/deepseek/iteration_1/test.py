import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import sys

# --- CRITICAL FIX: Add the path to the RETFound directory ---
sys.path.append(os.path.join(os.path.dirname(__file__), "../../RETFound"))

# Now these imports should work
import models_vit
from util.pos_embed import interpolate_pos_embed


def get_args_parser():
    parser = argparse.ArgumentParser("RETFound testing", add_help=False)
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU")
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--model",
        default="RETFound_mae",
        type=str,
        metavar="MODEL",  # Changed default
        help="Name of model to train",
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin_mem", action="store_true", help="Pin CPU memory in DataLoader"
    )
    parser.add_argument("--device", default="cuda", help="device to use for testing")

    # Path arguments
    parser.add_argument(
        "--data_path", default="../../data/test", type=str, help="test dataset path"
    )
    parser.add_argument(
        "--weights_path",
        default="../../project/results/checkpoint-best.pth",
        type=str,
        help="path to trained model weights",
    )
    parser.add_argument(
        "--output_dir",
        default="../../project/results",
        type=str,
        help="path where to save results",
    )

    return parser


def load_model(args, num_classes, device):
    """Load the model with trained weights"""
    # Build model - using the correct model name from your available models
    print(f"Loading model: {args.model}")

    # Get available models for debugging
    print("Available models in models_vit:")
    for key in models_vit.__dict__.keys():
        if not key.startswith("_"):
            print(f"  - {key}")

    # Use the correct model class name from your available models
    model_class_name = "RETFound_mae"  # Based on your previous error output
    if hasattr(models_vit, model_class_name):
        model = getattr(models_vit, model_class_name)(
            num_classes=num_classes,
            drop_path_rate=0.1,
            global_pool=True,
        )
    else:
        # Fallback to VisionTransformer
        print(
            f"Model '{model_class_name}' not found, falling back to VisionTransformer"
        )
        model = models_vit.__dict__["VisionTransformer"](
            num_classes=num_classes,
            drop_path_rate=0.1,
            global_pool=True,
        )

    # Load trained weights
    checkpoint = torch.load(args.weights_path, map_location="cpu")

    if "model" in checkpoint:
        checkpoint_model = checkpoint["model"]
    else:
        checkpoint_model = checkpoint

    # Remove head weights if shape doesn't match
    for k in ["head.weight", "head.bias"]:
        if (
            k in checkpoint_model
            and checkpoint_model[k].shape != model.state_dict()[k].shape
        ):
            print(f"Removing key {k} from checkpoint")
            del checkpoint_model[k]

    # Interpolate position embedding if needed
    interpolate_pos_embed(model, checkpoint_model)

    # Load weights
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Missing keys: {msg.missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")

    model.to(device)
    model.eval()

    return model


# ... rest of your test.py code remains the same (main function, calculate_metrics, etc.)


def main(args):
    device = torch.device(args.device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build transform
    transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset
    dataset_test = ImageFolder(args.data_path, transform=transform)

    # Get class names
    class_names = dataset_test.classes
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Create data loader
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        drop_last=False,
    )

    # Load model
    model = load_model(args, num_classes, device)

    # Test the model
    print("Starting testing...")
    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_filenames = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader_test):
            images = images.to(device, non_blocking=True)

            # Get predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.numpy())

            # Get filenames for this batch
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + len(images)
            batch_filenames = [
                dataset_test.samples[i][0]
                for i in range(start_idx, min(end_idx, len(dataset_test)))
            ]
            all_filenames.extend(batch_filenames)

            if batch_idx % 10 == 0:
                print(
                    f"Processed {batch_idx * args.batch_size}/{len(dataset_test)} images"
                )

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    all_filenames = np.array(all_filenames)

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "filename": all_filenames,
            "true_label": all_targets,
            "predicted_label": all_predictions,
            "true_class": [class_names[i] for i in all_targets],
            "predicted_class": [class_names[i] for i in all_predictions],
        }
    )

    # Add probability scores for each class
    for i, class_name in enumerate(class_names):
        results_df[f"prob_{class_name}"] = all_probabilities[:, i]

    # Save results to CSV
    results_csv_path = os.path.join(args.output_dir, "test_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    # Calculate and save metrics
    metrics_df = calculate_metrics(
        all_targets, all_predictions, all_probabilities, class_names
    )
    metrics_csv_path = os.path.join(args.output_dir, "test_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")

    # Print summary
    print("\n=== TEST RESULTS SUMMARY ===")
    accuracy = (all_predictions == all_targets).mean()
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))

    # For binary classification, calculate AUC
    if len(class_names) == 2:
        try:
            auc_score = roc_auc_score(all_targets, all_probabilities[:, 1])
            print(f"AUC Score: {auc_score:.4f}")
        except:
            print("Could not calculate AUC score")

    return results_df, metrics_df


def calculate_metrics(true_labels, predictions, probabilities, class_names):
    """Calculate comprehensive evaluation metrics"""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    # Basic metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None
    )

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average="macro"
    )

    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(true_labels, predictions, average="weighted")
    )

    # Create metrics dataframe
    metrics_data = []

    # Per-class metrics
    for i, class_name in enumerate(class_names):
        metrics_data.append(
            {
                "class": class_name,
                "precision": precision[i],
                "recall": recall[i],
                "f1_score": f1[i],
                "support": support[i],
                "type": "per_class",
            }
        )

    # Overall metrics
    metrics_data.extend(
        [
            {
                "class": "macro_avg",
                "precision": precision_macro,
                "recall": recall_macro,
                "f1_score": f1_macro,
                "support": len(true_labels),
                "type": "macro",
            },
            {
                "class": "weighted_avg",
                "precision": precision_weighted,
                "recall": recall_weighted,
                "f1_score": f1_weighted,
                "support": len(true_labels),
                "type": "weighted",
            },
            {
                "class": "accuracy",
                "precision": accuracy,
                "recall": accuracy,
                "f1_score": accuracy,
                "support": len(true_labels),
                "type": "overall",
            },
        ]
    )

    return pd.DataFrame(metrics_data)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    results_df, metrics_df = main(args)
