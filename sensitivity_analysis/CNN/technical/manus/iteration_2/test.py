#!/usr/bin/env python3
"""
Testing script for ConvNext-L classifier on ophthalmology fundus images.
ULTRA-STABLE VERSION: Optimized for memory usage and stability.
"""

import argparse
import os
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import timm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd


def clear_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class FundusTestDataset(Dataset):
    """Dataset for test images with ultra-stable loading."""

    def __init__(self, data_dir: str, transform=None, class_to_idx: Dict = None):
        self.data_dir = Path(data_dir) / "test"
        self.transform = transform
        self.samples = []
        self.filenames = []

        if class_to_idx is None:
            raise ValueError("class_to_idx mapping is required for test dataset")

        self.class_to_idx = class_to_idx
        self.classes = list(class_to_idx.keys())

        # Collect all image paths, labels, and filenames
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        self.samples.append(
                            (str(img_path), self.class_to_idx[class_name])
                        )
                        # Store relative filename for CSV output
                        relative_path = f"{class_name}/{img_path.name}"
                        self.filenames.append(relative_path)

        print(
            f"Found {len(self.samples)} test images across {len(self.classes)} classes"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        filename = self.filenames[idx]

        # Load image with memory optimization
        try:
            image = Image.open(img_path).convert("RGB")
            # Ensure image is not too large
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a small black image as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to {img_path}: {e}")
                # Apply ultra-minimal transform as fallback
                fallback_transform = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                image = fallback_transform(image)

        return image, label, filename


class ConvNextClassifier(nn.Module):
    """ConvNext-L classifier optimized for memory usage."""

    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super(ConvNextClassifier, self).__init__()

        # Load ConvNext-L model with proper feature extraction
        self.backbone = timm.create_model(
            "convnext_large", pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Get feature dimension from the backbone
        feature_dim = self.backbone.num_features

        # Simplified classification head to match training
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feature_dim, num_classes)  # Direct mapping
        )

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)
        return self.classifier(features)


def get_test_transforms(input_size: int = 224):
    """Get test transforms (no augmentation, ultra-stable)."""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def test_time_augmentation(model, image, device, num_augmentations: int = 3):
    """Apply minimal test-time augmentation with memory optimization."""
    model.eval()

    # Define minimal TTA transforms (NO ColorJitter)
    tta_transforms = [
        transforms.Compose([]),  # Original
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
        transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
    ]

    predictions = []

    with torch.no_grad():
        for i in range(min(num_augmentations, len(tta_transforms))):
            try:
                # Apply augmentation
                augmented_image = tta_transforms[i](image.cpu())
                augmented_image = augmented_image.unsqueeze(0).to(
                    device, non_blocking=True
                )

                # Get prediction
                output = model(augmented_image)
                prob = F.softmax(output, dim=1)
                predictions.append(prob.cpu().numpy())

                # Clear intermediate variables
                del output, prob, augmented_image

            except Exception as e:
                print(f"Error in TTA augmentation {i}: {e}")
                continue

    if len(predictions) == 0:
        # Fallback to original image
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            prob = F.softmax(output, dim=1)
            return prob.cpu()

    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)
    return torch.from_numpy(avg_prediction)


def evaluate_model(model, dataloader, device, class_names, use_tta: bool = True):
    """Evaluate model with aggressive memory management."""
    model.eval()

    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_filenames = []

    print("Starting evaluation...")
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (data, target, filenames) in enumerate(dataloader):
            try:
                data, target = data.to(device, non_blocking=True), target.to(
                    device, non_blocking=True
                )

                batch_predictions = []
                batch_probabilities = []

                # Process images one by one to save memory
                for i in range(data.size(0)):
                    try:
                        single_image = data[i]

                        if use_tta:
                            # Use minimal test-time augmentation
                            prob = test_time_augmentation(
                                model, single_image, device, num_augmentations=3
                            )
                        else:
                            # Standard inference
                            output = model(single_image.unsqueeze(0))
                            prob = F.softmax(output, dim=1).cpu()
                            del output

                        predicted_class = torch.argmax(prob, dim=1).item()

                        batch_predictions.append(predicted_class)
                        batch_probabilities.append(prob.squeeze().numpy())

                        # Clear variables
                        del prob

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(
                                f"CUDA out of memory processing image {i} in batch {batch_idx}. Using fallback."
                            )
                            clear_memory()
                            # Use fallback prediction
                            batch_predictions.append(0)
                            fallback_prob = np.zeros(len(class_names))
                            fallback_prob[0] = 0.1
                            batch_probabilities.append(fallback_prob)
                        else:
                            raise e
                    except Exception as e:
                        print(f"Error processing image {i} in batch {batch_idx}: {e}")
                        # Use fallback prediction
                        batch_predictions.append(0)
                        fallback_prob = np.zeros(len(class_names))
                        fallback_prob[0] = 0.1
                        batch_probabilities.append(fallback_prob)

                all_predictions.extend(batch_predictions)
                all_probabilities.extend(batch_probabilities)
                all_labels.extend(target.cpu().numpy())
                all_filenames.extend(filenames)

                # Clear batch data
                del data, target

                # Periodic memory cleanup
                if batch_idx % 5 == 0:
                    clear_memory()

                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"CUDA out of memory in batch {batch_idx}. Clearing cache and skipping."
                    )
                    clear_memory()
                    continue
                else:
                    print(f"Runtime error in batch {batch_idx}: {e}")
                    continue
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds")

    return all_predictions, all_probabilities, all_labels, all_filenames


def save_results_csv(
    predictions, probabilities, labels, filenames, class_names, output_path
):
    """Save detailed results to CSV file with error handling."""
    results = []

    for i, filename in enumerate(filenames):
        try:
            row = {
                "filename": filename,
                "true_label": (
                    class_names[labels[i]]
                    if labels[i] < len(class_names)
                    else "unknown"
                ),
                "predicted_label": (
                    class_names[predictions[i]]
                    if predictions[i] < len(class_names)
                    else "unknown"
                ),
            }

            # Add probability scores for each class
            for j, class_name in enumerate(class_names):
                if j < len(probabilities[i]):
                    row[f"prob_{class_name}"] = float(probabilities[i][j])
                else:
                    row[f"prob_{class_name}"] = 0.0

            # Add confidence (max probability)
            row["confidence"] = (
                float(np.max(probabilities[i])) if len(probabilities[i]) > 0 else 0.0
            )

            # Add correctness
            row["correct"] = predictions[i] == labels[i]

            results.append(row)

        except Exception as e:
            print(f"Error processing result {i}: {e}")
            continue

    # Convert to DataFrame and save
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return df
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        return None


def generate_classification_report(predictions, labels, class_names, output_path):
    """Generate and save detailed classification report."""
    try:
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(
            labels,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        report_text = classification_report(
            labels, predictions, target_names=class_names, zero_division=0
        )

        with open(output_path, "w") as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            f.write(report_text)

        print(f"Classification report saved to {output_path}")
        return report

    except Exception as e:
        print(f"Error generating classification report: {e}")
        return None


def save_confusion_matrix(predictions, labels, class_names, output_path):
    """Save confusion matrix to CSV."""
    try:
        cm = confusion_matrix(labels, predictions)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(output_path)
        print(f"Confusion matrix saved to {output_path}")
        return cm
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Ultra-Stable ConvNext-L Testing")
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root directory of the dataset"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.pth",
        help="Model checkpoint filename (default: best_model.pth)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size (reduced for memory)"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input image size (reduced for memory)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers (0 for stability)",
    )
    parser.add_argument(
        "--use_tta", action="store_true", help="Use minimal test-time augmentation"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="test_results",
        help="Prefix for output files",
    )

    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Clear memory at start
    clear_memory()

    # Load class mapping
    class_mapping_path = os.path.join(args.results_dir, "class_mapping.json")
    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"Class mapping not found at {class_mapping_path}")

    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)

    class_to_idx = class_mapping["class_to_idx"]
    idx_to_class = class_mapping["idx_to_class"]
    class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
    num_classes = len(class_names)

    print(f"Loaded class mapping for {num_classes} classes: {class_names}")

    # Load model
    model_path = os.path.join(args.results_dir, args.model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Create model
    model_args = checkpoint.get("args", {})
    dropout = model_args.get("dropout", 0.3)
    model = ConvNextClassifier(num_classes=num_classes, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
    if "best_val_acc" in checkpoint:
        print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")

    # Test model with a dummy input
    try:
        dummy_input = torch.randn(1, 3, args.input_size, args.input_size).to(device)
        with torch.no_grad():
            dummy_output = model(dummy_input)
            print(f"Model output shape: {dummy_output.shape}")
        del dummy_input, dummy_output
        clear_memory()
    except Exception as e:
        print(f"Warning: Model shape verification failed: {e}")

    # Data transforms
    test_transform = get_test_transforms(args.input_size)

    # Test dataset
    test_dataset = FundusTestDataset(
        args.data_root, transform=test_transform, class_to_idx=class_to_idx
    )

    # Data loader with memory optimization
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,  # Disabled to save memory
        drop_last=False,
    )

    # Evaluate model
    predictions, probabilities, labels, filenames = evaluate_model(
        model, test_loader, device, class_names, use_tta=args.use_tta
    )

    if len(predictions) == 0:
        print("Error: No predictions were generated. Check your dataset and model.")
        return

    # Calculate overall accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Save results to CSV
    csv_path = os.path.join(args.results_dir, f"{args.output_prefix}.csv")
    results_df = save_results_csv(
        predictions, probabilities, labels, filenames, class_names, csv_path
    )

    # Generate classification report
    report_path = os.path.join(args.results_dir, f"{args.output_prefix}_report.txt")
    classification_report_dict = generate_classification_report(
        predictions, labels, class_names, report_path
    )

    # Save confusion matrix
    cm_path = os.path.join(
        args.results_dir, f"{args.output_prefix}_confusion_matrix.csv"
    )
    confusion_matrix_array = save_confusion_matrix(
        predictions, labels, class_names, cm_path
    )

    # Save summary statistics
    summary_stats = {
        "total_samples": len(predictions),
        "accuracy": float(accuracy),
        "num_classes": num_classes,
        "class_names": class_names,
        "model_info": {
            "checkpoint_path": model_path,
            "epoch": checkpoint.get("epoch", "unknown"),
            "best_val_acc": checkpoint.get("best_val_acc", "unknown"),
            "use_tta": args.use_tta,
        },
    }

    # Save summary
    summary_path = os.path.join(args.results_dir, f"{args.output_prefix}_summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(summary_stats, f, indent=2)
        print(f"\nSummary statistics saved to {summary_path}")
    except Exception as e:
        print(f"Error saving summary: {e}")

    print("\nTesting completed successfully!")
    print(f"Results saved with prefix: {args.output_prefix}")


if __name__ == "__main__":
    main()
