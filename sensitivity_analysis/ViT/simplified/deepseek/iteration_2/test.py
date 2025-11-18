# test_swin_ophthalmology_fixed.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import SwinForImageClassification
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import json


class OphthalmologyDataset:
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(
            [
                d
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                ):
                    self.samples.append(
                        (
                            os.path.join(class_dir, img_file),
                            self.class_to_idx[class_name],
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label, img_path

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            dummy_image = torch.randn(3, 224, 224)
            return dummy_image, label, img_path


def load_model(model_path, device):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)

    # Get number of classes from class mapping
    num_classes = len(checkpoint["class_mapping"])
    args = checkpoint.get("args", None)

    # Load model architecture
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-base-patch4-window7-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["class_mapping"], args


def test_model(model, test_loader, class_mapping, device, output_dir):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs.logits, dim=1)
            _, preds = torch.max(outputs.logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")

    # Per-class metrics
    class_report = classification_report(
        all_labels, all_preds, target_names=class_mapping.values(), output_dict=True
    )

    # Create results dictionary
    results = {
        "overall": {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        },
        "per_class": class_report,
    }

    # Create detailed predictions dataframe
    predictions_df = pd.DataFrame(
        {
            "image_path": all_paths,
            "true_label": [class_mapping[label] for label in all_labels],
            "predicted_label": [class_mapping[pred] for pred in all_preds],
            "confidence": [max(prob) for prob in all_probs],
            "correct": [pred == label for pred, label in zip(all_preds, all_labels)],
        }
    )

    # Add probability for each class
    for i, class_name in class_mapping.items():
        predictions_df[f"prob_{class_name}"] = [prob[i] for prob in all_probs]

    return results, predictions_df, all_preds, all_labels, all_probs


def plot_confusion_matrix(true_labels, pred_labels, class_mapping, output_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_mapping.values(),
        yticklabels=class_mapping.values(),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def main(args):
    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model, class_mapping, train_args = load_model(args.model_path, device)
    print(f"Model loaded. Number of classes: {len(class_mapping)}")
    print(f"Classes: {list(class_mapping.values())}")

    # Create transforms
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = OphthalmologyDataset(args.test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Run evaluation
    print("Running evaluation...")
    results, predictions_df, all_preds, all_labels, all_probs = test_model(
        model, test_loader, class_mapping, device, args.output_dir
    )

    # Generate plots
    print("Generating visualizations...")
    plot_confusion_matrix(all_labels, all_preds, class_mapping, args.output_dir)

    # Save results
    print("Saving results...")

    # Save metrics
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save predictions
    predictions_df.to_csv(
        os.path.join(args.output_dir, "all_predictions.csv"), index=False
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Accuracy: {results['overall']['accuracy']:.4f}")
    print(f"F1-Score: {results['overall']['f1_score']:.4f}")
    print(f"Precision: {results['overall']['precision']:.4f}")
    print(f"Recall: {results['overall']['recall']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Swin Transformer V2 for Ophthalmology"
    )

    # Required parameters
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Path to test data directory"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )

    # Optional parameters
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()

    main(args)
