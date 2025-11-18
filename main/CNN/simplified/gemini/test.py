import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test a trained ConvNext-L classifier."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root data directory (containing a 'test' folder).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file (.pth).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save the confusion matrix.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing."
    )
    return parser.parse_args()


def main():
    """Main function to test the classifier."""
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Define data transformations for the test set
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load the test dataset
    image_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), data_transform
    )
    dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    class_names = image_dataset.classes
    num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model structure
    model = models.convnext_large(weights=None)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    # Initialize metrics
    accuracy_metric = MulticlassAccuracy(device=device)
    precision_metric = MulticlassPrecision(
        device=device, average="macro", num_classes=num_classes
    )
    recall_metric = MulticlassRecall(
        device=device, average="macro", num_classes=num_classes
    )
    f1_metric = MulticlassF1Score(
        device=device, average="macro", num_classes=num_classes
    )

    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Testing")

    # Perform inference
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            accuracy_metric.update(outputs, labels)
            precision_metric.update(outputs, labels)
            recall_metric.update(outputs, labels)
            f1_metric.update(outputs, labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute and print metrics
    accuracy = accuracy_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1_score = f1_metric.compute().item()

    print("\n--- Test Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-Score: {f1_score:.4f}")

    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    print(
        f"\nConfusion matrix saved to {os.path.join(args.output_dir, 'confusion_matrix.png')}"
    )


if __name__ == "__main__":
    main()
