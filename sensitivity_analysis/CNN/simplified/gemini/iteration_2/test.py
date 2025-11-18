import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def test_model(args):
    """
    Main function to test the trained ConvNext-L model.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class names from the training output
    class_names_path = os.path.join(args.results_dir, "class_names.json")
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(
            f"Error: 'class_names.json' not found in {args.results_dir}. Please run training first."
        )

    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"Testing with {num_classes} classes: {', '.join(class_names)}")

    # Data transformations for testing
    test_transform = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load test dataset
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Initialize model architecture
    model = timm.create_model(
        "convnext_large",
        pretrained=False,  # We are loading our own weights
        num_classes=num_classes,
    )

    # Load the trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Evaluation loop
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Generate and Save Results ---

    # Overall Accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%\n")

    # Classification Report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print("Classification Report:")
    print(report)

    # Save classification report to a file
    report_path = os.path.join(args.results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
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

    # Save confusion matrix image
    cm_path = os.path.join(args.results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a trained ConvNext-L classifier."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the test dataset.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file (e.g., best_model.pth).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory where results and class names are stored.",
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size used during training."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Testing batch size."
    )

    args = parser.parse_args()
    test_model(args)
