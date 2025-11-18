import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model  # Added missing import
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
    checkpoint = torch.load(args.model_path)

    # Create model
    model = create_model(
        "convnext_large", pretrained=False, num_classes=len(checkpoint["class_to_idx"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Data transform
    test_transform = transforms.Compose(
        [
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )

    # Create test dataset
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), transform=test_transform
    )

    # Map class indices to match training
    test_dataset.class_to_idx = checkpoint["class_to_idx"]

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Inference
    all_preds = []
    all_targets = []
    all_probs = []

    # Collect all filenames (fixed implementation)
    filenames = [os.path.basename(path) for path, _ in test_dataset.samples]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Generate metrics
    class_names = list(test_dataset.class_to_idx.keys())
    report = classification_report(
        all_targets, all_preds, target_names=class_names, output_dict=True
    )

    # Save results
    results_df = pd.DataFrame(
        {
            "filename": filenames,
            "true_label": [class_names[i] for i in all_targets],
            "predicted_label": [class_names[i] for i in all_preds],
            **{
                f"prob_{class_names[i]}": [p[i] for p in all_probs]
                for i in range(len(class_names))
            },
        }
    )
    results_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(
        os.path.join(args.output_dir, "confusion_matrix.png"), bbox_inches="tight"
    )

    # Print summary
    print(f"Test Accuracy: {report['accuracy']:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")

    # Save full report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(args.output_dir, "classification_report.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ConvNeXt-L Ophthalmology Classifier Testing"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to test data directory"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
