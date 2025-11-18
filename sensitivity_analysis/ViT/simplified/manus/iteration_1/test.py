"""
Evaluate a trained Swin Transformer V2 model on a test dataset.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description="Test a Swin-V2-B classifier.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the test dataset directory.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save evaluation results."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        help="Name of the timm model.",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Input image size for the model."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Testing batch size."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Data transformations
    data_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset and dataloader
    image_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), data_transform
    )
    dataloader = DataLoader(
        image_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    class_names = image_dataset.classes
    num_classes = len(class_names)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        args.model_name, pretrained=False, num_classes=num_classes
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Testing loop
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Evaluation metrics
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Save classification report
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
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
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))

    print(f"Evaluation results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
