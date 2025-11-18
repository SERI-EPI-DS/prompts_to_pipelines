# test.py (Updated)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
import argparse
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def test_model(data_dir, model_path, output_dir, batch_size=32):
    """
    Tests a trained Swin-V2-B model and generates evaluation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Class Names ---
    class_names_path = os.path.join(output_dir, "class_names.json")
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(
            f"Error: 'class_names.json' not found in {output_dir}. Please run training first."
        )

    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {', '.join(class_names)}")

    # --- 2. Data Preparation ---
    data_transform = transforms.Compose(
        [
            transforms.Resize(288),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), data_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Test set size: {len(test_dataset)}")

    # --- 3. Model Loading ---
    # **FIX:** Use the same corrected model name as in the training script.
    model_name = "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 4. Inference and Evaluation ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 5. Generate and Save Results ---
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Test Accuracy: {accuracy:.4f}\n")

    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print("Classification Report:")
    print(report)

    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall Test Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 10},
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained Swin-V2-B classifier.")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory."
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
        help="Directory where results will be saved.",
    )

    args = parser.parse_args()

    test_model(args.data_dir, args.model_path, args.output_dir)
