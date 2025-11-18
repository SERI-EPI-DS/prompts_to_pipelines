import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser("Swin-V2-B Testing", add_help=False)
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Path to the root directory of your dataset (containing test folder)",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to the trained model (.pth file)",
    )
    parser.add_argument(
        "--output_dir", required=True, type=str, help="Path to save the test results"
    )
    # --- CORRECTED MODEL NAME ---
    parser.add_argument(
        "--model_name",
        default="swinv2_base_window12_192.ms_in22k",
        type=str,
        help="Name of the Swin-V2 model from timm (must match training)",
    )
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    return parser


def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Using model architecture: {args.model_name}")

    # --- Data Loading and Transformations ---
    test_dir = os.path.join(args.data_dir, "test")

    # Get model-specific transforms (must be same as validation)
    # This line also needs the corrected model name to build the architecture
    model_for_transforms = timm.create_model(args.model_name, pretrained=False)
    data_config = timm.data.resolve_model_data_config(model_for_transforms)
    test_transform = timm.data.create_transform(**data_config, is_training=False)

    test_dataset = ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    print(f"Testing on {num_classes} classes: {class_names}")

    # --- Model Loading ---
    model = timm.create_model(
        args.model_name, pretrained=False, num_classes=num_classes
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Testing Loop ---
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # --- Evaluation ---
    accuracy = accuracy_score(test_labels, test_preds)
    report = classification_report(
        test_labels,
        test_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    conf_matrix = confusion_matrix(test_labels, test_preds)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            test_labels, test_preds, target_names=class_names, zero_division=0
        )
    )

    # --- Save Results ---
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(args.output_dir, "classification_report.csv"))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
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
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Swin-V2-B Testing", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
