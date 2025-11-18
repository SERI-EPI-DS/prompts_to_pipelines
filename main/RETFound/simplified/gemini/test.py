import sys
import os

# --- NEW: Self-Aware Path Configuration ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)
    print(f"Project root added to path: {project_root}")
except Exception as e:
    print(f"Error adjusting Python path: {e}")
    project_root = os.path.abspath(".")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Fallback: Added current directory to path: {project_root}")
# --- END NEW BLOCK ---

import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# This import should now work reliably
from models_vit import vit_large_patch16


def get_args_parser():
    parser = argparse.ArgumentParser(description="RETFound Classifier Testing")
    parser.add_argument(
        "--data_path", required=True, type=str, help="Path to the root of your dataset"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to the fine-tuned model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of data loading workers"
    )
    return parser


# ... (The rest of the test.py script remains the same) ...


def main(args):
    # --- Setup ---
    print("Starting RETFound Classifier Testing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Preparation ---
    print(f"Loading test data from: {args.data_path}")
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "test"), transform=transform_test
    )
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    print(f"Found {num_classes} classes: {class_names}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Model Loading ---
    print(f"Loading model from: {args.model_path}")
    model = vit_large_patch16(num_classes=num_classes)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # --- Evaluation ---
    print("Running evaluation on the test set...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Results ---
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)

    print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(report)

    # Save the results to a file
    output_dir = os.path.dirname(args.model_path)
    if not output_dir:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
        f.write(f"Overall Test Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Test results saved to {os.path.join(output_dir, 'test_results.txt')}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
