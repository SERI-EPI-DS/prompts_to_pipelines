# /project/code/test.py

import argparse
import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import csv


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image paths."""

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test a fine-tuned ConvNext-L classifier."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the dataset (e.g., /main/data)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save the output CSV (e.g., /project/results)",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to the saved model weights (.pth file)",
    )
    return parser.parse_args()


def main():
    """Main function to run the testing pipeline."""
    args = get_args()

    # --- 1. Setup ---
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class names from the training phase
    class_map_path = os.path.join(args.results_dir, "class_map.json")
    try:
        with open(class_map_path, "r") as f:
            class_names = json.load(f)
        num_classes = len(class_names)
        print(f"Loaded {num_classes} classes: {', '.join(class_names)}")
    except FileNotFoundError:
        print(f"Error: class_map.json not found in {args.results_dir}.")
        print("Please run train.py first to generate this file.")
        return

    # --- 2. Data Preparation ---
    print("Preparing test data...")
    IMG_SIZE = 224  # Should be the same as used in training
    BATCH_SIZE = 32  # Can be larger for inference

    test_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dir = os.path.join(args.data_dir, "test")

    test_dataset = ImageFolderWithPaths(root=test_dir, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- 3. Model Loading ---
    print(f"Loading model from {args.weights_path}...")
    # Initialize the model architecture
    model = models.convnext_large(weights=None)  # No pretrained weights needed
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)

    # Load the fine-tuned weights
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # --- 4. Inference Loop ---
    print("Running inference on the test set...")
    results = []

    with torch.no_grad():
        for inputs, _, paths in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)

            outputs = model(inputs)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # Get the top prediction
            _, preds = torch.max(probabilities, 1)

            # Move to CPU and convert to numpy for easier handling
            probabilities = probabilities.cpu().numpy()
            preds = preds.cpu().numpy()

            for i in range(len(paths)):
                filename = os.path.basename(paths[i])
                pred_class_name = class_names[preds[i]]
                scores = probabilities[i]

                result_row = {"filename": filename}
                for j, score in enumerate(scores):
                    result_row[f"{class_names[j]}_score"] = score
                result_row["predicted_class"] = pred_class_name

                results.append(result_row)

    # --- 5. Save Results to CSV ---
    output_csv_path = os.path.join(args.results_dir, "test_results.csv")
    print(f"Saving results to {output_csv_path}...")

    # Create a DataFrame and save it
    df = pd.DataFrame(results)

    # Define column order
    column_order = (
        ["filename"] + [f"{c}_score" for c in class_names] + ["predicted_class"]
    )
    df = df[column_order]

    df.to_csv(output_csv_path, index=False, float_format="%.6f")

    print("Testing complete. Results saved.")


if __name__ == "__main__":
    main()
