# /main/project/code/test.py

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F


def main(args):
    """
    Main function to execute the testing process.
    """
    print("PyTorch Version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Data Preparation ---
    # The test transformations should be the same as the validation transformations
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Create test dataset
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), data_transform
    )

    # Create test dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Testing on {len(test_dataset)} images belonging to {num_classes} classes.")

    # --- 2. Model Loading ---
    # Initialize the model architecture
    model = models.convnext_large(
        weights=None
    )  # We don't need pretrained weights, we will load our own
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)

    # Load the fine-tuned weights
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    # --- 3. Inference and Results Collection ---
    results = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs = inputs.to(device)

            outputs = model(inputs)

            # Get probabilities using softmax
            probabilities = F.softmax(outputs, dim=1)

            # Get final predictions
            _, preds = torch.max(outputs, 1)

            # Get file paths for the current batch
            batch_size = inputs.size(0)
            start_index = i * test_loader.batch_size
            end_index = start_index + batch_size
            batch_files = [
                os.path.basename(path)
                for path, _ in test_dataset.samples[start_index:end_index]
            ]

            for j in range(batch_size):
                result_row = {"filename": batch_files[j]}

                # Add scores for each class
                for k, class_name in enumerate(class_names):
                    result_row[f"score_{class_name}"] = probabilities[j, k].item()

                result_row["predicted_class"] = class_names[preds[j]]
                results.append(result_row)

    # --- 4. Save Results to CSV ---
    results_df = pd.DataFrame(results)

    os.makedirs(args.results_dir, exist_ok=True)
    csv_save_path = os.path.join(args.results_dir, "test_results.csv")
    results_df.to_csv(csv_save_path, index=False)

    print(f"Testing complete. Results saved to {csv_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a trained ConvNext-L classifier."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root data directory (containing test folder).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model weights (.pth file).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where the output CSV will be saved.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing."
    )

    args = parser.parse_args()
    main(args)
