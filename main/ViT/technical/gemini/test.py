# /main/project/code/test.py

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import pandas as pd


def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test a fine-tuned Swin-V2-B classifier."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the dataset (e.g., ../../data)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save the output CSV (e.g., ../results)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model weights (.pth file)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing."
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Size to which images are resized."
    )
    return parser.parse_args()


def main():
    """Main function to run the testing process."""
    args = get_args()
    print("Starting testing script with the following arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")

    # --- 1. Setup and Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)

    # --- 2. Data Preparation ---
    # The test transform should be the same as the validation transform
    test_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size + 32),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    print("Initializing test dataset and dataloader...")
    # Use the 'train' folder to reliably get class names and their order.
    # The test set might not contain all classes.
    train_dataset_for_classes = datasets.ImageFolder(
        os.path.join(args.data_dir, "train")
    )
    class_names = train_dataset_for_classes.classes
    num_classes = len(class_names)
    print(f"Model configured for {num_classes} classes: {', '.join(class_names)}")

    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- 3. Model Loading ---
    print(f"Loading model from {args.model_path}")
    model = models.swin_v2_b(weights=None)  # Do not load pre-trained weights here
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)

    # Load the fine-tuned weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # --- 4. Inference Loop ---
    results = []
    softmax = nn.Softmax(dim=1)

    print("\n--- Running Inference on Test Set ---")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs = inputs.to(device)

            outputs = model(inputs)
            scores = softmax(outputs)
            preds = torch.argmax(scores, 1)

            # Get file paths for the current batch
            start_index = i * args.batch_size
            end_index = start_index + len(inputs)
            batch_filepaths = [
                os.path.basename(test_dataset.samples[j][0])
                for j in range(start_index, end_index)
            ]

            # Store results for each image in the batch
            for j in range(len(inputs)):
                result = {
                    "filename": batch_filepaths[j],
                    "predicted_class": class_names[preds[j].item()],
                }
                # Add scores for each class
                for k, class_name in enumerate(class_names):
                    result[f"score_{class_name}"] = scores[j, k].item()

                results.append(result)

    # --- 5. Save Results to CSV ---
    df = pd.DataFrame(results)

    # Reorder columns to be more readable
    column_order = ["filename", "predicted_class"] + [f"score_{c}" for c in class_names]
    df = df[column_order]

    output_csv_path = os.path.join(args.results_dir, "test_results.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"\nTesting complete. Results saved to {output_csv_path}")


if __name__ == "__main__":
    main()
