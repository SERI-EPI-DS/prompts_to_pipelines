# /main/project/code/test.py

import argparse
import os
import torch
import torch.nn as nn  # <-- ADDED THIS LINE
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import torch.nn.functional as F


def main(args):
    """
    Main function to run the testing script.
    """
    print("Initializing testing...")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Dataset and Dataloader Preparation ---
    print("Preparing test dataset...")

    # Use the same transforms as validation (without augmentation)
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dataset_path = os.path.join(args.data_dir, "test")
    test_dataset = datasets.ImageFolder(test_dataset_path, test_transform)

    # Set shuffle=False to ensure filenames align with predictions
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    class_names = test_dataset.classes
    num_classes = len(class_names)

    print(f"Found {num_classes} classes for testing: {', '.join(class_names)}")
    print(f"Test set size: {len(test_dataset)}")

    # --- 2. Model Loading ---
    print("Loading fine-tuned Swin-V2-B model...")
    model = models.swin_v2_b(weights=None)  # No pretrained weights needed here

    # Reconfigure the head to match the number of classes
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)

    # Load the saved fine-tuned weights
    weights_path = os.path.join(args.results_dir, "best_model_weights.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found at {weights_path}. Please run train.py first."
        )

    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # --- 3. Inference Loop ---
    all_filenames = [os.path.basename(path) for path, _ in test_dataset.samples]
    all_scores = []
    all_preds = []

    print("\nRunning inference on the test set...")
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)

            outputs = model(inputs)

            # Get class probabilities using softmax
            scores = F.softmax(outputs, dim=1)

            # Get the final predicted class index
            _, preds = torch.max(outputs, 1)

            all_scores.extend(scores.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # --- 4. Save Results to CSV ---
    print("Saving results to CSV...")

    # Map prediction indices back to class names
    predicted_class_names = [class_names[i] for i in all_preds]

    # Create a DataFrame for scores with class names as columns
    score_columns = [f"score_{name}" for name in class_names]
    scores_df = pd.DataFrame(all_scores, columns=score_columns)

    # Create the final results DataFrame
    results_df = pd.DataFrame(
        {"filename": all_filenames, "predicted_class": predicted_class_names}
    )

    # Combine the two DataFrames
    final_df = pd.concat([results_df, scores_df], axis=1)

    # Ensure results directory exists
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    output_path = os.path.join(args.results_dir, "test_results.csv")
    final_df.to_csv(output_path, index=False)

    print(f"\nTesting complete. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a fine-tuned Swin-V2-B classifier"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root data directory (containing train/val/test folders)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where model weights are stored and results will be saved",
    )

    # Hyperparameters for testing
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )

    args = parser.parse_args()
    main(args)
