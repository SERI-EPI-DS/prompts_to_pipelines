# /main/project/code/test.py

import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

import torch.nn as nn


def test_model(
    data_dir: str,
    results_dir: str,
    batch_size: int,
    weights_name: str = "best_model.pth",
):
    """
    Tests the fine-tuned Swin-V2-B model on the test set.

    Args:
        data_dir (str): Root directory of the dataset.
        results_dir (str): Directory where model weights are stored and results will be saved.
        batch_size (int): Number of images per batch for inference.
        weights_name (str): Name of the model weights file to load.
    """
    # 1. Setup and Configuration
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = os.path.join(results_dir, weights_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Please run train.py first."
        )

    # 2. Data Preparation
    # -------------------
    # Use the same transformations as validation, but for the test set
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dir = os.path.join(data_dir, "test")
    train_dir = os.path.join(data_dir, "train")  # Used to infer class names

    # Infer class names and number of classes from the train folder structure
    class_names = sorted([d.name for d in os.scandir(train_dir) if d.is_dir()])
    num_classes = len(class_names)
    print(f"Testing on {num_classes} classes: {', '.join(class_names)}")

    test_dataset = datasets.ImageFolder(test_dir, test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Loading
    # ----------------
    model = models.swin_v2_b(weights=None)  # No pre-trained weights needed here
    num_ftrs = model.head.in_features
    model.head = nn.Linear(
        num_ftrs, num_classes
    )  # Recreate the head to match saved weights

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # 4. Inference Loop
    # -----------------
    all_filenames = [os.path.basename(path) for path, _ in test_dataset.samples]
    all_scores = []
    all_preds = []

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)

            outputs = model(inputs)

            # Get probabilities using Softmax
            scores = torch.nn.functional.softmax(outputs, dim=1)
            all_scores.extend(scores.cpu().numpy())

            # Get predicted class index
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    # 5. Save Results to CSV
    # ----------------------
    predicted_class_names = [class_names[i] for i in all_preds]

    # Create a DataFrame for scores
    score_columns = [f"score_{cls}" for cls in class_names]
    scores_df = pd.DataFrame(all_scores, columns=score_columns)

    # Create the final results DataFrame
    results_df = pd.DataFrame(
        {"filename": all_filenames, "predicted_class": predicted_class_names}
    )

    # Concatenate filename/predictions with the scores
    final_df = pd.concat([results_df, scores_df], axis=1)

    # Save to CSV
    output_csv_path = os.path.join(results_dir, "test_results.csv")
    final_df.to_csv(output_csv_path, index=False)

    print(f"\nTesting complete. Results saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a fine-tuned Swin-V2-B classifier."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root data directory (e.g., /main/data).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory with model weights and for saving results (e.g., /main/project/results).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for testing."
    )

    args = parser.parse_args()

    test_model(
        data_dir=args.data_dir, results_dir=args.results_dir, batch_size=args.batch_size
    )

    # Example usage:
    # python test.py --data_dir ../../data --results_dir ../results
