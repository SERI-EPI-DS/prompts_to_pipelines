import argparse
import logging
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(args):
    """
    Main function to run inference with the trained model on the test set.
    """
    # --- 1. Device Configuration ---
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. This script requires a GPU.")
        return
    device = torch.device("cuda")
    logging.info(f"Using device: {device}")

    # --- 2. Data Preparation ---
    # The transformations must exactly match the validation transforms from training.
    data_transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dir = os.path.join(args.data_dir, "test")
    train_dir = os.path.join(args.data_dir, "train")  # Needed for class names

    if not os.path.isdir(test_dir) or not os.path.isdir(train_dir):
        logging.error(
            "Test or train directory not found. Please check the data_dir path."
        )
        return

    # Infer class names from the training directory structure to ensure consistency.
    class_names = sorted([d.name for d in os.scandir(train_dir) if d.is_dir()])
    num_classes = len(class_names)
    logging.info(f"Loading model for {num_classes} classes: {class_names}")

    # --- 3. Dataset and DataLoader ---
    image_dataset = datasets.ImageFolder(test_dir, data_transform)
    dataloader = DataLoader(
        image_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- 4. Model Loading ---
    # Re-create the model architecture.
    model = models.convnext_large(weights=None)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)

    # Load the trained weights from the specified path.
    if not os.path.exists(args.model_weights_path):
        logging.error(f"Model weights file not found at: {args.model_weights_path}")
        return
    model.load_state_dict(torch.load(args.model_weights_path))
    model = model.to(device)
    model.eval()

    # --- 5. Inference and Results Collection ---
    results = []

    progress_bar = tqdm(dataloader, desc="Testing")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(progress_bar):
            inputs = inputs.to(device)

            # Use autocast for potential speed-up during inference.
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                # Apply softmax to convert logits to probabilities.
                scores = F.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            # Get the file paths for the images in the current batch.
            start_idx = i * args.batch_size
            batch_filepaths = [
                path
                for path, _ in image_dataset.samples[
                    start_idx : start_idx + len(inputs)
                ]
            ]

            for j in range(inputs.size(0)):
                filename = os.path.basename(batch_filepaths[j])
                predicted_class = class_names[preds[j]]

                # Store results in a structured format.
                result_row = {"filename": filename}
                for k, class_name in enumerate(class_names):
                    result_row[f"{class_name}_score"] = scores[j, k].item()
                result_row["predicted_class"] = predicted_class

                results.append(result_row)

    # --- 6. Save Results to CSV ---
    results_df = pd.DataFrame(results)

    # Reorder columns for clarity: filename, scores for each class, final prediction.
    cols = ["filename"] + [f"{cn}_score" for cn in class_names] + ["predicted_class"]
    results_df = results_df[cols]

    os.makedirs(args.results_dir, exist_ok=True)
    output_csv_path = os.path.join(args.results_dir, "test_results.csv")
    results_df.to_csv(output_csv_path, index=False)
    logging.info(f"✅ Test results successfully saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to test a trained ConvNext-L model."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the root data directory."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where the output CSV will be saved.",
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        required=True,
        help="Path to the trained model weights (.pth file), e.g., '/path/to/results/best_model_weights.pth'.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for testing."
    )

    args = parser.parse_args()
    main(args)
