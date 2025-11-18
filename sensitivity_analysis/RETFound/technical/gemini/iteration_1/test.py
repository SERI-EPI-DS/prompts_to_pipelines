# /project/code/test.py

import os
import sys
import argparse
import traceback
import time
from functools import partial  # Import partial for the model definition

# --- Path Setup ---
# This ensures the script can find the RETFound package and its utilities.
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
retfound_path = os.path.join(main_path, "RETFound")
if main_path not in sys.path:
    sys.path.insert(0, main_path)
if retfound_path not in sys.path:
    sys.path.insert(0, retfound_path)

# --- Core Imports ---
from RETFound import models_vit
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="RETFound Fine-Tuned Model Testing Script"
    )
    parser.add_argument(
        "--data_path", required=True, type=str, help="Path to the root data directory"
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        type=str,
        help="Path to save the output CSV file",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to the trained model weights (.pth file)",
    )

    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers", default=8, type=int, help="Number of data loading workers"
    )

    return parser


def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)

    # --- Data Preparation ---
    # NOTE: It's crucial to use the same validation transforms for testing.
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load test dataset
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "test"), transform=test_transform
    )

    # We need the class names to build the model and the output CSV.
    # A robust way is to infer this from the train folder structure.
    train_dataset_for_classes = datasets.ImageFolder(
        os.path.join(args.data_path, "train")
    )
    class_names = train_dataset_for_classes.classes
    num_classes = len(class_names)
    print(f"Testing on {num_classes} classes: {', '.join(class_names)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- ⭐ FIX: Build the ViT-Large model directly, matching the training script ---
    print(
        "✅ Constructing ViT-Large model directly from the VisionTransformer class..."
    )
    model = models_vit.VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        global_pool=True,
    )

    # Load the fine-tuned weights saved from the training script
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Successfully loaded fine-tuned model weights from {args.model_path}")

    # --- Inference Loop ---
    results = []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)

            # Get model outputs (logits)
            outputs = model(images)

            # Convert logits to probabilities
            scores = softmax(outputs)

            # Get the predicted class index
            _, predicted_indices = torch.max(outputs.data, 1)

            # Store results for each image in the batch
            for j in range(images.size(0)):
                image_index = i * args.batch_size + j
                file_path, _ = test_dataset.samples[image_index]
                filename = os.path.basename(file_path)

                result_entry = {"filename": filename}

                # Add score for each class
                for k, class_name in enumerate(class_names):
                    result_entry[f"score_{class_name}"] = scores[j, k].item()

                # Add final predicted class name
                result_entry["predicted_class"] = class_names[
                    predicted_indices[j].item()
                ]

                results.append(result_entry)

    # --- Save Results to CSV ---
    df = pd.DataFrame(results)

    # Reorder columns for clarity: filename, predicted_class, then all scores
    cols = ["filename", "predicted_class"] + [f"score_{cn}" for cn in class_names]
    df = df[cols]

    output_csv_path = os.path.join(args.results_dir, "test_predictions.csv")
    df.to_csv(output_csv_path, index=False)

    print(f"✅ Testing complete. Results saved to {output_csv_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
