import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from functools import partial

# Add the RETFound repository to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RETFound"))
)
# --- CORRECTED IMPORT: Use the base class for robustness ---
from models_vit import VisionTransformer


def get_args_parser():
    parser = argparse.ArgumentParser(description="RETFound Fine-tuned Model Testing")
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to the root data directory (containing train, val, test folders)",
    )
    parser.add_argument(
        "--results_path",
        required=True,
        type=str,
        help="Path to the directory where results (CSV file) will be saved",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to the fine-tuned model weights (.pth file)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    return parser


def main(args):
    # --- 1. Setup and Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.results_path, exist_ok=True)

    # --- 2. Data Preparation ---
    transform_test = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dir = os.path.join(args.data_path, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training directory not found at {train_dir}. It's needed to map class indices to names."
        )

    temp_train_dataset = datasets.ImageFolder(train_dir)
    class_names = temp_train_dataset.classes
    num_classes = len(class_names)
    print(f"Testing on {num_classes} classes: {class_names}")

    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "test"), transform=transform_test
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- 3. Model Loading ---
    # --- CORRECTED MODEL INSTANTIATION: Directly build the ViT-Large model ---
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
    )

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {args.model_path}")

    # --- 4. Inference and Evaluation ---
    results = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs = inputs.to(device)

            outputs = model(inputs)
            scores = softmax(outputs)
            _, predicted_indices = torch.max(outputs.data, 1)

            start_index = i * args.batch_size
            end_index = start_index + len(inputs)
            batch_filepaths = [
                s[0] for s in test_dataset.samples[start_index:end_index]
            ]

            for j in range(len(inputs)):
                filename = os.path.basename(batch_filepaths[j])
                predicted_class = class_names[predicted_indices[j].item()]

                result_item = {"filename": filename}
                for k, class_name in enumerate(class_names):
                    result_item[f"score_{class_name}"] = scores[j, k].item()

                result_item["predicted_class"] = predicted_class
                results.append(result_item)

    # --- 5. Save Results to CSV ---
    df = pd.DataFrame(results)

    cols = ["filename"] + [f"score_{cn}" for cn in class_names] + ["predicted_class"]
    df = df[cols]

    save_path = os.path.join(args.results_path, "test_results.csv")
    df.to_csv(save_path, index=False)

    print("\n--- Testing Finished ---")
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
