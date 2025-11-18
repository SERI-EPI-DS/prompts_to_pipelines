import argparse
import os
import sys
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from tqdm import tqdm
from functools import partial

# Add RETFound repository to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RETFound"))
)

# FIX: Import the base VisionTransformer class directly
from models_vit import VisionTransformer


def get_args_parser():
    parser = argparse.ArgumentParser(description="RETFound Testing")
    parser.add_argument(
        "--data_path", required=True, type=str, help="Path to the root data directory."
    )
    parser.add_argument(
        "--results_dir", required=True, type=str, help="Path to the results directory."
    )
    parser.add_argument(
        "--model_weights",
        default="best_model.pth",
        type=str,
        help="Name of the model weights file in the results directory.",
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size for testing."
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of data loading workers."
    )
    return parser


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available. Testing on CPU.")

    # --- Load Class Names ---
    class_names_path = os.path.join(args.results_dir, "class_names.json")
    try:
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find class_names.json in {args.results_dir}")
        sys.exit(1)

    num_classes = len(class_names)
    print(f"Testing with {num_classes} classes: {class_names}")

    # --- Data Preparation ---
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "test"), transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # --- Model Loading ---
    model_path = os.path.join(args.results_dir, args.model_weights)
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        sys.exit(1)

    # FIX: Manually define and instantiate the ViT-Large model
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

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Testing ---
    results = []
    test_pbar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_pbar):
            images = images.to(device)
            outputs = model(images)
            scores = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_indices = torch.max(scores, 1)

            for j in range(images.size(0)):
                image_index = i * test_loader.batch_size + j
                if image_index < len(test_dataset.imgs):
                    image_path, _ = test_dataset.imgs[image_index]
                    file_name = os.path.basename(image_path)

                    result = {"file_name": file_name}
                    for k, class_name in enumerate(class_names):
                        result[f"{class_name}_score"] = scores[j, k].item()

                    result["predicted_class"] = class_names[predicted_indices[j]]
                    results.append(result)

    # --- Save Results ---
    results_df = pd.DataFrame(results)
    output_csv_path = os.path.join(args.results_dir, "test_results.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"Test results saved to {output_csv_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
