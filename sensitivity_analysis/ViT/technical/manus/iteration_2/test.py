"""
Testing script for Swin-V2-B on a custom dataset.
"""

import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
from timm.data import create_transform
from torch.cuda.amp import autocast


def main():
    parser = argparse.ArgumentParser(description="Swin-V2-B Testing")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the root data directory"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to the results directory"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.pth",
        help="Path to the saved model weights",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Testing batch size")
    args = parser.parse_args()

    # --- Data Preparation ---
    test_dir = os.path.join(args.data_dir, "test")
    train_dir = os.path.join(args.data_dir, "train")  # Used to get class names

    # Get class names
    class_names = sorted([d.name for d in os.scandir(train_dir) if d.is_dir()])
    num_classes = len(class_names)

    # Create transforms
    test_transform = create_transform(input_size=256, is_training=False)

    # Create dataset
    test_dataset = ImageFolder(test_dir, transform=test_transform)

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- Model Preparation ---
    model = timm.create_model(
        "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        pretrained=False,  # Weights will be loaded
        num_classes=num_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weights_path = os.path.join(args.results_dir, args.model_path)
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()

    # --- Testing Loop ---
    results = []
    file_paths = [s[0] for s in test_dataset.samples]
    file_idx = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            batch_size = images.size(0)

            with autocast():
                outputs = model(images)
                scores = torch.softmax(outputs, dim=1)

            _, predicted_indices = torch.max(outputs.data, 1)

            for i in range(batch_size):
                image_path = file_paths[file_idx]
                pred_class = class_names[predicted_indices[i]]
                score_values = scores[i].cpu().numpy()

                result_row = {"filename": os.path.basename(image_path)}
                for j, class_name in enumerate(class_names):
                    result_row[f"{class_name}_score"] = score_values[j]
                result_row["predicted_class"] = pred_class
                results.append(result_row)
                file_idx += 1

    # --- Save Results ---
    df = pd.DataFrame(results)
    # Reorder columns
    cols = ["filename"] + [f"{cn}_score" for cn in class_names] + ["predicted_class"]
    df = df[cols]

    results_path = os.path.join(args.results_dir, "test_results.csv")
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
