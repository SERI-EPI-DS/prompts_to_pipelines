#!/usr/bin/env python
"""test.py
Runs inference with the fine‑tuned Swin‑V2‑B model on a held‑out test set and
emits a CSV with per‑image class probabilities and predicted label.
Command‑line usage e.g.:
    python test.py --data_dir ../data --model_path ../project/results/best_model.pth --output_dir ../project/results
"""
import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------


def get_transforms(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return tf


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Swin‑V2‑B model on the test split and save results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", required=True, help="Path to dataset root (containing test)"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to best_model.pth from training"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save CSV results"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & loader
    test_ds = datasets.ImageFolder(Path(args.data_dir) / "test", get_transforms())
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Rebuild model with checkpointed weights
    checkpoint = torch.load(args.model_path, map_location="cpu")
    class_names = checkpoint["classes"]
    num_classes = len(class_names)

    model = models.swin_v2_b(weights=None)
    model.head = torch.nn.Linear(model.head.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Prepare CSV
    csv_path = Path(args.output_dir) / "test_results.csv"
    fieldnames = (
        ["filename"] + [f"prob_{cls}" for cls in class_names] + ["predicted_class"]
    )
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        global_idx = 0  # Tracks position within test_ds.samples
        for inputs, _ in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu()
            preds = torch.argmax(probs, dim=1)

            for j in range(inputs.size(0)):
                img_path, _ = test_ds.samples[global_idx]
                global_idx += 1
                row = {
                    "filename": Path(img_path).name,
                    **{
                        f"prob_{class_names[c]}": f"{probs[j, c].item():.6f}"
                        for c in range(num_classes)
                    },
                    "predicted_class": class_names[preds[j].item()],
                }
                writer.writerow(row)

    print(f"Saved test results ➜ {csv_path}")


if __name__ == "__main__":
    main()
