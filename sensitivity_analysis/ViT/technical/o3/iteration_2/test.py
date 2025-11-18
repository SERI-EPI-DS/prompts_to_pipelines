# test.py
#!/usr/bin/env python3
"""
test.py - Evaluate a trained Swin‑V2‑B checkpoint on the test split and export CSV.

Example:
    python test.py \
        --data_dir /path/to/data \
        --checkpoint /path/to/project/results/best_model.pth \
        --output_csv /path/to/project/results/test_predictions.csv
"""

import argparse
import os
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import swin_v2_b


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Swin‑V2‑B checkpoint on test set"
    )
    parser.add_argument(
        "--data_dir", type=Path, required=True, help="Dataset root containing test/."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output_csv", type=Path, required=True, help="Path to write CSV predictions."
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # transforms must match validation transforms used during training
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_tfms = transforms.Compose(
        [
            transforms.Resize(288, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_ds = ImageFolder(args.data_dir / "test", transform=test_tfms)
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # load classes mapping (if available)
    classes_json = args.checkpoint.parent / "classes.json"
    if classes_json.exists():
        with open(classes_json) as f:
            classes = json.load(f)
    else:
        classes = test_ds.classes

    num_classes = len(classes)

    model = swin_v2_b(weights=None)
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features, num_classes)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename"] + [f"score_{c}" for c in classes] + ["predicted_class"]
        writer.writerow(header)

        sample_idx = 0
        for inputs, _ in test_dl:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().tolist()
            preds = torch.tensor(probs).argmax(dim=1).tolist()

            batch_size = len(probs)
            for i in range(batch_size):
                path, _ = test_ds.samples[sample_idx]
                filename = Path(path).name
                writer.writerow([filename] + probs[i] + [classes[preds[i]]])
                sample_idx += 1

    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
