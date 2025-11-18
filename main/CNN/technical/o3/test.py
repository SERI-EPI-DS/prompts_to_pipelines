# -----------------------------------------------------------------------------
# test.py
# -----------------------------------------------------------------------------
#!/usr/bin/env python3
"""
Run inference on a held‑out test set using the fine‑tuned ConvNeXt‑Large model
and store predictions in a CSV file.

Example
-------
python test.py --data_dir /main/data --weights /main/project/results/best_model.pth --output_dir /main/project/results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_large

# Re‑use same ImageNet statistics as training
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Test ConvNeXt‑Large model on fundus photographs")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing the 'test' folder",
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to best_model.pth saved by train.py",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where predictions.csv will be written",
    )
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.weights, map_location=device)
    classes: list[str] = ckpt["classes"]
    image_size: int = ckpt.get("image_size", 512)

    # Model
    model = convnext_large()
    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features, len(classes)
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    # Transforms
    tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # Dataset / Loader
    test_root = Path(args.data_dir).expanduser().resolve() / "test"
    test_ds = datasets.ImageFolder(test_root, transform=tfms)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Inference loop
    rows: list[dict[str, str]] = []
    total_seen = 0
    for images, _ in test_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu()
        preds = probs.argmax(dim=1)

        batch_size = images.size(0)
        for i in range(batch_size):
            sample_path, _ = test_ds.samples[total_seen + i]
            row = {
                "filename": Path(sample_path).name,
                "prediction": classes[preds[i].item()],
            }
            for c, cls in enumerate(classes):
                row[f"prob_{cls}"] = f"{probs[i, c].item():.6f}"
            rows.append(row)
        total_seen += batch_size

    # Write CSV
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "predictions.csv"
    header = ["filename"] + [f"prob_{c}" for c in classes] + ["prediction"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Predictions for {len(rows)} images written to {csv_path}")


if __name__ == "__main__":
    main()
