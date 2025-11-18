#!/usr/bin/env python3
"""
Evaluate a fine-tuned Swin-V2-B model on the test split and
write per-image predictions to a CSV.

Typical use:
python test.py \
    --data_dir /path/to/dataset \
    --model_path ./runs/exp1/best_model.pth \
    --csv_path ./runs/exp1/test_predictions.csv
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import Swin_V2_B_Weights as SwinWeights
import torch.nn.functional as F


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder that additionally returns the image file path."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        path = self.samples[index][0]
        return img, target, path


def parse_args():
    p = argparse.ArgumentParser(description="Test a fine-tuned Swin-V2-B.")
    p.add_argument(
        "--data_dir", required=True, help="Root folder containing test subfolder."
    )
    p.add_argument(
        "--model_path", required=True, help="Path to best_model.pth from training."
    )
    p.add_argument("--csv_path", required=True, help="Where to save CSV results.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def build_loader(test_dir: Path, batch_size: int, num_workers: int):
    tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            SwinWeights.IMAGENET1K_V1.transforms(),
        ]
    )
    ds = ImageFolderWithPaths(test_dir, transform=tfms)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dl, ds


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location="cpu")
    class_names: list[str] = checkpoint["classes"]

    model = models.swin_v2_b(weights=None)  # structure only
    model.head = torch.nn.Linear(model.head.in_features, len(class_names))
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device).eval()

    loader, ds = build_loader(
        Path(args.data_dir) / "test", args.batch_size, args.num_workers
    )

    rows = []
    with torch.no_grad():
        for imgs, labels, paths in loader:  # ← note the extra variable
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)

            for path, prob_row, pred_idx in zip(paths, probs, preds):
                rows.append(
                    {
                        "filename": Path(path).name,
                        **{
                            f"prob_{c}": float(prob_row[i])
                            for i, c in enumerate(class_names)
                        },
                        "prediction": class_names[pred_idx],
                    }
                )

    header = ["filename"] + [f"prob_{c}" for c in class_names] + ["prediction"]
    with open(args.csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} predictions → {args.csv_path}")


if __name__ == "__main__":
    main()
