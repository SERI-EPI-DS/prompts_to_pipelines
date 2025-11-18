#############################################
# test.py – evaluate checkpoint (supports variable img_size)
#############################################
"""Inference on an ImageFolder *test* set; outputs CSV with per‑class probs."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import Swin_V2_B_Weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_norm_stats():
    try:
        meta = Swin_V2_B_Weights.IMAGENET1K_V1.meta  # type: ignore[attr-defined]
        return meta.get("mean", IMAGENET_MEAN), meta.get("std", IMAGENET_STD)
    except AttributeError:
        return IMAGENET_MEAN, IMAGENET_STD


def build_loader(
    test_dir: Path, batch_size: int, img_size: int, num_workers: int
) -> DataLoader:
    mean, std = _get_norm_stats()
    tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.05)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    ds = datasets.ImageFolder(test_dir, transform=tfms)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main() -> None:
    p = argparse.ArgumentParser("Evaluate Swin‑V2‑B checkpoint")
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = build_loader(
        args.data_dir / "test", args.batch_size, args.img_size, args.num_workers
    )
    num_classes = len(test_loader.dataset.classes)

    model = models.swin_v2_b(weights=None)
    model.head = nn.Linear(model.head.in_features, num_classes)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "test_predictions.csv"

    softmax = nn.Softmax(dim=1)
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "filename",
                *[f"prob_{c}" for c in test_loader.dataset.classes],
                "predicted_class",
            ]
        )
        offset = 0
        with torch.no_grad():
            for imgs, _ in test_loader:
                bsz = imgs.size(0)
                imgs = imgs.to(device, non_blocking=True)
                probs = softmax(model(imgs)).cpu().numpy()
                preds = probs.argmax(1)
                for i in range(bsz):
                    fname = Path(test_loader.dataset.samples[offset + i][0]).name
                    writer.writerow(
                        [
                            fname,
                            *probs[i].tolist(),
                            test_loader.dataset.classes[preds[i]],
                        ]
                    )
                offset += bsz
    print(f"Predictions saved to {csv_path}")


if __name__ == "__main__":
    main()
