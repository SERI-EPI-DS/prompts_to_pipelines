#!/usr/bin/env python3
"""Runs inference with the fine-tuned ConvNeXt-L model (patched)."""

import os

os.environ.setdefault("TORCHVISION_DISABLE_ONNX_RUNTIME", "1")  # same workaround

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model on test/ and save CSV")
    p.add_argument("--data_dir", required=True, help="Root containing test/")
    p.add_argument("--model_path", required=True, help="Checkpoint (model_best.pth)")
    p.add_argument(
        "--results_dir", required=True, help="Where to write predictions.csv"
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def build_model(n_classes: int):
    try:
        model = models.convnext_large(weights=None)
    except AttributeError:
        model = models.convnext_large(pretrained=False)
    in_feat = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_feat, n_classes)
    return model


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_file = Path(args.model_path).with_suffix("").parent / "class_indices.json"
    with idx_file.open() as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    ds = datasets.ImageFolder(Path(args.data_dir) / "test", transform=tfm)
    ld = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(len(idx_to_class))
    ckpt = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.results_dir) / "predictions.csv"
    header = (
        ["filename"]
        + [idx_to_class[i] for i in range(len(idx_to_class))]
        + ["prediction"]
    )

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        ptr = 0
        for imgs, _ in ld:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu()

            bs = imgs.size(0)
            for j in range(bs):
                img_path, _ = ds.samples[ptr + j]
                fn = Path(img_path).name
                row_probs = [f"{probs[j, k]:.6f}" for k in range(len(idx_to_class))]
                pred = idx_to_class[int(probs[j].argmax())]
                writer.writerow([fn] + row_probs + [pred])
            ptr += bs

    print(f"Predictions written to {csv_path}")


if __name__ == "__main__":
    main()
