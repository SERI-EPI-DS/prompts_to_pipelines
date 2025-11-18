#!/usr/bin/env python
"""Evaluate fine‑tuned RETFound ViT‑L model and export predictions CSV.

Example:
--------
python test.py \
    --data_root ../../data \
    --checkpoint ../../project/results/best.pth \
    --output_dir ../../project/results
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import timm
from torch.nn import functional as F

# ------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(description="Test RETFound ViT‑L classifier")
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder containing test/ sub‑folder",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    p.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save CSV results"
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def build_transforms(img_size):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    tf = transforms.Compose(
        [
            transforms.Resize(img_size + 32, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    return tf


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = build_transforms(args.img_size)
    test_set = datasets.ImageFolder(Path(args.data_root) / "test", transform=tf)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- load checkpoint & model ---
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_names = ckpt.get("classes", test_set.classes)
    num_classes = len(class_names)
    model = timm.create_model(
        "vit_large_patch16_224", pretrained=False, num_classes=num_classes
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()

    # --- inference ---
    results = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(images)
                probs = F.softmax(logits, dim=1)
            probs_cpu = probs.cpu()
            offset = len(results)
            for i in range(images.size(0)):
                path, _ = test_set.samples[offset + i]
                fname = Path(path).name
                prob_list = probs_cpu[i].tolist()
                pred_idx = int(torch.argmax(probs_cpu[i]))
                pred_label = class_names[pred_idx]
                row = [fname] + prob_list + [pred_label]
                results.append(row)

    # --- save CSV ---
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.output_dir) / "test_predictions.csv"
    header = ["filename"] + [f"prob_{c}" for c in class_names] + ["prediction"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

    # also save metrics summary (accuracy) if GT labels present
    # (only works if test set retains class sub‑folders)
    if all(label is not None for _, label in test_set.samples):
        correct = sum(
            1
            for row, (_, label_idx) in zip(results, test_set.samples)
            if class_names[label_idx] == row[-1]
        )
        acc = correct / len(test_set)
        metrics_file = Path(args.output_dir) / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({"accuracy": acc}, f, indent=2)
        print(f"Test accuracy: {acc:.4f}")
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
