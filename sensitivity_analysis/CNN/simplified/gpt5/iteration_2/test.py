#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models.convnext import convnext_large
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int):
    model = convnext_large(weights=None)
    if isinstance(model.classifier, nn.Sequential):
        replaced = False
        for i in reversed(range(len(model.classifier))):
            if isinstance(model.classifier[i], nn.Linear):
                in_features = model.classifier[i].in_features
                model.classifier[i] = nn.Linear(in_features, num_classes)
                replaced = True
                break
        if not replaced:
            raise RuntimeError(
                "Could not locate final Linear layer in model.classifier"
            )
    else:
        raise RuntimeError("Unexpected classifier type; update replacement logic.")
    return model


def get_eval_transform(img_size: int, mean, std):
    return T.Compose(
        [
            T.Resize(
                int(img_size * 1.15),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Test ConvNeXt-L checkpoint on held-out test set"
    )
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root containing test/ folder (with class subfolders)",
    )
    p.add_argument("--test_dir", type=str, default="test")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt saved by train.py (best.pt or last.pt)",
    )
    p.add_argument("--out_dir", type=str, default="./outputs/convnextL_eval")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "test_predictions.csv"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    num_classes = int(ckpt.get("num_classes"))
    idx_to_class: Dict[int, str] = ckpt.get("idx_to_class")
    if isinstance(idx_to_class, dict):
        # keys might be strings in json, ensure int keys
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    img_size = int(ckpt.get("img_size", 224))
    mean = tuple(ckpt.get("mean", IMAGENET_MEAN))
    std = tuple(ckpt.get("std", IMAGENET_STD))

    tfm = get_eval_transform(img_size, mean, std)

    test_path = Path(args.data_root) / args.test_dir
    ds = ImageFolder(str(test_path), transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build and load model
    model = build_model(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if len(unexpected) > 0:
        print(f"Warning: unexpected keys in state_dict: {unexpected}")
    if len(missing) > 0:
        print(f"Warning: missing keys in state_dict: {missing}")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    softmax = nn.Softmax(dim=1)

    # Write CSV header
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = (
            ["filename"]
            + [f"score_{c}" for c in class_names]
            + ["pred_class", "pred_index", "true_class", "true_index"]
        )
        writer.writerow(header)

        # Iterate and record
        sample_idx = 0
        for imgs, labels in loader:
            bsz = imgs.size(0)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs)
                probs = softmax(logits).detach().cpu()

            preds = probs.argmax(dim=1)
            paths_batch = [
                ds.samples[i][0] for i in range(sample_idx, sample_idx + bsz)
            ]
            sample_idx += bsz

            for path, pvec, pred_idx, true_idx in zip(
                paths_batch, probs, preds.cpu(), labels.cpu()
            ):
                row = (
                    [Path(path).name]
                    + [f"{float(s):.6f}" for s in pvec.tolist()]
                    + [
                        class_names[int(pred_idx)],
                        int(pred_idx),
                        class_names[int(true_idx)],
                        int(true_idx),
                    ]
                )
                writer.writerow(row)

    print(f"Saved predictions to: {out_csv}")

    # Also save a small JSON metadata file for reproducibility
    meta = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "test_dir": args.test_dir,
        "num_classes": num_classes,
        "class_names": class_names,
        "img_size": img_size,
        "mean": mean,
        "std": std,
    }
    with open(out_dir / "eval_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
