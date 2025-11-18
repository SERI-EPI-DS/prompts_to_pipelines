#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a fine-tuned ConvNeXt-L classifier on the test set and export predictions.

This version avoids relying on `weights.meta["mean"]` since some TorchVision
builds omit it; instead it derives normalization stats from `weights.transforms()`
or falls back to ImageNet defaults.
"""
import argparse, csv, json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision.transforms import InterpolationMode


def get_norm_stats(weights):
    # 1) weights.meta
    try:
        meta = getattr(weights, "meta", None)
        if isinstance(meta, dict) and "mean" in meta and "std" in meta:
            return tuple(meta["mean"]), tuple(meta["std"])
    except Exception:
        pass
    # 2) parse from weights.transforms()
    try:
        t = weights.transforms()
        from torchvision.transforms import Normalize

        stack = [t]
        while stack:
            mod = stack.pop()
            if hasattr(mod, "transforms"):
                stack.extend(list(getattr(mod, "transforms")))
            if isinstance(mod, Normalize):
                mean = tuple(
                    float(x)
                    for x in (
                        mod.mean
                        if isinstance(mod.mean, (list, tuple))
                        else mod.mean.tolist()
                    )
                )
                std = tuple(
                    float(x)
                    for x in (
                        mod.std
                        if isinstance(mod.std, (list, tuple))
                        else mod.std.tolist()
                    )
                )
                return mean, std
    except Exception:
        pass
    # 3) defaults
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def create_model(num_classes: int, checkpoint_path: Path):
    model = convnext_large(weights=None)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    return model, ckpt


def build_eval_transform(img_size: int, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(
                int(img_size * 1.14), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def main():
    ap = argparse.ArgumentParser(
        description="Test ConvNeXt-L on a folder-structured dataset and export CSV"
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root of dataset with test subfolder",
    )
    ap.add_argument("--test_dir", type=str, default="test")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (best.pt). If omitted, tries <output_dir>/best.pt",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write predictions.csv",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--tta",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Number of TTA variants via flips (1,2,3,4)",
    )
    ap.add_argument(
        "--amp", action="store_true", help="Enable AMP for inference (helps on GPU)"
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir) if args.output_dir else Path("./runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (out_dir / "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "classes" in ckpt:
        classes = ckpt["classes"]
    else:
        cls_json = out_dir / "classes.json"
        if cls_json.exists():
            with open(cls_json, "r") as f:
                classes = json.load(f)["classes"]
        else:
            classes = datasets.ImageFolder(Path(args.data_dir) / args.test_dir).classes

    num_classes = len(classes)

    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
    mean, std = get_norm_stats(weights)
    eval_tfms = build_eval_transform(args.img_size, mean, std)

    test_path = Path(args.data_dir) / args.test_dir
    test_ds = datasets.ImageFolder(test_path, transform=eval_tfms)
    if test_ds.classes != classes:
        name_to_idx = {name: i for i, name in enumerate(classes)}
        remapped_samples = [
            (p, name_to_idx[test_ds.classes[t]]) for (p, t) in test_ds.samples
        ]
        test_ds.samples = remapped_samples
        test_ds.targets = [t for _, t in remapped_samples]
        test_ds.class_to_idx = name_to_idx
        test_ds.classes = list(classes)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model, _ = create_model(num_classes=num_classes, checkpoint_path=ckpt_path)
    model.to(device).eval()

    all_logits_cpu, all_labels_cpu = [], []
    all_files = [p for (p, _) in test_ds.samples]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            variants = [images]
            if args.tta >= 2:
                variants.append(torch.flip(images, dims=[-1]))
            if args.tta >= 3:
                variants.append(torch.flip(images, dims=[-2]))
            if args.tta >= 4:
                variants.append(torch.flip(images, dims=[-1, -2]))

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                logits_sum = None
                for v in variants:
                    out = model(v)
                    logits_sum = out if logits_sum is None else (logits_sum + out)
                logits = logits_sum / len(variants)

            all_logits_cpu.append(logits.cpu())
            all_labels_cpu.append(labels.cpu())

    logits = torch.cat(all_logits_cpu, dim=0)
    labels = torch.cat(all_labels_cpu, dim=0)
    probs = F.softmax(logits, dim=1)

    preds = probs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    overall_acc = correct / max(1, total)

    per_class_acc = {}
    for idx, cls in enumerate(classes):
        mask = labels == idx
        denom = mask.sum().item()
        per_class_acc[cls] = (
            (preds[mask] == labels[mask]).sum().item() / denom
            if denom > 0
            else float("nan")
        )

    csv_path = out_dir / "predictions.csv"
    header = ["file"] + [f"prob_{c}" for c in classes] + ["pred", "true"]
    with open(csv_path, "w", newline="") as f:
        import csv

        writer = csv.writer(f)
        writer.writerow(header)
        for i, path in enumerate(all_files):
            row = [str(path)]
            row += [f"{p:.6f}" for p in probs[i].tolist()]
            row += [classes[preds[i].item()], classes[labels[i].item()]]
            writer.writerow(row)

    report = {
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
        "num_samples": total,
        "classes": classes,
        "checkpoint": str(ckpt_path),
    }
    with open(out_dir / "test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Overall accuracy: {overall_acc:.4f}")
    print("Per-class accuracy:")
    for c, a in per_class_acc.items():
        print(f"  {c}: {a:.4f}")
    print(f"Wrote: {csv_path.resolve()} and test_report.json")


if __name__ == "__main__":
    main()
