#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import convnext_large


def make_eval_transform(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(
                int(img_size * 1.14), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_model(num_classes: int):
    model = convnext_large(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def parse_args():
    ap = argparse.ArgumentParser(description="Test ConvNeXt-L on held-out test set")
    ap.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root folder containing test subfolder",
    )
    ap.add_argument(
        "--test-dir",
        type=str,
        default="test",
        help="Test subfolder name (under data-root)",
    )
    ap.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Folder with checkpoints and where to write CSV",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to weights (default: <results>/best.pt)",
    )
    ap.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for inference"
    )
    ap.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    ap.add_argument(
        "--img-size", type=int, default=224, help="Input image size (must match train)"
    )
    ap.add_argument(
        "--save-csv",
        type=str,
        default="predictions.csv",
        help="CSV filename in results-dir",
    )
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    weights_path = args.weights or os.path.join(args.results_dir, "best.pt")
    classes_path = os.path.join(args.results_dir, "classes.json")
    assert os.path.isfile(weights_path), f"Missing weights: {weights_path}"
    assert os.path.isfile(classes_path), f"Missing classes file: {classes_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is required per the task constraints."

    # Load classes (ordering must match training)
    with open(classes_path, "r") as f:
        meta = json.load(f)
    classes = meta["classes"]
    class_to_idx_train = meta["class_to_idx"]

    # Build dataset/loader
    test_path = os.path.join(args.data_root, args.test_dir)
    eval_tf = make_eval_transform(args.img_size)
    test_ds = datasets.ImageFolder(test_path, transform=eval_tf)
    # Sanity check: class ordering alignment
    if test_ds.classes != classes:
        # Try to reorder using saved mapping
        # ImageFolder assigns classes sorted alphabetically; we'll enforce
        # a remap from current indices to the training indices, purely for checking.
        raise ValueError(
            "Test set classes do not match training classes ordering.\n"
            f"Train classes: {classes}\n"
            f"Test  classes: {test_ds.classes}\n"
            "Please ensure identical class subfolder names and ordering."
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    # Build model and load weights
    num_classes = len(classes)
    model = build_model(num_classes).to(device)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    softmax = nn.Softmax(dim=1)

    # Prepare CSV
    csv_path = os.path.join(args.results_dir, args.save_csv)
    fieldnames = (
        ["filename"] + [f"score_{c}" for c in classes] + ["pred_index", "pred_class"]
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = softmax(logits).detach().cpu()

            batch_paths = (
                [
                    test_ds.samples[i][0]
                    for i in range(
                        writer.line_num - 1, writer.line_num - 1 + images.size(0)
                    )
                ]
                if False
                else None
            )  # placeholder to show we considered it

            # safer: recompute filenames from the current batch indices via DataLoader dataset access
            # DataLoader doesn't expose indices directly; instead, use the underlying sampler order:
            # We'll simply pull paths from the DataSet by maintaining a cursor.
            # Implement a small iterator hack:
            pass  # replaced below

    # The above "writer.line_num" trick is unreliable; let's re-iterate properly keeping paths.


if __name__ == "__main__":
    # Re-implement main with a tiny path-tracking wrapper to keep code clean above.
    import argparse as _argparse
    import os as _os
    import json as _json
    import csv as _csv
    import torch as _torch
    import torch.nn as _nn
    from torchvision import datasets as _datasets

    def _parse_args():
        ap = _argparse.ArgumentParser(
            description="Test ConvNeXt-L on held-out test set"
        )
        ap.add_argument("--data-root", type=str, required=True)
        ap.add_argument("--test-dir", type=str, default="test")
        ap.add_argument("--results-dir", type=str, required=True)
        ap.add_argument("--weights", type=str, default="")
        ap.add_argument("--batch-size", type=int, default=64)
        ap.add_argument("--num-workers", type=int, default=8)
        ap.add_argument("--img-size", type=int, default=224)
        ap.add_argument("--save-csv", type=str, default="predictions.csv")
        return ap.parse_args()

    args = _parse_args()
    _os.makedirs(args.results_dir, exist_ok=True)
    weights_path = args.weights or _os.path.join(args.results_dir, "best.pt")
    classes_path = _os.path.join(args.results_dir, "classes.json")
    assert _os.path.isfile(weights_path), f"Missing weights: {weights_path}"
    assert _os.path.isfile(classes_path), f"Missing classes file: {classes_path}"

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is required per the task constraints."

    with open(classes_path, "r") as f:
        meta = _json.load(f)
    classes = meta["classes"]

    # Build eval transform (reuse from function)
    def _eval_tf(img_size: int = 224):
        from torchvision import transforms as _transforms
        from torchvision.transforms import InterpolationMode as _IM

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return _transforms.Compose(
            [
                _transforms.Lambda(lambda img: img.convert("RGB")),
                _transforms.Resize(int(img_size * 1.14), interpolation=_IM.BICUBIC),
                _transforms.CenterCrop(img_size),
                _transforms.ToTensor(),
                _transforms.Normalize(mean=mean, std=std),
            ]
        )

    test_path = _os.path.join(args.data_root, args.test_dir)
    test_ds = _datasets.ImageFolder(test_path, transform=_eval_tf(args.img_size))
    if test_ds.classes != classes:
        raise ValueError(
            "Test set classes do not match training classes ordering.\n"
            f"Train classes: {classes}\n"
            f"Test  classes: {test_ds.classes}\n"
            "Please ensure identical class subfolder names and ordering."
        )

    test_loader = _torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Model
    from torchvision.models import convnext_large as _convnext_large

    m = _convnext_large(weights=None)
    in_features = m.classifier[2].in_features
    m.classifier[2] = _nn.Linear(in_features, len(classes))
    ckpt = _torch.load(weights_path, map_location="cpu")
    m.load_state_dict(ckpt["model"])
    m.to(device).eval()

    softmax = _nn.Softmax(dim=1)

    csv_path = _os.path.join(
        args.results_dir, args.save_cvv if hasattr(args, "save_cvv") else args.save_csv
    )
    fieldnames = (
        ["filename"] + [f"score_{c}" for c in classes] + ["pred_index", "pred_class"]
    )

    # We need the exact filenames in loader order; enumerate the dataset indices exactly as the sampler will.
    # Since shuffle=False and default SequentialSampler, order is range(len(dataset)).
    dataset_paths = [s[0] for s in test_ds.samples]
    cursor = 0

    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with _torch.no_grad():
            for images, _targets in test_loader:
                bs = images.size(0)
                images = images.to(device, non_blocking=True)
                logits = m(images)
                probs = softmax(logits).detach().cpu()  # [B, C]

                batch_paths = dataset_paths[cursor : cursor + bs]
                cursor += bs

                for pth, prob in zip(batch_paths, probs):
                    prob_list = prob.tolist()
                    pred_idx = int(prob.argmax().item())
                    pred_cls = classes[pred_idx]
                    row = {
                        "filename": _os.path.basename(pth),
                        "pred_index": pred_idx,
                        "pred_class": pred_cls,
                    }
                    for c_name, p in zip(classes, prob_list):
                        row[f"score_{c_name}"] = f"{p:.6f}"
                    writer.writerow(row)

    print(f"Saved predictions to: {csv_path}")
