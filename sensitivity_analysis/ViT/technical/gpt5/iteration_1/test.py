#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

from tqdm import tqdm


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_eval_transforms(img_size: int, mean: List[float], std: List[float]):
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14), antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_model(num_classes: int) -> nn.Module:
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    model = swin_v2_b(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def resolve_imagenet_stats(weights_enum):
    try:
        meta = getattr(weights_enum, "meta", None)
        if isinstance(meta, dict) and "mean" in meta and "std" in meta:
            return list(meta["mean"]), list(meta["std"])
    except Exception:
        pass
    try:
        pipeline = weights_enum.transforms(antialias=True)
        tlist = getattr(pipeline, "transforms", None) or getattr(
            pipeline, "_transforms", None
        )
        if tlist:
            from torchvision.transforms import Normalize as NormalizeV1

            try:
                from torchvision.transforms.v2 import Normalize as NormalizeV2
            except Exception:
                NormalizeV2 = ()
            for t in tlist:
                if isinstance(t, (NormalizeV1, NormalizeV2)):

                    def _to_list(x):
                        if hasattr(x, "tolist"):
                            return [float(y) for y in x.tolist()]
                        return [float(y) for y in x]

                    return _to_list(t.mean), _to_list(t.std)
    except Exception:
        pass
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


@torch.no_grad()
def evaluate_and_write_csv(
    model, loader, device, out_csv_path: str, class_names: List[str], root_dir: str
):
    model.eval()
    softmax = nn.Softmax(dim=1)

    # CSV header: filename, prob_<class1>, prob_<class2>, ..., pred_class
    headers = ["filename"] + [f"prob_{c}" for c in class_names] + ["pred_class"]

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        n_correct = 0
        n_total = 0

        for images, targets in tqdm(loader, ncols=100, desc="Test"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = softmax(logits).cpu()

            # Write rows
            for i in range(images.size(0)):
                path, target = loader.dataset.samples[
                    n_total + i
                ]  # samples aligned with loader order
                rel_path = os.path.relpath(path, root_dir)
                prob_row = probs[i].tolist()
                pred_idx = int(torch.argmax(probs[i]).item())
                pred_name = class_names[pred_idx]
                writer.writerow(
                    [rel_path] + [f"{p:.6f}" for p in prob_row] + [pred_name]
                )

            # Accuracy (if test set is foldered by class, targets are valid)
            preds = torch.argmax(probs, dim=1)
            n_correct += (preds.cpu() == targets).sum().item()
            n_total += targets.numel()

    acc = n_correct / max(n_total, 1)
    return acc, n_total


def parse_args():
    p = argparse.ArgumentParser(
        description="Test Swin-V2-B classifier and write CSV outputs"
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing test subfolder.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write predictions CSV.",
    )
    p.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to checkpoint .pt (use best.pt from training).",
    )
    p.add_argument(
        "--use_ema",
        action="store_true",
        help="If the checkpoint contains EMA weights, use them.",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    # Load class mapping from training
    classes_path = os.path.join(args.output_dir, "classes.json")
    if not os.path.isfile(classes_path):
        # Try sibling: sometimes users point output_dir elsewhere
        alt = os.path.join(os.path.dirname(args.weights), "..", "classes.json")
        alt = os.path.abspath(alt)
        if os.path.isfile(alt):
            classes_path = alt
        else:
            raise FileNotFoundError(
                f"classes.json not found in {args.output_dir}. "
                f"Please copy the classes.json saved during training to the output directory."
            )
    classes_obj = load_json(classes_path)
    class_names: List[str] = classes_obj["classes"]
    num_classes = len(class_names)

    # Transforms
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    mean, std = resolve_imagenet_stats(Swin_V2_B_Weights.IMAGENET1K_V1)
    tfms = build_eval_transforms(args.img_size, mean, std)

    # Dataset / Loader
    test_dir = os.path.join(args.data_dir, "test")
    test_ds = ImageFolder(test_dir, transform=tfms)
    if test_ds.classes != class_names:
        print(
            "Warning: class order in test set differs from training. "
            "We'll still use the training class order for CSV column names."
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Model
    model = build_model(num_classes)
    device = torch.device(args.device)
    model.to(device)

    # Load weights
    if not args.weights or not os.path.isfile(args.weights):
        # Try default locations
        cand1 = os.path.join(args.output_dir, "best.pt")
        cand2 = os.path.join(args.output_dir, "checkpoints", "best.pt")
        if os.path.isfile(cand1):
            args.weights = cand1
        elif os.path.isfile(cand2):
            args.weights = cand2
        else:
            raise FileNotFoundError(
                "No weights provided and best.pt not found in output_dir."
            )

    ckpt = torch.load(args.weights, map_location="cpu")
    if args.use_ema and "ema" in ckpt:
        # Load EMA weights into model
        # Need compatible keys; we saved full state_dict for ema under same arch.
        model.load_state_dict(ckpt["ema"], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=False)

    # Predict and write CSV
    out_csv = os.path.join(args.output_dir, "test_predictions.csv")
    acc, n = evaluate_and_write_csv(
        model, test_loader, device, out_csv, class_names, root_dir=test_dir
    )

    if n > 0:
        print(f"Test images: {n} | Top-1 accuracy (foldered test): {acc*100:.2f}%")
        # Also save metrics.json
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump({"num_images": n, "top1_acc": acc}, f, indent=2)
    print(f"Saved predictions to: {out_csv}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
