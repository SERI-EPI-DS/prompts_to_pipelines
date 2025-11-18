# =============================
# file: test.py
# =============================
import argparse
import csv
import json
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_val_transform(image_size: int, mean: List[float], std: List[float]):
    return transforms.Compose(
        [
            transforms.Resize(
                int(image_size * 1.14), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_model(checkpoint_path: str, num_classes: int):
    # We keep the architecture identical to training time
    model = convnext_large(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    image_size = ckpt.get("image_size", 224)
    return model, ckpt.get("class_names", None), image_size


def main():
    parser = argparse.ArgumentParser(
        description="Test ConvNeXt-L classifier and export CSV"
    )
    parser.add_argument(
        "--data_root",
        required=True,
        type=str,
        help="Path to dataset root containing test folder",
    )
    parser.add_argument("--test_dir", default="test", type=str)
    parser.add_argument(
        "--output_dir", required=True, type=str, help="Directory to save CSV and logs"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to trained checkpoint (best.pt)",
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        help="If set, compute top-1 accuracy using folder labels",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load class names saved at train time
    saved_class_names_path = os.path.join(
        os.path.dirname(args.checkpoint), "class_names.json"
    )
    if os.path.isfile(saved_class_names_path):
        with open(saved_class_names_path, "r") as f:
            class_names = json.load(f)
    else:
        class_names = None

    # Build model
    # Temporarily set num_classes based on saved names if available; otherwise infer from dataset below
    tmp_num_classes = len(class_names) if class_names is not None else None
    # Load model to get image_size and validate class_names
    # We'll finalize num_classes after loading dataset if needed
    # Load checkpoint (will set classifier to tmp_num_classes if available; else fall back after dataset build)

    # Dataset (uses normalization from ImageNet)
    image_size_fallback = 224
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    # Build dataset first to know num_classes if needed
    test_path = os.path.join(args.data_root, args.test_dir)
    # We will create a temporary transform and may rebuild later if checkpoint suggests different image_size
    tmp_transform = build_val_transform(image_size_fallback, mean, std)
    test_set = datasets.ImageFolder(test_path, transform=tmp_transform)

    if tmp_num_classes is None:
        tmp_num_classes = len(test_set.classes)

    # Now load model & checkpoint
    model, ckpt_class_names, ckpt_image_size = load_model(
        args.checkpoint, tmp_num_classes
    )

    # Finalize class_names order (the order the model output uses)
    if ckpt_class_names is not None:
        class_names = ckpt_class_names
    elif class_names is None:
        class_names = test_set.classes  # best effort

    # Rebuild transform to match training image_size if present
    image_size = ckpt_image_size if ckpt_image_size is not None else image_size_fallback
    val_transform = build_val_transform(image_size, mean, std)
    test_set.transform = val_transform

    device = get_device()
    model.to(device)
    model.eval()

    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    # For metrics with potentially different class index orders, map by class name
    name_to_model_idx = {name: i for i, name in enumerate(class_names)}
    dataset_name_to_idx = test_set.class_to_idx  # mapping in the dataset

    all_rows = []
    correct = 0
    total = 0

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = softmax(logits).cpu()
            preds = torch.argmax(probs, dim=1)

            # Build rows
            for i in range(images.size(0)):
                # dataset.samples aligns with loader order via sampler indices
                # safer: we track filenames via loader.dataset.samples over the slice indices
                # However, DataLoader doesn't expose indices; instead access via underlying dataset
                # We'll reconstruct filenames from the batch index: not reliable.
                # Alternative: build an iterator over samples externally.
                pass

    # To correctly align filenames, iterate over dataset indices directly
    all_rows = []
    correct = 0
    total = 0

    for batch_start in range(0, len(test_set), args.batch_size):
        batch_indices = list(
            range(batch_start, min(batch_start + args.batch_size, len(test_set)))
        )
        batch_imgs = [test_set[i][0] for i in batch_indices]
        batch_targets = torch.tensor([test_set[i][1] for i in batch_indices])
        batch_paths = [test_set.samples[i][0] for i in batch_indices]

        batch_tensor = torch.stack(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = torch.argmax(probs, dim=1)

        for j, idx in enumerate(batch_indices):
            path = batch_paths[j]
            true_ds_idx = batch_targets[j].item()
            # map dataset idx to model idx via class name
            cls_name = test_set.classes[true_ds_idx]
            model_true_idx = name_to_model_idx.get(cls_name, true_ds_idx)

            row = {
                "filename": os.path.basename(path),
                "pred_index": int(preds[j].item()),
                "pred_label": class_names[int(preds[j].item())],
            }
            for k, cname in enumerate(class_names):
                row[f"prob_{cname}"] = float(probs[j, k].item())
            all_rows.append(row)

            if args.compute_metrics:
                if int(preds[j].item()) == int(model_true_idx):
                    correct += 1
                total += 1

    # Write CSV
    csv_path = os.path.join(args.output_dir, "test_results.csv")
    fieldnames = (
        ["filename"] + [f"prob_{c}" for c in class_names] + ["pred_index", "pred_label"]
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Saved results to {csv_path}")

    if args.compute_metrics and total > 0:
        acc = correct / total
        print(f"Test Top-1 Accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
