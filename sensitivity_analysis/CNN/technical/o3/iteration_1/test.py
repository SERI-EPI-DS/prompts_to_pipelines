"""test.py

Evaluate a fine‑tuned ConvNeXt‑Large model on the held‑out test set and
export predictions to CSV.

Usage:
    python test.py --data-dir /path/to/data --output-dir /path/to/project/results --weights /path/to/model_best.pth
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths"""

    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)  # (image, label)
        path, _ = self.samples[index]
        return original_tuple + (path,)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Test ConvNeXt‑L model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root folder containing test images (data_dir/test/...)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory to write CSV results"
    )
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to model weights (.pth)"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # try to load class mapping from meta.json if exists alongside weights
    meta_path = args.weights.parent / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            classes = meta["classes"]
            class_to_idx = {k: int(v) for k, v in meta["class_to_idx"].items()}
    else:
        # Fallback: derive from test dataset folder names (sorted)
        classes = sorted(
            [d.name for d in (args.data_dir / "test").iterdir() if d.is_dir()]
        )
        class_to_idx = {c: i for i, c in enumerate(classes)}

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    tfm = transforms.Compose(
        [
            transforms.Resize(426, antialias=True),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_ds = ImageFolderWithPaths(args.data_dir / "test", transform=tfm)
    # Ensure class order matches training
    test_ds.class_to_idx = class_to_idx
    test_ds.classes = classes

    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = models.convnext_large(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(classes))
    state = torch.load(args.weights, map_location="cpu")
    # Support two formats (state_dict or full checkpoint)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    csv_path = args.output_dir / "predictions.csv"
    header = ["filename"] + [f"prob_{cls}" for cls in classes] + ["prediction"]
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(header)

        for images, labels, paths in test_dl:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu()

            preds = probs.argmax(dim=1).tolist()
            probs_list = probs.tolist()

            for path, prob_vec, pred_idx in zip(paths, probs_list, preds):
                writer.writerow([Path(path).name] + prob_vec + [classes[pred_idx]])

    print(f"Saved predictions to {csv_path}")


if __name__ == "__main__":
    main()
