# =====================
# test.py — evaluate model (robust to older TorchVision)
# =====================
"""
Example
-------
python test.py \
  --data_dir /path/to/dataset/test \
  --checkpoint ./outputs/best.pt \
  --class_map ./outputs/class_indices.json \
  --output_csv ./outputs/predictions.csv
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import swin_v2_b

try:
    from torchvision.models import Swin_V2_B_Weights  # type: ignore
except ImportError:
    Swin_V2_B_Weights = None  # type: ignore


def _imagenet_stats():
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_transforms():
    if Swin_V2_B_Weights and hasattr(Swin_V2_B_Weights, "IMAGENET1K_V1"):
        mean = Swin_V2_B_Weights.IMAGENET1K_V1.meta.get("mean", _imagenet_stats()[0])
        std = Swin_V2_B_Weights.IMAGENET1K_V1.meta.get("std", _imagenet_stats()[1])
    else:
        mean, std = _imagenet_stats()
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_model(num_classes: int, checkpoint: Path, device: torch.device):
    model = swin_v2_b(weights=None)
    model.head = nn.Linear(model.head.in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    return model.to(device)


@torch.no_grad()
def run_inference(model, dl, device, class_names, csv_path: Path):
    model.eval()
    softmax = nn.Softmax(dim=1)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename"] + [f"prob_{c}" for c in class_names] + ["predicted"]
        writer.writerow(header)

        for imgs, _ in dl:
            batch_indices = range(run_inference.idx, run_inference.idx + imgs.size(0))
            filepaths = [Path(dl.dataset.samples[i][0]) for i in batch_indices]
            run_inference.idx += imgs.size(0)

            probs = softmax(model(imgs.to(device))).cpu()
            preds = probs.argmax(dim=1)

            for fp, p_vec, pred in zip(filepaths, probs, preds):
                writer.writerow(
                    [fp.name]
                    + [f"{p:.6f}" for p in p_vec.tolist()]
                    + [class_names[pred.item()]]
                )


run_inference.idx = 0  # static variable


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_dict = json.loads(Path(args.class_map).read_text())
    class_names = [class_dict[str(i)] for i in range(len(class_dict))]

    tfm = build_transforms()
    test_ds = datasets.ImageFolder(args.data_dir, transform=tfm)
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(len(class_names), Path(args.checkpoint), device)
    run_inference(model, test_dl, device, class_names, Path(args.output_csv))
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate Swin‑V2‑B model on colour fundus photographs"
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Test dataset root (class sub‑folders)",
    )
    p.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model weights (.pt)"
    )
    p.add_argument(
        "--class_map",
        type=str,
        required=True,
        help="class_indices.json file from training",
    )
    p.add_argument("--output_csv", type=str, default="predictions.csv")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    main(p.parse_args())
