# ============================
# test.py
# Evaluates a trained RETFound classifier and outputs predictions to CSV.
# --------------------------------------------

"""Example usage
python test.py --data_dir /path/to/dataset --weights ./outputs/best_model.pth --output_csv ./outputs/test_predictions.csv
"""

import argparse
import csv
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


@torch.inference_mode()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_tfms = transforms.Compose(
        [
            transforms.Resize(int(args.img_size * 1.15)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_ds = datasets.ImageFolder(Path(args.data_dir) / "test", transform=test_tfms)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Build model and load weights
    checkpoint = torch.load(args.weights, map_location="cpu")
    classes = checkpoint["classes"]
    model = timm.create_model("vit_large_patch16_224", num_classes=len(classes))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    preds_all = []
    for imgs, targets in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        outputs = model(imgs).softmax(dim=1)
        preds = outputs.argmax(1).cpu()
        preds_all.extend(zip(test_ds.samples, outputs.cpu().tolist(), preds.tolist()))

    # Write CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        header = ["filename"] + [f"score_{c}" for c in classes] + ["prediction"]
        writer.writerow(header)
        for (filename, _), scores, pred in preds_all:
            writer.writerow(
                [Path(filename).name] + [f"{s:.6f}" for s in scores] + [classes[pred]]
            )
    print(f"\u2705 Predictions saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test RETFound classifier")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Dataset root directory containing test folder",
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (.pth)",
    )
    p.add_argument("--output_csv", type=str, default="./predictions.csv")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    main(p.parse_args())
