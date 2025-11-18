# test.py – Inference using fine‑tuned RETFound_mae ViT‑L
"""
Usage example
-------------
python test.py \
    --data_dir /path/to/data/test \
    --weights ./results/best_model.pth \
    --output_csv ./results/test_predictions.csv \
    --batch_size 16

The script expects a flat <data_dir>/<class_folders>/images structure (similar to train/val).
Outputs
-------
* CSV file with columns: filename, <class0>, <class1>, …, predicted_label
"""

import argparse
from pathlib import Path
import csv

import torch
from torch import nn
from torchvision.datasets import ImageFolder

try:
    import torchvision.transforms.v2 as transforms
except ImportError:
    from torchvision import transforms  # type: ignore

try:
    import timm
except ImportError as e:
    raise ImportError("Please install timm: pip install timm") from e


def parse_args():
    p = argparse.ArgumentParser(description="RETFound ViT‑L Inference")
    p.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing test images (class sub‑folders)",
    )
    p.add_argument(
        "--weights", type=Path, required=True, help="Path to fine‑tuned .pth weights"
    )
    p.add_argument("--output_csv", type=Path, default=Path("./test_predictions.csv"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    return p.parse_args()


def get_transform(img_size: int):
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            norm,
        ]
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = get_transform(args.image_size)
    ds = ImageFolder(root=args.data_dir, transform=tfm)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_classes = len(ds.classes)
    model = timm.create_model(
        "vit_large_patch16_224", num_classes=num_classes, pretrained=False
    )
    state = torch.load(args.weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded weights with {len(missing)} missing / {len(unexpected)} unexpected keys"
    )
    model.to(device)
    model.eval()

    softmax = nn.Softmax(dim=1)

    with args.output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename"] + [f"prob_{c}" for c in ds.classes] + ["predicted_label"]
        writer.writerow(header)

        with torch.no_grad(), torch.cuda.amp.autocast():
            for imgs, labels in loader:  # labels unused but keeps ImageFolder happy
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = softmax(outputs).cpu()
                preds = probs.argmax(dim=1)

                for idx, prob_row in enumerate(probs):
                    filepath = ds.samples[idx][0]
                    fname = Path(filepath).name
                    prob_list = prob_row.tolist()
                    pred_label = ds.classes[preds[idx]]
                    writer.writerow(
                        [fname] + [f"{p:.6f}" for p in prob_list] + [pred_label]
                    )

    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
