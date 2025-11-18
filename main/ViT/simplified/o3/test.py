#!/usr/bin/env python3
# test.py  –  Fundus-image diagnosis inference script (v6, 2025-06-30)

import argparse
from pathlib import Path

import torch
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm


# ───────────────── Dataset helper ─────────────────────────────────────────────
class ImageFolderWithPaths(ImageFolder):
    """`torchvision.datasets.ImageFolder` that also returns the image file path."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, target, str(path)  # ← convert Path → str


def get_transforms(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(
                img_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ───────────────── Model helper ───────────────────────────────────────────────
def safe_create_model(
    model_name: str, num_classes: int, allow_pretrained: bool = True
) -> torch.nn.Module:
    avail = set(timm.list_models(pretrained=allow_pretrained))
    for cand in [
        model_name,
        "swinv2_small_window8_256",
        "swinv2_tiny_window8_256",
        "swin_base_patch4_window7_224",
        "vit_base_patch16_224",
        "resnet50",
        "resnet18",
    ]:
        if cand in avail:
            print(f"[INFO] Using backbone “{cand}”.")
            return timm.create_model(
                cand, pretrained=allow_pretrained, num_classes=num_classes
            )

    # Last-ditch: torchvision ResNet-18 (random init)
    from torchvision.models import resnet18

    print("[WARN] No usable timm backbone found; falling back to torchvision.resnet18.")
    return resnet18(num_classes=num_classes)


# ───────────────── CLI parsing ────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fundus diagnosis inference")
    p.add_argument("--test_dir", required=True, help="Folder with test images.")
    p.add_argument("--checkpoint", required=True, help=".pth checkpoint to load.")
    p.add_argument("--model_name", default="swinv2_base_window12_192_22k")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_csv", default="predictions.csv")
    return p.parse_args()


# ───────────────── Main ───────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on {device}")

    # Dataset & loader
    test_set = ImageFolderWithPaths(
        args.test_dir, transform=get_transforms(args.img_size)
    )
    class_names = test_set.classes
    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    model = safe_create_model(args.model_name, num_classes=len(class_names))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.to(device).eval()

    # Inference
    records = []
    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc="Inference"):
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)

            for pth, true_idx, pred_idx, prob_vec in zip(
                paths, labels.cpu().tolist(), preds.cpu().tolist(), probs
            ):
                records.append(
                    {
                        "image": Path(pth).name,
                        "pred_label": class_names[pred_idx],
                        "true_label": class_names[true_idx],
                        **{
                            f"conf_{cls}": float(prob_vec[j])
                            for j, cls in enumerate(class_names)
                        },
                    }
                )

    # Metrics & save
    df = pd.DataFrame(records)
    acc = (df["pred_label"] == df["true_label"]).mean()
    print(f"\nTop-1 accuracy: {acc*100:.2f}%  (on {len(df)} images)")
    df.to_csv(args.output_csv, index=False)
    print(
        f"[INFO] Predictions written to “{args.output_csv}”. "
        f"Columns: image | pred_label | true_label | conf_<class> …"
    )


if __name__ == "__main__":
    main()
