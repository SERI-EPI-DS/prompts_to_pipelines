#!/usr/bin/env python3
"""
Evaluate a fine-tuned ConvNeXt-Large checkpoint.

Works whether --data_dir contains sub-folders per class (labels available)
or a plain folder of images (no labels → only a CSV of predictions).
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# ----------------------------------------------------------------------
# TorchMetrics compatibility helper
# ----------------------------------------------------------------------
try:  # ≥ 0.11  (current API)
    from torchmetrics.functional import accuracy, auroc

    _ACC_KW = dict(task="multiclass")
    _AUROC_KW = dict(task="multiclass")
except ImportError:  # very old versions (<0.11)
    from torchmetrics.functional import accuracy, multiclass_auroc as _auroc  # type: ignore

    def auroc(preds, target, num_classes, **_):  # wrapper with the new signature
        return _auroc(preds, target, num_classes=num_classes)

    _ACC_KW, _AUROC_KW = {}, {}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Folder with images or class sub-folders.",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to checkpoint .pth saved by training script.",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out_csv", type=Path, default=Path("predictions.csv"))
    p.add_argument("--out_metrics", type=Path, default=Path("metrics.json"))
    return p.parse_args()


# ----------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------
def build_loader(
    root: Path, img_size: int, bs: int, workers: int, class_names: list[str]
) -> DataLoader:
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tfm = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            norm,
        ]
    )

    # Case A: root has sub-folders for each class  ➜ ImageFolder (labels exist)
    if any((root / c).is_dir() for c in class_names):
        ds = datasets.ImageFolder(root, tfm)
        collate_with_path = False
    else:
        # Case B: plain folder of images ➜ build dummy dataset, no labels
        class Dummy(torch.utils.data.Dataset):
            def __init__(self, folder: Path):
                self.files = sorted(
                    [
                        p
                        for p in folder.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    ]
                )

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx: int):
                from PIL import Image

                img = Image.open(self.files[idx]).convert("RGB")
                return tfm(img), -1, str(self.files[idx])  # -1: “no label”

        ds = Dummy(root)
        collate_with_path = True

    return DataLoader(
        ds,
        bs,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=None if not collate_with_path else None,
    )


# ----------------------------------------------------------------------
def main(a):
    ckpt = torch.load(a.ckpt, map_location="cpu")
    class_names: list[str] = ckpt["classes"]
    img_size: int = ckpt["img_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model = timm.create_model(
        "convnext_large_in22k", num_classes=len(class_names), pretrained=False
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)

    # Data
    loader = build_loader(
        a.data_dir, img_size, a.batch_size, a.num_workers, class_names
    )

    # ------------------------------------------------------------------
    paths, preds_int, labels_int, probs_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            # Handle ImageFolder vs DummyDataset format
            if len(batch) == 3:
                imgs, labels, pths = batch
            else:
                imgs, labels = batch
                pths = ["?"] * len(imgs)

            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs = logits.softmax(-1).cpu()  # for AUROC
            preds = probs.argmax(1)

            probs_list.append(probs)
            preds_int.extend(preds.tolist())
            labels_int.extend(labels.tolist())
            paths.extend(pths)

    # ------------------------------------------------------------------
    # 1) CSV with predictions
    with open(a.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "pred_class"])
        for pth, pred_idx in zip(paths, preds_int):
            writer.writerow([pth, class_names[pred_idx]])
    print(f"[✓] Predictions written to {a.out_csv.resolve()}")

    # 2) Metrics (if any ground-truth labels present)
    if all(l >= 0 for l in labels_int):
        labels_t = torch.tensor(labels_int)
        preds_t = torch.tensor(preds_int)
        probs_t = torch.cat(probs_list, 0)

        acc = accuracy(
            preds_t, labels_t, num_classes=len(class_names), **_ACC_KW
        ).item()
        auroc_val = auroc(
            probs_t, labels_t, num_classes=len(class_names), **_AUROC_KW
        ).item()

        with open(a.out_metrics, "w") as f:
            json.dump({"accuracy": acc, "auroc": auroc_val}, f, indent=2)
        print(
            f"[✓] Accuracy = {acc:.4f} | AUROC = {auroc_val:.4f} "
            f"(saved to {a.out_metrics.resolve()})"
        )
    else:
        print("No ground-truth labels detected → metrics skipped.")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main(parse_args())
