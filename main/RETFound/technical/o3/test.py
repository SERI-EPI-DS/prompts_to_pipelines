#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a fine-tuned RETFound ViT-L model on a held-out test set and
save per-image class probabilities to CSV.

Compatible with both 'vit_large_patch16' and 'RETFound_mae' constructors
and with either `.head` (timm-style) or `.heads.head` (torchvision-style)
classification layers.

Author: OpenAI-o3
"""
from pathlib import Path
import argparse, sys, importlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torchvision import datasets, transforms
import pandas as pd
from torch.nn.functional import softmax


# ------------------------------------------------------------------------- #
# Helper: dynamic import of RETFound repo                                    #
# ------------------------------------------------------------------------- #
def import_models_vit(retfound_dir: Path):
    parent = retfound_dir.parent.resolve()
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    return importlib.import_module(f"{retfound_dir.name}.models_vit")


# ------------------------------------------------------------------------- #
# Helper: construct model in a fork-agnostic way                             #
# ------------------------------------------------------------------------- #
def build_model(num_classes: int, ckpt_path: Path, retfound_dir: Path, device):
    mv = import_models_vit(retfound_dir)

    # pick whichever constructor this fork provides
    for cand in ("vit_large_patch16", "RETFound_mae"):
        if hasattr(mv, cand):
            model = getattr(mv, cand)()
            break
    else:
        raise AttributeError("No ViT-L constructor found in models_vit.py")

    # replace classification head so its shape matches `num_classes`
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):  # timm
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, "heads") and hasattr(model.heads, "head"):  # torchvision
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise AttributeError("Cannot locate classification layer to replace")

    # load fine-tuned weights (strict=True ensures shapes match)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    return model.to(device).eval()


# ------------------------------------------------------------------------- #
# Data transforms                                                            #
# ------------------------------------------------------------------------- #
def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# ------------------------------------------------------------------------- #
# Main                                                                       #
# ------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned RETFound model"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Folder that contains the 'test/' sub-dir",
    )
    parser.add_argument(
        "--retfound_dir",
        type=Path,
        default=Path("../RETFound"),
        help="Path to cloned RETFound repo",
    )
    parser.add_argument(
        "--weights", type=Path, required=True, help="Fine-tuned model weights (.pth)"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Destination for test_predictions.csv",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # dataset / loader ------------------------------------------------------
    test_set = datasets.ImageFolder(args.data_dir / "test", transform=get_transform())
    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(test_set.classes), args.weights, args.retfound_dir, device)

    # inference -------------------------------------------------------------
    rows, idx_offset = [], 0
    with torch.no_grad(), autocast():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            prob_batch = softmax(model(imgs), dim=1).cpu()  # [B, C]

            for i in range(prob_batch.size(0)):
                img_path, _ = test_set.samples[idx_offset + i]
                probs = prob_batch[i]
                row = {"image": Path(img_path).name}
                for c, cls_name in enumerate(test_set.classes):
                    row[f"{cls_name}_prob"] = probs[c].item()
                row["prediction"] = test_set.classes[probs.argmax().item()]
                rows.append(row)

            idx_offset += prob_batch.size(0)

    # save CSV --------------------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = args.out_dir / "test_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to: {csv_path}")


if __name__ == "__main__":
    main()
