#!/usr/bin/env python
# Inference with fixed filename ordering & norm_layer
import argparse, json, random, copy
from pathlib import Path
from functools import partial
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

# --- make sure we import the *repo’s* models_vit ----------------------------
REPO_ROOT = Path(__file__).resolve().parent / "RETFound_MAE"
import sys

sys.path.insert(0, str(REPO_ROOT))  # <-- prepend instead of append
import models_vit  # now contains vit_large_patch16 :contentReference[oaicite:2]{index=2}
from engine_finetune import train_one_epoch, evaluate

# ----------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--nb_classes", required=True, type=int)
    p.add_argument("--checkpoint", default="best_model.pt")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def build_model(num_classes):
    """Factory wrapper so we fall back gracefully."""
    if hasattr(models_vit, "vit_large_patch16"):  # normal path
        return models_vit.vit_large_patch16(
            num_classes=num_classes, global_pool=True, drop_path_rate=0.2
        )
    # -- very unlikely fallback: construct manually – keeps ‘norm_layer’ key
    return models_vit.VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        global_pool=True,
    )


def main():
    a = get_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    ds = datasets.ImageFolder(a.data_dir / "test", tf)
    dl = DataLoader(ds, a.batch_size, False, num_workers=a.workers, pin_memory=True)

    model = build_model(num_classes=a.nb_classes)
    model.load_state_dict(
        torch.load(Path(a.output_dir) / a.checkpoint, map_location="cpu")
    )
    model.cuda().eval()

    probs = []
    with torch.no_grad():
        for imgs, _ in dl:
            logits = model(imgs.cuda(non_blocking=True))
            probs.append(F.softmax(logits, 1).cpu().numpy())
    probs = np.concatenate(probs)

    classes = json.load(open(Path(a.output_dir) / "classes.json"))["classes"]
    df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in classes])
    df.insert(0, "filename", [Path(p).name for p, _ in ds.samples])
    df["prediction"] = [classes[i] for i in probs.argmax(1)]
    df.to_csv(Path(a.output_dir) / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
