#!/usr/bin/env python3
"""
test.py — Evaluate a RETFound ViT-L classifier on the test split and export metrics + CSVs.

Example:
  python test.py \
    --data_dir /path/to/dataset \
    --checkpoint runs/retfound_mydata/checkpoint-best.pth \
    --output_dir runs/retfound_mydata \
    --input_size 224 --batch_size 64
"""
import argparse, os, json, csv
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        accuracy_score,
    )

    HAVE_SK = True
except Exception:
    HAVE_SK = False

# -------------------- ViT-L (RETFound-style) --------------------
import timm.models.vision_transformer as tvit


class VisionTransformer(tvit.VisionTransformer):
    """Vision Transformer with global avg pool + fc_norm (RETFound)."""

    def __init__(self, global_pool=True, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs.get("norm_layer", partial(nn.LayerNorm, eps=1e-6))
            embed_dim = kwargs.get("embed_dim")
            self.fc_norm = norm_layer(embed_dim)
            # RETFound uses pooled tokens + its own norm; remove original norm
            del self.norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tok, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # pool tokens (exclude cls)
            out = self.fc_norm(x)
        else:
            x = self.norm(x)
            out = x[:, 0]
        return out  # (B, C)

    def forward(self, x):
        # ✅ Bypass timm.forward_head to avoid a second pooling step
        x = self.forward_features(x)  # (B, C)
        if hasattr(self, "head_drop"):
            x = self.head_drop(x)
        if hasattr(self, "head"):
            x = self.head(x)
        return x


def vit_large_patch16(global_pool=True, **kwargs):
    # ✅ Prevent passing img_size twice
    img_size = kwargs.pop("img_size", 224)
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        global_pool=global_pool,
        **kwargs,
    )


# -------------------- Pos-embed interpolation --------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" not in checkpoint_model or "pos_embed" not in model.state_dict():
        return
    pos_ckpt = checkpoint_model["pos_embed"]  # (1, 1+N, C)
    pos_model = model.pos_embed
    if pos_ckpt.shape == pos_model.shape:
        return
    cls_pe = pos_ckpt[:, :1, :]
    patch_pe = pos_ckpt[:, 1:, :]
    C = patch_pe.shape[-1]
    num_new = model.patch_embed.num_patches
    H_new = W_new = int(num_new**0.5)
    H_old = W_old = int(patch_pe.shape[1] ** 0.5)
    patch_pe = patch_pe.reshape(1, H_old, W_old, C).permute(0, 3, 1, 2)
    patch_pe = F.interpolate(
        patch_pe, size=(H_new, W_new), mode="bicubic", align_corners=False
    )
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, H_new * W_new, C)
    checkpoint_model["pos_embed"] = torch.cat((cls_pe, patch_pe), dim=1)


# -------------------- Data loader --------------------
def build_loader(data_dir, split, input_size, batch_size, workers):
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = datasets.ImageFolder(os.path.join(data_dir, split), transform=eval_tf)
    ld = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    return ds, ld


# -------------------- Inference utils --------------------
def load_model_from_ckpt(checkpoint_path, input_size):
    ck = torch.load(checkpoint_path, map_location="cpu")
    state = ck.get("model", ck)
    class_names = ck.get("class_names")
    if class_names is not None:
        nb_classes = len(class_names)
    else:
        # fallback if head weights are present
        nb_classes = state["head.weight"].shape[0]

    model = vit_large_patch16(
        global_pool=True, img_size=input_size, num_classes=nb_classes
    )
    # If loading a pretrain (not finetuned) checkpoint, also handle pos-embed resize
    if "pos_embed" in state and state["pos_embed"].shape != model.pos_embed.shape:
        tmp = dict(state)  # shallow copy
        interpolate_pos_embed(model, tmp)
        state = tmp
    msg = model.load_state_dict(state, strict=True)
    return model, class_names, msg


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    ds_test, ld_test = build_loader(
        args.data_dir, "test", args.input_size, args.batch_size, args.workers
    )

    # Model + checkpoint
    model, class_names, _ = load_model_from_ckpt(args.checkpoint, args.input_size)
    if class_names is None:
        class_names = ds_test.classes
    model = model.to(device).eval()

    # Run inference
    all_probs, all_preds, all_tgts = [], [], []
    with torch.no_grad():
        for imgs, tgts in ld_test:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs = logits.softmax(dim=1).cpu()
            preds = probs.argmax(dim=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_tgts.append(tgts)

    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    tgts = torch.cat(all_tgts).numpy()

    # Metrics + reports
    acc = accuracy_score(tgts, preds) if HAVE_SK else float((preds == tgts).mean())
    try:
        auc = (
            roc_auc_score(tgts, probs[:, 1])
            if probs.shape[1] == 2
            else roc_auc_score(tgts, probs, multi_class="ovr")
        )
    except Exception:
        auc = None
    print(
        f"Test accuracy: {acc:.4f} | Test AUC: {auc if auc is not None else float('nan'):.4f}"
    )

    if HAVE_SK:
        report = classification_report(tgts, preds, target_names=class_names, digits=4)
        with open(out_dir / "classification_report.txt", "w") as f:
            f.write(report)
        cm = confusion_matrix(tgts, preds)
        with open(out_dir / "confusion_matrix.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + class_names)
            for i, row in enumerate(cm):
                w.writerow([class_names[i]] + row.tolist())

    # Raw predictions
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "target", "pred"] + [f"p_{c}" for c in class_names])
        for i, (t, p, pr) in enumerate(zip(tgts, preds, probs)):
            w.writerow([i, class_names[t], class_names[p]] + [f"{x:.6f}" for x in pr])

    with open(out_dir / "test_summary.json", "w") as f:
        json.dump(
            {"test_acc": acc, "test_auc": auc, "classes": class_names}, f, indent=2
        )


if __name__ == "__main__":
    main()
