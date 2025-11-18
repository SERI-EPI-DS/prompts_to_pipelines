#!/usr/bin/env python
"""
Fine-tune RETFound ViT-Large – all runtime errors fixed
Author: ChatGPT (OpenAI o3)
"""
import argparse, json, random, copy
from pathlib import Path
from functools import partial
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, accuracy_score

# -------------------------------------------------------------------------
#  make sure we import the repo-local helpers first
REPO_ROOT = Path(__file__).resolve().parent / "RETFound_MAE"
import sys

sys.path.insert(0, str(REPO_ROOT))  # shadow any pip installs
import models_vit  # includes vit_large_patch16
from engine_finetune import train_one_epoch  # still use the robust trainer
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# -------------------------------------------------------------------------


# ---------- CLI ----------
def get_args():
    p = argparse.ArgumentParser(description="Fine-tune RETFound-MAE ViT-L")
    p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--pretrained_ckpt", required=True, type=Path)
    p.add_argument("--nb_classes", required=True, type=int)

    # optimiser & schedule --------------------------------------------------
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5, help="for LR warm-up")
    p.add_argument("--min_lr", type=float, default=1e-6, help="final cosine LR")
    p.add_argument("--accum_iter", type=int, default=1, help="grad accumulation")
    p.add_argument("--clip_grad", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------- dataset ----------
def build_loaders(root, bs, nw):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, (0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    trn = datasets.ImageFolder(root / "train", train_tf)
    val = datasets.ImageFolder(root / "val", val_tf)
    return (
        DataLoader(trn, bs, True, num_workers=nw, pin_memory=True),
        DataLoader(val, bs, False, num_workers=nw, pin_memory=True),
        trn.classes,
    )


# ---------- simple evaluator (sidesteps fragile repo helper) -------------
@torch.no_grad()
def evaluate_safe(model, dl, device, num_classes):
    model.eval()
    ce = nn.CrossEntropyLoss()
    tot_loss, n = 0.0, 0
    probs_all, gts_all = [], []

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, 1).cpu().numpy()
        probs_all.append(probs)
        gts_all.append(y.cpu().numpy())
        tot_loss += ce(logits, y).item() * x.size(0)
        n += x.size(0)

    probs_all = np.concatenate(probs_all)
    gts_all = np.concatenate(gts_all)

    # --- per-class AUROC only where that class exists ---------------------
    present_classes = np.unique(gts_all)
    aucs = []
    for c in present_classes:
        try:
            auc = roc_auc_score((gts_all == c).astype(int), probs_all[:, c])
            aucs.append(auc)
        except ValueError:  # still undefined (all 0 or 1)
            continue
    macro_auc = float(np.mean(aucs)) if aucs else np.nan

    acc = accuracy_score(gts_all, probs_all.argmax(1))
    return tot_loss / n, macro_auc, acc


# ---------- model factory ----------
def build_model(nc):
    if hasattr(models_vit, "vit_large_patch16"):
        return models_vit.vit_large_patch16(
            num_classes=nc, global_pool=True, drop_path_rate=0.2
        )
    # fallback – manual constructor with mandatory norm_layer
    return models_vit.VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=nc,
        global_pool=True,
    )


# ---------- main ----------
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # model & weights -------------------------------------------------------
    model = build_model(args.nb_classes)
    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")["model"]
    msg = model.load_state_dict(ckpt, strict=False)
    print("Backbone loaded:", msg)
    model.cuda()

    # optimiser ------------------------------------------------------------
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = NativeScaler()  # AMP scaler :contentReference[oaicite:9]{index=9}
    crit = nn.CrossEntropyLoss()

    tr_dl, va_dl, class_names = build_loaders(
        args.data_dir, args.batch_size, args.workers
    )
    best_auc, best_state = 0.0, None

    for ep in range(args.epochs):
        train_one_epoch(
            model,
            crit,
            tr_dl,
            opt,
            torch.device("cuda"),
            ep,
            scaler,
            args.clip_grad,
            mixup_fn=None,
            log_writer=None,
            args=args,
        )  # correct signature :contentReference[oaicite:10]{index=10}
        sched.step()

        val_loss, val_auc, val_acc = evaluate_safe(
            model, va_dl, torch.device("cuda"), args.nb_classes
        )
        print(
            f"Epoch {ep}: loss={val_loss:.4f} | acc={val_acc:.4f} | AUROC={val_auc:.4f}"
        )
        score = val_auc if not np.isnan(val_auc) else val_acc
        if score > best_auc:  # best_auc keeps same name
            best_auc, best_state = score, copy.deepcopy(model.state_dict())
            torch.save(best_state, args.output_dir / "best_model.pt")

    json.dump({"classes": class_names}, open(args.output_dir / "classes.json", "w"))
    print(f"Training finished; best val AUROC = {best_auc:.4f}")


if __name__ == "__main__":
    main()
