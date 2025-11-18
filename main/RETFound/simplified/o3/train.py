#!/usr/bin/env python
"""
Fine-tune RETFound (or any timm ViT) on a fundus dataset.

Example – RETFound MAE backbone + CFP weights:
python train_retfound.py \
  --data_dir ./dataset --out_dir ./runs \
  --backbone RETFound_mae \
  --ckpt_url https://huggingface.co/open-eye/RETFound_MAE/resolve/main/RETFound_cfp_weights.pth \
  --epochs 50 --batch 32 --nb_classes 4

Example – vanilla ImageNet ViT-Large backbone (no RETFound weights):
python train_retfound.py \
  --data_dir ./dataset --out_dir ./runs \
  --backbone vit_large_patch16_224 \
  --epochs 50 --batch 32 --nb_classes 4
"""
# --------------------------------------------------------------------------- #
import argparse, os, time
from pathlib import Path
import torch, torch.nn as nn
from torchvision import datasets, transforms
from timm.scheduler.cosine_lr import CosineLRScheduler
import timm

# RETFound repo utilities
import models_vit  # repo module
from util.pos_embed import interpolate_pos_embed  # repo util


# --------------------------------------------------------------------------- #
def build_model(backbone: str, nb_classes: int, drop_path: float = 0.2):
    """Return a ViT model; backbone can be from RETFound or timm."""
    if hasattr(models_vit, backbone):  # RETFound
        model_fn = getattr(models_vit, backbone)
        model = model_fn(
            num_classes=nb_classes, global_pool=True, drop_path_rate=drop_path
        )
    else:  # any timm ViT
        model = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=nb_classes,
            drop_path_rate=drop_path,
            global_pool="avg",
        )
    return model


# --------------------------------------------------------------------------- #
def load_checkpoint(model, ckpt_path=None, ckpt_url=None):
    if ckpt_path is None and ckpt_url is None:  # nothing to load
        return
    state = (
        torch.hub.load_state_dict_from_url(ckpt_url, map_location="cpu")
        if ckpt_url
        else torch.load(ckpt_path, map_location="cpu")
    )
    # unwrap dicts from different conventions
    state = state.get("model", state)
    # remove mismatching head weights
    for k in ["head.weight", "head.bias"]:
        if k in state and state[k].shape != model.state_dict()[k].shape:
            del state[k]
    interpolate_pos_embed(model, state)
    _ = model.load_state_dict(state, strict=False)


# --------------------------------------------------------------------------- #
def get_dataloaders(root, img_size, batch, workers=8):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tr_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    te_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_ds = datasets.ImageFolder(Path(root) / "train", tr_tf)
    val_ds = datasets.ImageFolder(Path(root) / "val", te_tf)
    return (
        train_ds.classes,
        torch.utils.data.DataLoader(
            train_ds, batch, True, num_workers=workers, pin_memory=True
        ),
        torch.utils.data.DataLoader(
            val_ds, batch, False, num_workers=workers, pin_memory=True
        ),
    )


# --------------------------------------------------------------------------- #
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes, dl_train, dl_val = get_dataloaders(
        args.data_dir, args.img_size, args.batch
    )
    model = build_model(args.backbone, len(classes)).to(device)
    load_checkpoint(model, args.ckpt_path, args.ckpt_url)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = CosineLRScheduler(
        optim,
        t_initial=args.epochs,
        lr_min=1e-6,
        warmup_t=5,
        warmup_lr_init=args.lr / 10,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0
    for ep in range(1, args.epochs + 1):
        # ---------- train ----------
        model.train()
        n, correct = 0, 0
        for x, y in dl_train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optim.step()
            sched.step(ep + n / len(dl_train))
            n += y.size(0)
            correct += (out.argmax(1) == y).sum().item()
        tr_acc = correct / n

        # ---------- validate ----------
        model.eval()
        n, correct = 0, 0
        with torch.no_grad():
            for x, y in dl_val:
                out = model(x.to(device))
                pred = out.argmax(1).cpu()
                n += y.size(0)
                correct += (pred == y).sum().item()
        val_acc = correct / n
        print(f"Epoch {ep:3d}/{args.epochs}  train {tr_acc:.4f}  val {val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "classes": classes},
                f"{args.out_dir}/best_{args.backbone}.pth",
            )
    print("Best val-acc:", best)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="train/val/test folders root")
    ap.add_argument("--out_dir", default="./runs")
    ap.add_argument(
        "--backbone",
        default="RETFound_mae",
        help="RETFound_mae | RETFound_dinov2 | any timm ViT name",
    )
    ap.add_argument(
        "--ckpt_url",
        default=None,
        help="(optional) URL to weights, e.g. HuggingFace file URL",
    )
    ap.add_argument(
        "--ckpt_path", default=None, help="(optional) local .pth checkpoint"
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    train(args)
