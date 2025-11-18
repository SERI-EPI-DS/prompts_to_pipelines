#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune RETFound (ViT-L/16) for fundus classification on an ImageFolder dataset.
- Imports RETFound's vit_large_patch16 from the official repo (models_vit.py).
- Loads RETFound MAE weights (expects a checkpoint dict with key 'model' as per official releases).
- Uses LLRD, AdamW, cosine schedule w/ warmup, AMP, and saves best checkpoint.
- Logs CSV metrics and saves class mapping for inference.
"""

import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------- Args ----------
def get_args():
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root folder with subfolders train/val[/test] (ImageFolder).",
    )
    p.add_argument("--train_subdir", type=str, default="train")
    p.add_argument("--val_subdir", type=str, default="val")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--retfound_repo",
        type=str,
        default="",
        help="Path to RETFound repo (if this script is not placed inside it).",
    )
    p.add_argument(
        "--finetune",
        type=str,
        required=True,
        help="Path to RETFound CFP weights .pth (or other RETFound MAE weights).",
    )
    # Model / data
    p.add_argument("--nb_classes", type=int, required=True)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--drop_path", type=float, default=0.2)
    p.add_argument("--global_pool", action="store_true", default=True)
    # Optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument(
        "--blr",
        type=float,
        default=5e-3,
        help="Base LR (will be scaled by batch size / 256).",
    )
    p.add_argument("--layer_decay", type=float, default=0.65, help="LLRD factor.")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True)
    return p.parse_args()


# ---------- Utilities ----------
def seed_all(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as np

    random.seed(seed)
    np.random.seed(seed)


def ensure_repo_on_path(repo_path: str):
    if repo_path:
        sys.path.insert(0, repo_path)
    else:
        # Try current dir first
        here = Path(__file__).resolve().parent
        cand = here
        if (cand / "models_vit.py").exists():
            sys.path.insert(0, str(cand))
        # else assume user passed --retfound_repo explicitly


def build_retfound_vit_large(
    models_vit, *, num_classes, drop_path_rate=0.2, global_pool=True
):
    """
    Robustly construct a ViT-L/16 compatible with RETFound weights, even if the
    function name differs across repos.
    """
    # 1) Helpful for debugging: confirm which models_vit you imported
    try:
        print(f"[info] using models_vit from: {models_vit.__file__}")
    except Exception:
        pass

    # 2) Try common constructor names first
    for name in (
        "vit_large_patch16",
        "vit_large_patch16_224",
        "retfound_vit_large_patch16",
    ):
        ctor = getattr(models_vit, name, None)
        if callable(ctor):
            return ctor(
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                global_pool=global_pool,
            )

    # 3) Fallback: construct VisionTransformer directly with RETFound ViT-L/16 config
    if hasattr(models_vit, "VisionTransformer"):
        from functools import partial
        import torch.nn as nn

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
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
        )

    raise RuntimeError(
        "Could not find a ViT-L/16 constructor in models_vit. "
        "Ensure you're importing RETFound's models_vit.py."
    )


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    # Canonical 224 pipeline; conservative aug for medical images
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, val_tf


def create_dataloaders(
    data_dir: str,
    train_subdir: str,
    val_subdir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
):
    train_dir = Path(data_dir) / train_subdir
    val_dir = Path(data_dir) / val_subdir
    train_tf, val_tf = build_transforms(img_size)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_ds, val_ds, train_loader, val_loader


def param_groups_llrd(
    model: nn.Module, base_lr: float, layer_decay: float, weight_decay: float
) -> List[Dict]:
    """
    Create parameter groups with layer-wise LR decay for ViT-L (RETFound).
    Assumes transformer blocks are in model.blocks.<idx>.
    """
    n_layers = len(
        [
            n
            for n, _ in model.named_modules()
            if n.startswith("blocks.") and n.count(".") == 1
        ]
    )
    # Fallback if named_modules() filtering fails:
    if n_layers == 0 and hasattr(model, "blocks"):
        n_layers = len(model.blocks)

    def layer_id_for_name(name: str) -> int:
        if (
            name.startswith("cls_token")
            or name.startswith("pos_embed")
            or name.startswith("patch_embed")
        ):
            return 0
        if name.startswith("blocks."):
            try:
                idx = int(name.split(".")[1])
                return idx + 1
            except Exception:
                return 1
        # heads / fc_norm treated as top layer
        return n_layers + 1

    num_total = n_layers + 2  # [embed]=0, blocks=1..n_layers, head=last
    parameter_group_names = {}
    parameter_groups = []
    decay_exclude = ("bias", "norm", "bn", "ln", "gn")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = layer_id_for_name(name)
        group_lr = base_lr * (layer_decay ** (num_total - layer_id - 1))
        wd = 0.0 if any(x in name.lower() for x in decay_exclude) else weight_decay
        group_name = f"layer_{layer_id}_{'decay' if wd>0 else 'nodecay'}"
        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = len(parameter_groups)
            parameter_groups.append({"params": [], "lr": group_lr, "weight_decay": wd})
        parameter_groups[parameter_group_names[group_name]]["params"].append(param)
    return parameter_groups


def cosine_scheduler(optimizer, base_lr, epochs, steps_per_epoch, warmup_epochs):
    def lr_lambda(step: int):
        total_steps = epochs * steps_per_epoch
        warmup_steps = max(1, warmup_epochs * steps_per_epoch)
        if step < warmup_steps:
            return step / float(warmup_steps)
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return LambdaLR(optimizer, lr_lambda)


def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)


# ---------- Train/Eval ----------
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, scheduler=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = ce(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = ce(logits, labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(logits, labels) * bs
        n += bs
    return running_loss / n, running_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = ce(logits, labels)
        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(logits, labels) * bs
        n += bs
    return running_loss / n, running_acc / n


def main():
    args = get_args()
    seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.output_dir) / "train_log.csv"

    # Ensure RETFound repo on path, then import
    ensure_repo_on_path(args.retfound_repo)
    try:
        import models_vit  # from RETFound repo
        from util.pos_embed import interpolate_pos_embed  # repo util
        from timm.models.layers import trunc_normal_
    except Exception as e:
        raise RuntimeError(
            f"Could not import RETFound modules. "
            f"Use --retfound_repo to point to the repo. Original error: {e}"
        )

    # Build model robustly (no KeyError even if name differs)
    model = build_retfound_vit_large(
        models_vit,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path if hasattr(args, "drop_path") else 0.2,
        global_pool=True,
    )

    # Load RETFound pretrain (delete head if mismatched, interpolate pos-embed)
    ckpt = torch.load(args.finetune, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    for k in ("head.weight", "head.bias"):
        if k in state and state[k].shape != model.state_dict()[k].shape:
            del state[k]
    interpolate_pos_embed(model, state)
    msg = model.load_state_dict(state, strict=False)

    # (Re)initialize the new classifier head
    trunc_normal_(model.head.weight, std=2e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Data
    train_ds, val_ds, train_loader, val_loader = create_dataloaders(
        args.data_dir,
        args.train_subdir,
        args.val_subdir,
        args.input_size,
        args.batch_size,
        args.num_workers,
    )

    # Save class mapping (for inference)
    with open(Path(args.output_dir) / "class_to_idx.json", "w") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)

    # LR scaled by batch size (DeiT/MAE convention)
    base_lr = args.blr * args.batch_size / 256.0

    # Optimizer with LLRD
    param_groups = param_groups_llrd(
        model, base_lr, args.layer_decay, args.weight_decay
    )
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))

    # Scheduler (cosine + warmup)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = cosine_scheduler(
        optimizer, base_lr, args.epochs, steps_per_epoch, args.warmup_epochs
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    # Train loop
    best_val_acc = 0.0
    start = time.time()
    if not log_csv.exists():
        with open(log_csv, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, scheduler
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        # Track current LR (first group)
        cur_lr = optimizer.param_groups[0]["lr"]

        with open(log_csv, "a") as f:
            f.write(
                f"{epoch},{tr_loss:.6f},{tr_acc:.6f},{val_loss:.6f},{val_acc:.6f},{cur_lr:.8f}\n"
            )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                    "args": vars(args),
                    "class_to_idx": train_ds.class_to_idx,
                },
                ckpt_dir / "checkpoint-best.pth",
            )

        # Save last
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "args": vars(args),
                "class_to_idx": train_ds.class_to_idx,
            },
            ckpt_dir / "checkpoint-last.pth",
        )

        global_step += steps_per_epoch

    elapsed = (time.time() - start) / 60.0
    print(f"Done. Best val acc={best_val_acc:.4f}. Time (min)={elapsed:.1f}")


if __name__ == "__main__":
    main()
