#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune RETFound ViT-L on an ImageFolder dataset.

Key fix: robust model builder selection.
- Default --model=auto picks `retfound_mae` if available (new repo),
  else uses `vit_large_patch16` (older repo), else errors with choices.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import roc_auc_score, accuracy_score


# -------------------- CLI --------------------
def get_args():
    p = argparse.ArgumentParser(description="Fine-tune RETFound ViT-L classifier")

    # Paths
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root containing train/val/test folders",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save checkpoints and logs",
    )
    p.add_argument(
        "--retfound_dir",
        type=str,
        required=True,
        help="Path to RETFound repo (contains models_vit.py, util/*)",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pre-trained RETFound weights (.pth). "
        "Default: <retfound_dir>/RETFound_CFP_weights.pth",
    )

    # Model
    p.add_argument(
        "--model",
        type=str,
        default="auto",
        help=(
            "Model builder name in models_vit.py. "
            "'auto' picks 'retfound_mae' if present, else 'vit_large_patch16'. "
            "You may also pass an explicit name, e.g. 'retfound_mae'."
        ),
    )

    # Data / aug
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--color_jitter", type=float, default=None)
    p.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    p.add_argument("--reprob", type=float, default=0.25)
    p.add_argument("--remode", type=str, default="pixel")
    p.add_argument("--recount", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--balanced",
        action="store_true",
        help="Use WeightedRandomSampler for class imbalance",
    )

    # Train
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--accum_iter", type=int, default=1)
    p.add_argument("--blr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--layer_decay", type=float, default=0.75)
    p.add_argument("--drop_path", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.1)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------- Data --------------------
def build_datasets(args):
    # use repo’s transform pipeline
    from util.datasets import build_dataset

    ds_args = SimpleNamespace(
        input_size=args.input_size,
        color_jitter=args.color_jitter,
        aa=args.aa,
        reprob=args.reprob,
        remode=args.remode,
        recount=args.recount,
        data_path=str(args.data_root),
    )
    train_ds = build_dataset(is_train="train", args=ds_args)
    val_ds = build_dataset(is_train="val", args=ds_args)
    test_ds = build_dataset(is_train="test", args=ds_args)
    return train_ds, val_ds, test_ds


def make_weighted_sampler(dataset):
    targets = (
        dataset.targets
        if hasattr(dataset, "targets")
        else [y for _, y in dataset.samples]
    )
    counts = {cls: 0 for cls in set(targets)}
    for t in targets:
        counts[t] += 1
    total = len(targets)
    class_w = {c: total / (len(counts) * n) for c, n in counts.items()}
    sample_w = [class_w[t] for t in targets]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


# -------------------- Model --------------------
def _pick_model_fn(models_vit, wanted: str):
    # list available callables
    avail = [
        k
        for k, v in models_vit.__dict__.items()
        if callable(v) and not k.startswith("_")
    ]
    # preferred names by repo generation
    pref = []
    if wanted and wanted != "auto":
        pref = [wanted]
    else:
        pref = ["retfound_mae", "vit_large_patch16"]

    for name in pref:
        if hasattr(models_vit, name):
            return getattr(models_vit, name), name

    # last-chance fallback: any builder that *looks* like retfound/vit large
    for name in avail:
        if name.startswith("retfound_") or name.startswith("vit_large"):
            return getattr(models_vit, name), name

    # give user a helpful error
    raise ValueError(
        "Could not find a suitable ViT-L builder in models_vit.\n"
        f"Requested: {wanted!r}\n"
        f"Available builders: {', '.join(sorted(avail)) or '(none)'}"
    )


def build_model_and_optimizer(args, num_classes: int):
    from timm.models.layers import trunc_normal_
    import models_vit
    from util.pos_embed import interpolate_pos_embed
    import util.lr_decay as lrd

    model_fn, chosen_name = _pick_model_fn(models_vit, args.model)
    print(f"[Info] Using model builder: {chosen_name}")

    # Many RETFound builders accept global_pool & drop_path_rate (same as official scripts)
    try:
        model = model_fn(
            num_classes=num_classes, drop_path_rate=args.drop_path, global_pool=True
        )
    except TypeError:
        # if builder doesn't accept those kwargs, retry with fewer
        model = model_fn(num_classes=num_classes)

    # Load pre-trained RETFound weights
    weights_path = args.weights or os.path.join(
        args.retfound_dir, "RETFound_CFP_weights.pth"
    )
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Pre-trained weights not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu")
    checkpoint_model = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # remove incompatible head if shapes differ
    state = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if (
            k in checkpoint_model
            and k in state
            and checkpoint_model[k].shape != state[k].shape
        ):
            del checkpoint_model[k]

    # interpolate pos_embed for size mismatch
    interpolate_pos_embed(model, checkpoint_model)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print("[Info] Loaded pretrain with msg:", msg)

    # (re)init classification head
    if hasattr(model, "head") and hasattr(model.head, "weight"):
        trunc_normal_(model.head.weight, std=2e-5)
        if getattr(model.head, "bias", None) is not None:
            nn.init.zeros_(model.head.bias)

    # layer-wise lr decay param groups
    param_groups = lrd.param_groups_lrd(
        model,
        args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay,
    )

    eff_batch = args.batch_size * args.accum_iter
    lr = args.blr * eff_batch / 256.0
    optimizer = AdamW(param_groups, lr=lr, weight_decay=args.weight_decay)
    return model, optimizer


# -------------------- Eval --------------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    all_probs, all_targets = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = ce(logits, targets)
            probs = torch.softmax(logits, dim=1)
        total_loss += loss.item()
        all_probs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    all_probs = torch.cat(all_probs, 0).numpy()
    all_targets = torch.cat(all_targets, 0).numpy()
    preds = all_probs.argmax(1)
    acc = accuracy_score(all_targets, preds)
    try:
        auc = roc_auc_score(
            np.eye(num_classes)[all_targets],
            all_probs,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        auc = float("nan")
    return dict(loss=total_loss / len(loader.dataset), acc=acc, auc=auc)


# -------------------- Train --------------------
def main():
    args = get_args()
    assert args.epochs <= 50, "Please keep epochs <= 50 as requested."

    # path + cuda
    import sys

    sys.path.insert(0, args.retfound_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # data
    train_ds, val_ds, _ = build_datasets(args)
    num_classes = len(train_ds.classes)

    if args.balanced:
        train_sampler = make_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # model/opt/loss
    model, optimizer = build_model_and_optimizer(args, num_classes)
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # schedulers
    warmup = max(0, min(args.warmup_epochs, args.epochs - 1))
    scheds, milestones = [], []
    if warmup > 0:
        scheds.append(LinearLR(optimizer, start_factor=0.001, total_iters=warmup))
        milestones.append(warmup)
    scheds.append(
        CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - warmup), eta_min=args.min_lr
        )
    )
    scheduler = SequentialLR(optimizer, scheds, milestones=milestones or [0])

    # logging
    log_csv = Path(args.output_dir) / "train_log.csv"
    if not log_csv.exists():
        with open(log_csv, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_auc,lr\n")

    best_metric = -float("inf")
    best_is_auc = True
    best_path = Path(args.output_dir) / "best_model.pth"

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0

        optimizer.zero_grad(set_to_none=True)
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets) / args.accum_iter
            scaler.scale(loss).backward()

            if (step + 1) % args.accum_iter == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(1)
                correct += (preds == targets).sum().item()
                seen += targets.numel()
                running_loss += loss.item() * args.accum_iter

        val_stats = evaluate(model, val_loader, device, num_classes)
        train_acc = correct / max(1, seen)
        lr_now = optimizer.param_groups[0]["lr"]

        with open(log_csv, "a") as f:
            f.write(
                f"{epoch},{running_loss/len(train_loader):.6f},{train_acc:.6f},"
                f"{val_stats['loss']:.6f},{val_stats['acc']:.6f},{val_stats['auc']:.6f},{lr_now:.8f}\n"
            )

        # model selection: prefer AUC if finite, else ACC
        metric = val_stats["auc"]
        use_auc = np.isfinite(metric)
        if not use_auc:
            metric = val_stats["acc"]

        if metric > best_metric:
            best_metric = metric
            best_is_auc = use_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "num_classes": num_classes,
                    "class_to_idx": train_ds.class_to_idx,
                    "args": vars(args),
                    "selected_metric": "val_auc" if use_auc else "val_acc",
                    "selected_value": float(metric),
                },
                best_path,
            )
            with open(Path(args.output_dir) / "class_to_idx.json", "w") as f:
                json.dump(train_ds.class_to_idx, f, indent=2)
            print(
                f"[Info] Saved best to {best_path} "
                f"(by {'AUC' if use_auc else 'ACC'}={metric:.4f})"
            )

        scheduler.step()

    print("Training complete.")
    print(
        f"Best validation metric: {best_metric:.4f} ({'AUC' if best_is_auc else 'ACC'})"
    )


if __name__ == "__main__":
    main()
