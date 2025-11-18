#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune RETFound_mae (ViT-L/16) for multi-class retinal image classification.

- Expects folder structure:
  data_root/
    train/<class>/*.png|jpg
    val/<class>/*.png|jpg
    test/<class>/*.png|jpg   (unused here; evaluated by test.py)

- Loads initial weights from a local RETFound checkpoint (e.g., RETFound_CFP_weights.pth).
- Uses CE loss with label smoothing, cosine schedule with warmup, and layer-wise LR decay.
- Saves best checkpoint (by val accuracy) and training log into output directory.

Tested with: Python 3.11.0, PyTorch 2.3.1, TorchVision 0.18.1, CUDA 12.1.
"""
import os
import sys
import json
import math
import time
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(s: str):
    return s


def pretty_time(sec: float) -> str:
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h{int(m):02d}m{int(s):02d}s"
    return f"{int(m)}m{int(s):02d}s"


# ------------------------------------------------------------
# RETFound model utilities
# ------------------------------------------------------------
def add_retfound_to_syspath(retfound_root: str):
    retfound_path = Path(retfound_root).resolve()
    if not retfound_path.exists():
        raise FileNotFoundError(f"RETFound repo path not found: {retfound_path}")
    if str(retfound_path) not in sys.path:
        sys.path.insert(0, str(retfound_path))


def build_model(
    num_classes: int,
    input_size: int = 224,
    drop_path_rate: float = 0.2,
    global_pool: bool = True,
):
    """
    Tries RETFound_mae first; falls back to vit_large_patch16 in models_vit.py.
    """
    import models_vit as rv  # from RETFound repo (sys.path injected)

    # Attempt RETFound_mae entrypoint (preferred per repo README)
    model = None
    if hasattr(rv, "RETFound_mae"):
        try:
            # Many repos accept (num_classes, drop_path_rate, img_size, global_pool)
            model = rv.RETFound_mae(
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                img_size=input_size,
                global_pool=global_pool,
            )
        except TypeError:
            # Fallback signature
            model = rv.RETFound_mae(num_classes=num_classes)
    elif hasattr(rv, "vit_large_patch16"):
        # Fallback direct ViT-L/16 constructor exposed by RETFound repo
        try:
            model = rv.vit_large_patch16(
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                img_size=input_size,
                global_pool=global_pool,
            )
        except TypeError:
            model = rv.vit_large_patch16(num_classes=num_classes)
    else:
        raise RuntimeError(
            "Could not locate RETFound_mae or vit_large_patch16 in models_vit.py"
        )

    # Defensive classifier reset (covers variations of attribute names)
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_f = model.head.in_features
        model.head = nn.Linear(in_f, num_classes)
    elif (
        hasattr(model, "heads")
        and hasattr(model.heads, "head")
        and isinstance(model.heads.head, nn.Linear)
    ):
        in_f = model.heads.head.in_features
        model.heads.head = nn.Linear(in_f, num_classes)
    return model


@torch.no_grad()
def _maybe_interpolate_pos_embed(model, state_dict: Dict[str, torch.Tensor]):
    """
    Use RETFound's util.pos_embed.interpolate_pos_embed if available.
    This function may mutate the state_dict to match model's grid size.
    """
    try:
        from util.pos_embed import interpolate_pos_embed  # from RETFound repo
    except Exception:
        return
    try:
        # Some implementations expect the full checkpoint dict; others accept (model, state_dict)
        # Here we follow the common pattern used in MAE/DeiT derivatives: mutate state_dict in place.
        interpolate_pos_embed(model, state_dict)
    except Exception:
        # Silent best-effort; mismatch will be handled by strict=False load
        pass


def load_retfound_weights(model: nn.Module, weights_path: str, device: torch.device):
    """
    Loads arbitrary checkpoint formats:
      - raw state dict
      - {"model": state_dict, ...}
      - {"state_dict": state_dict, ...}
    Removes classifier weights if shapes mismatch; strips DistributedDataParallel prefixes.
    """
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt

    # Strip 'module.' (DDP) prefixes if present
    new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Interpolate positional embeddings if needed
    _maybe_interpolate_pos_embed(model, new_state)

    # Drop head weights if shape mismatch
    model_state = model.state_dict()
    drop_keys = []
    for k, v in new_state.items():
        if "head" in k and k in model_state and v.shape != model_state[k].shape:
            drop_keys.append(k)
    for k in drop_keys:
        new_state.pop(k, None)

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[load] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if len(missing) < 20:  # informational
        print(f"[load] missing: {missing}")
    if len(unexpected) < 20:
        print(f"[load] unexpected: {unexpected}")

    model.to(device)


# ------------------------------------------------------------
# Layer-wise LR decay (ViT-friendly)
# ------------------------------------------------------------
def param_groups_lrd(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.05,
    layer_decay: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with layer-wise learning-rate decay.
    Assumes ViT-like structure with `blocks`, `patch_embed`, `pos_embed`, `cls_token`.
    """
    # Number of transformer blocks
    n_blocks = len(getattr(model, "blocks", []))
    n_layers = n_blocks + 2  # +2 for patch_embed (0) and head/norm (last)

    def layer_id(name: str) -> int:
        if name.startswith("patch_embed") or name in {"cls_token", "pos_embed"}:
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        # final layers (norm/head)
        return n_layers - 1

    lr_scales = [layer_decay ** (n_layers - 1 - i) for i in range(n_layers)]
    groups: Dict[Tuple[int, bool], Dict[str, Any]] = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_norm = (".norm" in name) or name.endswith(".bias") or (".bias" in name)
        wd = 0.0 if is_norm else weight_decay
        lid = layer_id(name)
        key = (lid, is_norm)
        if key not in groups:
            groups[key] = {
                "params": [],
                "weight_decay": wd,
                "lr": base_lr * lr_scales[lid],
            }
        groups[key]["params"].append(p)

    return list(groups.values())


# ------------------------------------------------------------
# Training / validation loops
# ------------------------------------------------------------
def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    scheduler=None,
    grad_clip: float = 1.0,
    accum_steps: int = 1,
) -> Tuple[float, float]:
    model.train()
    running_loss, running_acc = 0.0, 0.0
    optimizer.zero_grad(set_to_none=True)
    steps = 0

    for it, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = model(images)
            loss = criterion(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        if (it + 1) % accum_steps == 0:
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * accum_steps
        running_acc += accuracy(logits, labels)
        steps += 1

    return running_loss / steps, running_acc / steps


@torch.no_grad()
def validate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss, running_acc, steps = 0.0, 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = model(images)
            loss = criterion(logits, labels)
        running_loss += loss.item()
        running_acc += accuracy(logits, labels)
        steps += 1
    return running_loss / steps, running_acc / steps


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune RETFound_mae (ViT-L/16)")
    ap.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root with train/ val/ test/ subfolders",
    )
    ap.add_argument(
        "--retfound-root",
        type=str,
        required=True,
        help="Path to cloned RETFound repo (contains models_vit.py, util/ etc.)",
    )
    ap.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to pre-trained RETFound weights (.pth), e.g. RETFound_CFP_weights.pth",
    )
    ap.add_argument(
        "--out-dir", type=str, required=True, help="Where to write checkpoints and logs"
    )
    ap.add_argument("--epochs", type=int, default=50, help="Max epochs (<=50 per spec)")
    ap.add_argument("--batch-size", type=int, default=16, help="Per-GPU batch size")
    ap.add_argument(
        "--accum-steps", type=int, default=1, help="Grad accumulation steps"
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--drop-path", type=float, default=0.2)
    ap.add_argument("--blr", type=float, default=5e-3, help="Base LR (pre-decay)")
    ap.add_argument("--layer-decay", type=float, default=0.65)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If >0, save a checkpoint every N epochs in addition to best.",
    )
    ap.add_argument(
        "--global-pool",
        action="store_true",
        help="Use global pooling head if available",
    )
    return ap.parse_args()


def make_dataloaders(data_root: str, input_size: int, batch_size: int, workers: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.15)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.ImageFolder(
        os.path.join(data_root, "train"), transform=train_tf
    )
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Make RETFound code importable
    add_retfound_to_syspath(args.retfound_root)

    # Repro + CUDA setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Data
    train_loader, val_loader, classes = make_dataloaders(
        args.data_root, args.input_size, args.batch_size, args.workers
    )
    num_classes = len(classes)
    print(f"[data] classes ({num_classes}): {classes}")

    # Model
    model = build_model(
        num_classes=num_classes,
        input_size=args.input_size,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load pre-trained RETFound (foundation) weights
    print(f"[load] loading RETFound weights from: {args.weights}")
    load_retfound_weights(model, args.weights, device)

    # Loss / Optim / Sched
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    param_groups = param_groups_lrd(
        model,
        base_lr=args.blr,
        weight_decay=args.weight_decays if False else args.weight_decay,
        layer_decay=args.layer_decay,
    )
    optimizer = optim.AdamW(
        param_groups, lr=args.blr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Train
    best_acc = 0.0
    best_path = os.path.join(args.out_dir, "checkpoint-best.pth")
    classes_path = os.path.join(args.out_dir, "classes.json")
    log_path = os.path.join(args.out_dir, "train_log.jsonl")

    print(
        f"[train] epochs={args.epochs}, batch_size={args.batch_size}, accum={args.accum_steps}, input={args.input_size}"
    )
    t0 = time.time()
    with open(log_path, "w") as flog:
        for epoch in range(1, args.epochs + 1):
            t_ep = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                scheduler=scheduler,
                grad_clip=1.0,
                accum_steps=args.accum_steps,
            )
            va_loss, va_acc = validate(model, val_loader, criterion, device)
            dt = time.time() - t_ep

            log_rec = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "time_sec": dt,
                "lr_groups": [g["lr"] for g in optimizer.param_groups],
            }
            flog.write(json.dumps(log_rec) + "\n")
            flog.flush()

            print(
                f"[epoch {epoch:03d}/{args.epochs}] "
                f"train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | "
                f"val: loss={va_loss:.4f}, acc={va_acc:.4f} | {pretty_time(dt)}"
            )

            # Save periodic
            if args.save_every and (epoch % args.save_every == 0):
                pth = os.path.join(args.out_dir, f"checkpoint-epoch{epoch}.pth")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "classes": classes,
                        "args": vars(args),
                    },
                    pth,
                )

            # Save best
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "classes": classes,
                        "args": vars(args),
                    },
                    best_path,
                )
                with open(classes_path, "w") as f:
                    json.dump({"classes": classes}, f, indent=2)
                print(f"[save] new best acc={best_acc:.4f} -> {best_path}")

    print(
        f"[done] total time: {pretty_time(time.time() - t0)} | best val acc={best_acc:.4f}"
    )


if __name__ == "__main__":
    main()
