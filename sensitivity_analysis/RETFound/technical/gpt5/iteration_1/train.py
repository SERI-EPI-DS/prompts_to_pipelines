#!/usr/bin/env python3
"""
Fine-tune RETFound_mae (ViT-L/16) for multi-class classification on colour fundus photographs.

Key changes vs previous version:
- Imports RETFound modules by absolute file path (importlib) to avoid name collisions.
- Builds ViT-L robustly: tries known factory names, else instantiates VisionTransformer directly.
- Still loads RETFound MAE weights, interpolates pos-emb, and re-inits classifier head.

Environment: Python 3.11, PyTorch 2.3+, TorchVision 0.18+, CUDA 12.1, single GPU.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


# -----------------------------
# CLI
# -----------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="Fine-tune RETFound_mae (ViT-L) for retinal classification"
    )
    # Paths
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder with train/val/test subfolders",
    )
    p.add_argument(
        "--retfound_dir", type=str, required=True, help="Path to RETFound repo folder"
    )
    p.add_argument(
        "--pretrained_ckpt",
        type=str,
        required=True,
        help="Path to RETFound MAE weights (.pth)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save checkpoints/results",
    )

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pin_mem", action="store_true", default=True)
    p.add_argument("--accum_steps", type=int, default=1)

    # Model & image
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--drop_path", type=float, default=0.2)
    p.add_argument("--global_pool", action="store_true", default=True)

    # Optim / sched
    p.add_argument(
        "--blr",
        type=float,
        default=5e-4,
        help="Base LR; actual lr = blr * batch_size / 256",
    )
    p.add_argument("--lr", type=float, default=None, help="Absolute LR override")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--clip_grad_norm", type=float, default=None)

    # Loss / misc
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--save_every", type=int, default=0)
    return p.parse_args()


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(
                int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, eval_tf


def epoch_cosine_lambda(epoch, warmup_epochs, total_epochs, min_lr_ratio):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0).item()
            res.append(100.0 * correct_k / target.size(0))
        return res


# -----------------------------
# Safe file-based imports
# -----------------------------
import importlib.util


def import_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def load_retfound_modules(retfound_dir: str):
    # models_vit.py
    mv_path = os.path.join(retfound_dir, "models_vit.py")
    models_vit = import_from_file("retfound_models_vit", mv_path)
    # util/pos_embed.py
    pe_path = os.path.join(retfound_dir, "util", "pos_embed.py")
    pos_embed = import_from_file("retfound_pos_embed", pe_path)
    return models_vit, pos_embed


def build_vitl_model(
    models_vit, num_classes: int, drop_path: float, global_pool: bool, img_size: int
):
    """
    Try factory names first; else instantiate VisionTransformer directly with ViT-L/16 config
    (embed_dim=1024, depth=24, heads=16).
    """
    candidate_factories = [
        "vit_large_patch16",
        "vit_large_patch16_224",
        "vit_large_patch16_mae",
        "RETFound_vit_large_patch16",
    ]
    for name in candidate_factories:
        if hasattr(models_vit, name):
            return getattr(models_vit, name)(
                num_classes=num_classes,
                drop_path_rate=drop_path,
                global_pool=global_pool,
                img_size=img_size,  # harmless if factory ignores it
            )
    # Fallback: construct VisionTransformer manually (matches RETFound MAE ViT-L)
    VT = getattr(models_vit, "VisionTransformer", None)
    if VT is None:
        import timm.models.vision_transformer as tvit

        VT = tvit.VisionTransformer
    return VT(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        drop_path_rate=drop_path,
        global_pool=global_pool,
    )


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    if args.epochs > 50:
        print(f"[warn] epochs capped to 50 (was {args.epochs})")
        args.epochs = 50

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_tf, eval_tf = build_transforms(args.img_size)
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")
    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(val_dir, transform=eval_tf)
    num_classes = len(train_set.classes)
    print(f"Classes ({num_classes}): {train_set.classes}")

    # Save class mapping for test.py
    classes_path = os.path.join(args.output_dir, "classes.json")
    json.dump(
        {
            "classes": train_set.classes,
            "class_to_idx": train_set.class_to_idx,
            "idx_to_class": {int(v): k for k, v in train_set.class_to_idx.items()},
        },
        open(classes_path, "w"),
        indent=2,
    )

    # Load RETFound modules by file path (no collisions)
    models_vit, pos_embed_mod = load_retfound_modules(args.retfound_dir)
    from timm.models.layers import trunc_normal_

    interpolate_pos_embed = getattr(pos_embed_mod, "interpolate_pos_embed")

    # Build model robustly
    model = build_vitl_model(
        models_vit=models_vit,
        num_classes=num_classes,
        drop_path=args.drop_path,
        global_pool=args.global_pool,
        img_size=args.img_size,
    )

    # Load RETFound MAE weights, drop incompatible head, interpolate pos-emb
    print(f"Loading RETFound pre-trained checkpoint: {args.pretrained_ckpt}")
    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
    checkpoint_model = ckpt.get("model", ckpt)

    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if (
            k in checkpoint_model
            and checkpoint_model[k].shape != state_dict.get(k, torch.empty(0)).shape
        ):
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate positional embeddings (RETFound util)
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # Re-init classifier head
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        trunc_normal_(model.head.weight, std=2e-5)
        if model.head.bias is not None:
            nn.init.zeros_(model.head.bias)

    model = model.to(device)

    # LR
    if args.lr is None:
        lr = args.blr * args.batch_size / 256.0
    else:
        lr = args.lr
    print(f"Using LR = {lr:.3e}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    min_lr_ratio = max(args.min_lr / lr, 0.0) if lr > 0 else 0.0
    lr_lambda = lambda ep: epoch_cosine_lambda(
        ep, args.warmup_epochs, args.epochs, min_lr_ratio
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Train
    best_acc = 0.0
    history_path = os.path.join(args.output_dir, "train_log.csv")
    if not os.path.exists(history_path):
        with open(history_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad(set_to_none=True)

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.fp16, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets) / args.accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % args.accum_steps == 0:
                if args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * args.accum_steps
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)

        # Validation
        model.eval()
        val_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.fp16, dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
                v_correct += (outputs.argmax(dim=1) == targets).sum().item()
                v_total += targets.size(0)

        val_loss = val_loss / max(1, len(val_loader))
        val_acc = 100.0 * v_correct / max(1, v_total)

        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f} lr={lr_now:.2e} time={dt:.1f}s"
        )

        # Save checkpoints
        last_path = os.path.join(args.output_dir, "last_model.pth")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "img_size": args.img_size,
                "global_pool": args.global_pool,
            },
            last_path,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "img_size": args.img_size,
                    "global_pool": args.global_pool,
                },
                best_path,
            )
            print(f"  -> Saved new best to {best_path} (val_acc={best_acc:.2f})")

        if args.save_every and ((epoch + 1) % args.save_every == 0):
            ep_path = os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}.pth")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "img_size": args.img_size,
                    "global_pool": args.global_pool,
                },
                ep_path,
            )

    print(f"Training complete. Best val_acc={best_acc:.2f}%")


if __name__ == "__main__":
    main()
