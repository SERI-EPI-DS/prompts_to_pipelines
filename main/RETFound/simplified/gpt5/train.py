#!/usr/bin/env python3
# --- RETFound train script (robust model builder) ---

import argparse, json, math, os, random, time, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Ensure we import the local repo (RETFound_MAE) version of models_vit
REPO_ROOT = Path(__file__).resolve().parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models_vit  # from the RETFound repo
from util.pos_embed import interpolate_pos_embed

from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
from huggingface_hub import hf_hub_download

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# -----------------------------
# Utilities
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


def resolve_pretrained(pretrained: str) -> str:
    """
    - local path to .pth/.pt OR
    - 'hf:RETFound_cfp_weights.pth' (auto-download from open-eye/RETFound_MAE)
    """
    if not pretrained:
        raise ValueError(
            "Please pass --pretrained (local .pth) or 'hf:RETFound_cfp_weights.pth'"
        )
    if pretrained.startswith("hf:"):
        filename = pretrained.split("hf:")[1]
        return hf_hub_download(
            repo_id="open-eye/RETFound_MAE", filename=filename
        )  # :contentReference[oaicite:1]{index=1}
    if os.path.isfile(pretrained):
        return pretrained
    raise FileNotFoundError(f"Pretrained weights not found: {pretrained}")


def build_transforms(input_size: int = 224):
    train_tf = transforms.Compose(
        [
            transforms.Resize(
                int(input_size * 1.14),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomResizedCrop(
                input_size,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(
                int(input_size * 1.14),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def get_datasets(data_root: Path, train_name: str, val_name: str, input_size: int):
    train_tf, eval_tf = build_transforms(input_size)
    train_ds = datasets.ImageFolder(root=data_root / train_name, transform=train_tf)
    val_ds = datasets.ImageFolder(root=data_root / val_name, transform=eval_tf)
    return train_ds, val_ds


# -----------------------------
# Robust RETFound model builder
# -----------------------------
def _make_vit_large_patch16_from_class(
    nb_classes: int, drop_path: float, global_pool: bool
):
    """
    Directly instantiate VisionTransformer with ViT-L/16 settings from official models_vit.py.
    Some repo variants don't expose vit_large_patch16() helper. :contentReference[oaicite:2]{index=2}
    """
    from functools import partial

    model = models_vit.VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=drop_path,
        global_pool=global_pool,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=nb_classes,
    )
    return model


def build_model(
    nb_classes: int, drop_path: float, global_pool: bool, pretrained_ckpt_path: str
):
    """
    Try helper (vit_large_patch16); if missing, fall back to constructing VisionTransformer directly.
    Then load RETFound MAE weights, interpolate pos-emb, and re-init the classification head.
    """
    fn = getattr(models_vit, "vit_large_patch16", None)
    if callable(fn):
        model = fn(
            num_classes=nb_classes, drop_path_rate=drop_path, global_pool=global_pool
        )
    else:
        print(
            "[WARN] models_vit.vit_large_patch16 not found; constructing VisionTransformer directly."
        )
        model = _make_vit_large_patch16_from_class(nb_classes, drop_path, global_pool)

    checkpoint = torch.load(pretrained_ckpt_path, map_location="cpu")
    ckpt_model = checkpoint.get(
        "model", checkpoint
    )  # some files store weights under 'model'

    # Remove incompatible classifier if shapes differ
    state_dict = model.state_dict()
    for k in ("head.weight", "head.bias"):
        if (
            k in ckpt_model
            and k in state_dict
            and ckpt_model[k].shape != state_dict[k].shape
        ):
            del ckpt_model[k]

    # Interpolate positional embeddings for safety
    interpolate_pos_embed(model, ckpt_model)

    _ = model.load_state_dict(ckpt_model, strict=False)

    # Freshly init the head
    nn.init.trunc_normal_(model.head.weight, std=2e-5)
    if getattr(model.head, "bias", None) is not None:
        nn.init.zeros_(model.head.bias)

    return model


# -----------------------------
# Train / Eval loops
# -----------------------------
def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, epoch, mixup_fn=None, ema=None
):
    model.train()
    running_loss, correct, n = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # --- NEW: guard against odd batch when mixup is enabled ---
        if (mixup_fn is not None) and (images.size(0) % 2 == 1):
            images = images[:-1]
            targets = targets[:-1]
        # -----------------------------------------------------------

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(True):
            if mixup_fn is not None:
                images, targets = mixup_fn(images, targets)
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * images.size(0)
        if mixup_fn is None:
            preds = outputs.argmax(1)
            correct += (preds == targets).sum().item()
            n += images.size(0)

    avg_loss = running_loss / len(loader.dataset)
    acc = (correct / n) if n > 0 else float("nan")
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device, nb_classes: int):
    model.eval()
    from sklearn.metrics import accuracy_score, roc_auc_score

    logits_all, targets_all = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits_all.append(model(images).cpu())
        targets_all.append(targets)
    logits = torch.cat(logits_all, 0)
    targets = torch.cat(targets_all, 0)
    probs = logits.softmax(1).numpy()
    y_true = targets.numpy()
    y_pred = probs.argmax(1)
    acc = float(accuracy_score(y_true, y_pred))
    try:
        auroc = (
            float(roc_auc_score(y_true, probs[:, 1]))
            if probs.shape[1] == 2
            else float(roc_auc_score(y_true, probs, multi_class="ovo"))
        )
    except Exception:
        auroc = float("nan")
    return acc, auroc


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root containing train/val/test subfolders.",
    )
    ap.add_argument("--train_subdir", type=str, default="train")
    ap.add_argument("--val_subdir", type=str, default="val")
    ap.add_argument("--test_subdir", type=str, default="test")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--nb_classes", type=int, default=None)
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--base_lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--drop_path", type=float, default=0.2)
    ap.add_argument("--layer_decay", type=float, default=0.65)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Local .pth or 'hf:RETFound_cfp_weights.pth'",
    )  # :contentReference[oaicite:3]{index=3}
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_ds, val_ds = get_datasets(
        data_root, args.train_subdir, args.val_subdir, args.input_size
    )
    nb_classes = args.nb_classes or len(train_ds.classes)

    # Save class mapping
    with open(out_dir / "class_mapping.json", "w") as f:
        json.dump(
            {"classes": train_ds.classes, "class_to_idx": train_ds.class_to_idx},
            f,
            indent=2,
        )

    use_mix = (args.mixup > 0.0) or (args.cutmix > 0.0)

    # --- UPDATED: drop_last=True when using Mixup/CutMix (prevents odd last batch) ---
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=use_mix,  # <--- key fix
    )
    # ---------------------------------------------------------------------------------

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # --- snip: model, optimizer, scheduler unchanged ---

    # Loss & Mixup
    if use_mix:
        from timm.data.mixup import Mixup
        from timm.loss import SoftTargetCrossEntropy

        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=1.0,
            label_smoothing=args.label_smoothing,
            num_classes=nb_classes,
        )
        criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Model
    ckpt_path = resolve_pretrained(args.pretrained)
    model = build_model(nb_classes, args.drop_path, True, ckpt_path).to(device)

    # LLRD param groups
    def make_param_groups_with_llrd(
        model: nn.Module, base_lr: float, layer_decay: float = 0.65
    ):
        num_layers = len(model.blocks) + 2

        def lr_scale(layer_id: int) -> float:
            return layer_decay ** (num_layers - layer_id - 1)

        layer_scales: Dict[str, float] = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith(("patch_embed", "pos_embed", "cls_token")):
                layer_scales[n] = lr_scale(0)
            elif n.startswith("blocks."):
                blk_id = int(n.split(".")[1]) + 1
                layer_scales[n] = lr_scale(blk_id)
            else:
                layer_scales[n] = lr_scale(num_layers - 1)
        scale_to_params: Dict[float, List[torch.nn.Parameter]] = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = layer_scales[n]
            scale_to_params.setdefault(s, []).append(p)
        return [
            {"params": ps, "lr": args.base_lr * s} for s, ps in scale_to_params.items()
        ]

    optimizer = optim.AdamW(
        make_param_groups_with_llrd(model, args.base_lr, args.layer_decay),
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=True)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        progress = (epoch - args.warmup_epochs) / float(
            max(1, args.epochs - args.warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    ema = ModelEmaV2(model, decay=0.9999) if args.ema else None

    # Logging
    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_acc,val_auroc,lr\n")

    best_auroc = -1.0
    best_path = out_dir / "checkpoint-best.pth"
    last_path = out_dir / "checkpoint-last.pth"

    print(f"Start training for {args.epochs} epochs. Classes: {train_ds.classes}.")
    t0 = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            mixup_fn,
            ema,
        )
        scheduler.step()
        eval_model = ema.module if ema is not None else model
        val_acc, val_auroc = evaluate(eval_model, val_loader, device, nb_classes)

        torch.save(
            {
                "epoch": epoch,
                "model": eval_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
                "classes": train_ds.classes,
            },
            last_path,
        )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(
                {
                    "epoch": epoch,
                    "model": eval_model.state_dict(),
                    "args": vars(args),
                    "classes": train_ds.classes,
                },
                best_path,
            )

        cur_lr = optimizer.param_groups[0]["lr"]
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_acc:.6f},{val_auroc:.6f},{cur_lr:.8f}\n"
            )

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} val_auroc={val_auroc:.3f} best_auroc={best_auroc:.3f}"
        )

    print(
        f"Done. Best AUROC={best_auroc:.4f}. Best ckpt: {best_path}. Time={(time.time()-t0)/60:.1f}m"
    )


if __name__ == "__main__":
    main()
