#!/usr/bin/env python3
import argparse, json, random, time
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models.convnext import (
    convnext_large,
    ConvNeXt_Large_Weights as Weights,
)
from torchvision.transforms.functional import InterpolationMode

# -----------------------------
# Utilities
# -----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    num_classes: int, pretrained: bool = True, drop_path_rate: float = 0.2
) -> nn.Module:
    weights = Weights.DEFAULT if pretrained else None
    model = convnext_large(weights=weights, drop_path_rate=drop_path_rate)
    # Replace final Linear robustly
    if isinstance(model.classifier, nn.Sequential):
        for i in reversed(range(len(model.classifier))):
            if isinstance(model.classifier[i], nn.Linear):
                in_features = model.classifier[i].in_features
                model.classifier[i] = nn.Linear(in_features, num_classes)
                break
        else:
            raise RuntimeError(
                "Could not locate final Linear layer in model.classifier"
            )
    else:
        raise RuntimeError("Unexpected classifier type; update replacement logic.")
    return model


def get_transforms(img_size: int, use_autoaugment: bool, mean, std):
    train_tfms = [
        T.RandomResizedCrop(
            img_size,
            scale=(0.7, 1.0),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10, interpolation=InterpolationMode.BICUBIC),
    ]
    if use_autoaugment:
        train_tfms.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET))
    else:
        train_tfms.append(T.RandAugment(num_ops=2, magnitude=9))
    train_tfms += [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        T.RandomErasing(p=0.1, scale=(0.02, 0.2), value="random"),
    ]

    eval_tfms = T.Compose(
        [
            T.Resize(
                int(img_size * 1.15),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return T.Compose(train_tfms), eval_tfms


def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def mixup_data(x, y, num_classes, alpha=0.2):
    if alpha <= 0.0:
        return x, F.one_hot(y, num_classes=num_classes).float(), 1.0
    lam = (
        torch._sample_dirichlet(torch.tensor([alpha, alpha], device=x.device))[0]
        .max()
        .item()
    )
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index, :]
    return mixed_x, mixed_y, lam


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def compute_class_weights(targets: List[int], num_classes: int) -> torch.Tensor:
    import numpy as np

    counts = np.bincount(targets, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)


def _uniform_param_stats(params):
    devices = {p.device for p in params}
    dtypes = {p.dtype for p in params}
    layouts = {p.layout for p in params}
    return devices, dtypes, layouts


def _can_use_fused_adamw(params) -> bool:
    """Allow fused AdamW only if CUDA + single dtype/device/layout across trainable params and dtype FP32."""
    if not torch.cuda.is_available():
        return False
    devices, dtypes, layouts = _uniform_param_stats(params)
    if len(devices) != 1 or list(devices)[0].type != "cuda":
        return False
    if len(dtypes) != 1 or list(dtypes)[0] != torch.float32:
        return False
    if len(layouts) != 1 or list(layouts)[0] != torch.strided:
        return False
    return True


# -----------------------------
# Train / Validate
# -----------------------------
def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    num_classes,
    label_smoothing,
    mixup_alpha,
    grad_clip,
    ema: Optional[EMA],
):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )

        # Mixup -> soft labels; else label smoothing
        if mixup_alpha > 0.0:
            imgs, soft_targets, _ = mixup_data(
                imgs, labels, num_classes, alpha=mixup_alpha
            )
        else:
            eps = label_smoothing
            soft_targets = F.one_hot(labels, num_classes=num_classes).float()
            if eps > 0.0:
                soft_targets = soft_targets * (1 - eps) + eps / num_classes

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = soft_cross_entropy(logits, soft_targets)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Safe step: if fused kernel raises dtype/device/layout error, fall back once.
        try:
            scaler.step(optimizer)
        except RuntimeError as e:
            msg = str(e)
            if (
                "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout"
                in msg
                or "_fused_adamw" in msg
            ):
                print(
                    "[AdamW] Fused kernel failed at runtime; falling back to non-fused AdamW and retrying this step."
                )
                # Rebuild optimizer as non-fused, foreach=True; keep hyperparams
                new_opt = AdamW(
                    optimizer.param_groups[0]["params"],
                    lr=optimizer.param_groups[0]["lr"],
                    weight_decay=optimizer.param_groups[0]["weight_decay"],
                    fused=False,
                    foreach=True,
                )
                # Swap the optimizer object
                optimizer.__dict__.update(new_opt.__dict__)
                scaler.step(optimizer)  # retry
            else:
                raise
        scaler.update()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * imgs.size(0)
        running_acc += accuracy(logits.detach(), labels) * imgs.size(0)
        n += imgs.size(0)

    return running_loss / n, running_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n, loss_sum, acc_sum = 0, 0.0, 0.0
    ce = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = ce(logits, labels)
        loss_sum += loss.item() * imgs.size(0)
        acc_sum += accuracy(logits, labels) * imgs.size(0)
        n += imgs.size(0)
    return loss_sum / n, acc_sum / n


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ConvNeXt-L on fundus images")
    p.add_argument(
        "--data_root", type=str, required=True, help="Root containing train/val folders"
    )
    p.add_argument("--train_dir", type=str, default="train")
    p.add_argument("--val_dir", type=str, default="val")
    p.add_argument("--out_dir", type=str, default="./outputs/convnextL")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument(
        "--mixup_alpha", type=float, default=0.0, help=">0 enables mixup (e.g., 0.2)"
    )
    p.add_argument("--ema", action="store_true", help="Enable EMA of weights")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument(
        "--autoaugment",
        action="store_true",
        help="Use AutoAugment policy (else RandAugment)",
    )
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone, train classifier head only",
    )
    p.add_argument(
        "--balance_sampler",
        action="store_true",
        help="Use WeightedRandomSampler for class imbalance",
    )
    p.add_argument("--compile", action="store_true", help="Use torch.compile for speed")
    # Optimizer toggles
    p.add_argument(
        "--use_fused",
        action="store_true",
        help="Try fused AdamW (only if preflight checks pass)",
    )
    p.add_argument(
        "--no_foreach",
        action="store_true",
        help="Disable foreach (defaults to on when not fused)",
    )
    p.add_argument(
        "--force_fp32_params",
        action="store_true",
        help="Force all model params to FP32 to avoid hidden dtype mismatches",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stats from weights meta (fallback to ImageNet if missing)
    w = Weights.DEFAULT
    mean = tuple(w.meta.get("mean", IMAGENET_MEAN))
    std = tuple(w.meta.get("std", IMAGENET_STD))

    train_tfms, eval_tfms = get_transforms(args.img_size, args.autoaugment, mean, std)

    # Datasets
    train_path = Path(args.data_root) / args.train_dir
    val_path = Path(args.data_root) / args.val_dir

    train_ds = ImageFolder(str(train_path), transform=train_tfms)
    val_ds = ImageFolder(str(val_path), transform=eval_tfms)

    num_classes = len(train_ds.classes)
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Sampler (optional)
    if args.balance_sampler:
        targets = train_ds.targets
        class_weights = compute_class_weights(targets, num_classes)
        sample_weights = class_weights[torch.tensor(targets)]
        sampler = WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
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

    # Model
    model = build_model(num_classes=num_classes, pretrained=True, drop_path_rate=0.2)
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    model.to(device)

    # Force parameters to FP32 if requested (helps avoid dtype heterogeneity with AMP)
    if args.force_fp32_params:
        model = model.float()

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    # Optimizer: by default use non-fused AdamW with foreach=True
    params = [p for p in model.parameters() if p.requires_grad]
    devices, dtypes, layouts = _uniform_param_stats(params)

    use_fused = False
    fused_reason = ""
    if args.use_fused:
        if _can_use_fused_adamw(params):
            use_fused = True
        else:
            fused_reason = f"preflight failed (devices={devices}, dtypes={dtypes}, layouts={layouts})"

    foreach_flag = (not use_fused) and (
        not args.no_foreach
    )  # never True together with fused
    if use_fused:
        print("[AdamW] Using fused=True, foreach=False.")
    else:
        why = "" if fused_reason == "" else f" (Reason: {fused_reason})"
        print(f"[AdamW] Using fused=False, foreach={foreach_flag}.{why}")

    optimizer = AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=use_fused,
        foreach=foreach_flag,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    # Training loop
    history = []
    best_acc = 0.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            num_classes,
            args.label_smoothing,
            args.mixup_alpha,
            args.grad_clip,
            ema,
        )

        # Evaluate (EMA weights if enabled)
        if ema is not None:
            ema.apply_shadow(model)
        val_loss, val_acc = evaluate(model, val_loader, device)
        if ema is not None:
            ema.restore(model)

        scheduler.step()

        elapsed = time.time() - t0
        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "time_sec": round(elapsed, 2),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(log)
        print(
            f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} time={elapsed:.1f}s"
        )

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "model_name": "convnext_large",
                "state_dict": (
                    model.state_dict()
                    if not isinstance(model, nn.DataParallel)
                    else model.module.state_dict()
                ),
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "img_size": args.img_size,
                "mean": mean,
                "std": std,
                "args": vars(args),
                "best_val_acc": best_acc,
                "epoch": epoch,
            }
            torch.save(ckpt, best_path)
            print(f"  ↳ Saved new best to {best_path} (val_acc={best_acc:.4f})")

        # Always save a “last” checkpoint
        last_path = out_dir / "last.pt"
        torch.save(
            {
                "model_name": "convnext_large",
                "state_dict": (
                    model.state_dict()
                    if not isinstance(model, nn.DataParallel)
                    else model.module.state_dict()
                ),
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "img_size": args.img_size,
                "mean": mean,
                "std": std,
                "args": vars(args),
                "epoch": epoch,
            },
            last_path,
        )

        # Append to CSV history
        hist_path = out_dir / "history.csv"
        if len(history) == 1:
            with open(hist_path, "w") as f:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc,time_sec,lr\n")
        with open(hist_path, "a") as f:
            f.write(
                "{epoch},{train_loss},{train_acc},{val_loss},{val_acc},{time_sec},{lr}\n".format(
                    **log
                )
            )

    # Save label names and mapping
    with open(out_dir / "classes.txt", "w") as f:
        for c in train_ds.classes:
            f.write(f"{c}\n")
    with open(out_dir / "class_index.json", "w") as f:
        json.dump(idx_to_class, f, indent=2)

    print(
        f"Training complete. Best val acc: {best_acc:.4f}. Best checkpoint: {best_path}"
    )


if __name__ == "__main__":
    main()
