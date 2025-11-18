#!/usr/bin/env python3
# train_v2.py — RETFound ViT-L trainer (fixed)
# - Avoid double pooling: override forward() to bypass timm.forward_head
# - Pop img_size in vit_large_patch16 to avoid double-pass bug

import argparse, os, json, time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from sklearn.metrics import accuracy_score, roc_auc_score

    HAVE_SK = True
except Exception:
    HAVE_SK = False

# -------------------- ViT-L (RETFound-style) --------------------
import timm.models.vision_transformer as tvit


class VisionTransformer(tvit.VisionTransformer):
    """Vision Transformer with global avg pool + fc_norm (RETFound style)."""

    def __init__(self, global_pool=True, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs.get("norm_layer", partial(nn.LayerNorm, eps=1e-6))
            embed_dim = kwargs.get("embed_dim")
            self.fc_norm = norm_layer(embed_dim)
            # RETFound deletes the original norm and uses fc_norm after pooling
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


# -------------------- Data utils --------------------
def build_transforms(input_size=224, eval_resize=256):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(eval_resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def build_loaders(data_dir, batch_size, num_workers, input_size):
    train_tf, eval_tf = build_transforms(input_size)
    ds_train = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    ds_val = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tf)
    ds_test = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_tf)
    ld_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    ld_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    ld_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return ds_train, ds_val, ds_test, ld_train, ld_val, ld_test


# -------------------- Load RETFound weights --------------------
def load_retfound_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model = ckpt.get("model", ckpt)
    state_dict = model.state_dict()
    # Drop classifier head if shapes mismatch
    for k in ["head.weight", "head.bias"]:
        if (
            k in checkpoint_model
            and k in state_dict
            and checkpoint_model[k].shape != state_dict[k].shape
        ):
            checkpoint_model.pop(k)
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    # Init new head
    if "head.weight" in state_dict:
        nn.init.trunc_normal_(model.head.weight, std=2e-5)
        if model.head.bias is not None:
            nn.init.zeros_(model.head.bias)
    return msg


# -------------------- Eval --------------------
def evaluate(model, loader, device, nb_classes):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probs = logits.softmax(dim=1)
    preds = probs.argmax(dim=1)
    if HAVE_SK:
        acc = accuracy_score(targets.numpy(), preds.numpy())
        try:
            auc = roc_auc_score(
                targets.numpy(),
                probs[:, 1].numpy() if probs.shape[1] == 2 else probs.numpy(),
                multi_class="ovr" if probs.shape[1] > 2 else "raise",
            )
        except Exception:
            auc = None
    else:
        acc = float((preds == targets).float().mean())
        auc = None
    return acc, auc


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument(
        "--weights",
        required=True,
        type=str,
        help="Path to RETFound MAE .pth (Hugging Face).",
    )
    ap.add_argument("--nb_classes", required=True, type=int)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--drop_path", type=float, default=0.2)
    ap.add_argument("--freeze_backbone", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train, ds_val, ds_test, ld_train, ld_val, ld_test = build_loaders(
        args.data_dir, args.batch_size, args.workers, args.input_size
    )
    class_names = ds_train.classes
    assert (
        len(class_names) == args.nb_classes
    ), f"nb_classes={args.nb_classes} but found {len(class_names)} folders in train/."

    model = vit_large_patch16(
        global_pool=True,
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
    ).to(device)

    load_retfound_weights(model, args.weights)

    if args.freeze_backbone:
        for n, p in model.named_parameters():
            if not n.startswith("head."):
                p.requires_grad = False

    base_lr = args.lr * (args.batch_size / 256.0)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss()

    best_auc, best_acc = -1.0, -1.0
    best_path = os.path.join(args.output_dir, "checkpoint-best.pth")

    for epoch in range(args.epochs):
        model.train()
        running_loss, n_seen = 0.0, 0
        t0 = time.time()
        for imgs, targets in ld_train:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
        scheduler.step()

        val_acc, val_auc = evaluate(model, ld_val, device, args.nb_classes)
        is_better = (val_auc is not None and val_auc > best_auc) or (
            val_auc is None and val_acc > best_acc
        )
        if is_better:
            best_auc = max(best_auc, val_auc if val_auc is not None else -1)
            best_acc = max(best_acc, val_acc)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "args": vars(args),
                },
                best_path,
            )

        print(
            f"[{epoch+1:03d}/{args.epochs}] "
            f"loss={running_loss/max(1,n_seen):.4f}  val_acc={val_acc:.4f}  "
            f"val_auc={(val_auc if val_auc is not None else float('nan')):.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  time={time.time()-t0:.1f}s"
        )

    # Quick test with best checkpoint
    if os.path.isfile(best_path):
        ck = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ck["model"])
    test_acc, test_auc = evaluate(model, ld_test, device, args.nb_classes)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "test_acc": test_acc,
                "test_auc": test_auc,
                "best_val_auc": best_auc,
                "best_val_acc": best_acc,
                "class_names": class_names,
            },
            f,
            indent=2,
        )
    print("==> Test:", {"acc": test_acc, "auc": test_auc})


if __name__ == "__main__":
    main()
