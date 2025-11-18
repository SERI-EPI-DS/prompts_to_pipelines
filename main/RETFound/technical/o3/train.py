#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune the RETFound ViT-L MAE foundation model (robust head detection).
"""
import argparse, json, math, sys, importlib
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms


# -------------------------------------------------------------- #
# helper: import the local RETFound repo                         #
# -------------------------------------------------------------- #
def import_models_vit(retfound_dir: Path):
    parent = retfound_dir.parent.resolve()
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    return importlib.import_module(f"{retfound_dir.name}.models_vit")


# -------------------------------------------------------------- #
# helper: robust model builder                                   #
# -------------------------------------------------------------- #
def build_model(num_classes: int, ckpt: Path, retfound_dir: Path):
    mv = import_models_vit(retfound_dir)

    # choose constructor available in this fork
    for cand in ("vit_large_patch16", "RETFound_mae"):
        if hasattr(mv, cand):
            model = getattr(mv, cand)()
            break
    else:
        raise AttributeError("No ViT-L constructor found in models_vit.py")

    # load MAE pre-training weights (ignore head mismatch)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)

    # ---- locate existing classification head cleanly ----
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):  # timm style
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    elif hasattr(model, "heads") and hasattr(model.heads, "head"):  # torchvision style
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise AttributeError(
            "Cannot find a replaceable classification layer "
            "(looked for .head or .heads.head)"
        )

    nn.init.zeros_(
        model.head.weight if hasattr(model, "head") else model.heads.head.weight
    )
    return model


# -------------------------------------------------------------- #
# data transforms                                                #
# -------------------------------------------------------------- #
def tfm(train=True):
    a = (
        [
            transforms.RandomResizedCrop(224, (0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ]
        if train
        else [transforms.Resize(256), transforms.CenterCrop(224)]
    )
    a += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(a)


# -------------------------------------------------------------- #
# main                                                           #
# -------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--retfound_dir", type=Path, default=Path("../RETFound"))
    p.add_argument(
        "--weights", type=Path, default=Path("../RETFound/RETFound_CFP_weights.pth")
    )
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--accum_iter", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.ImageFolder(args.data_dir / "train", tfm(True))
    val_ds = datasets.ImageFolder(args.data_dir / "val", tfm(False))
    train_dl = DataLoader(
        train_ds, args.batch_size, True, num_workers=args.workers, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, args.batch_size, False, num_workers=args.workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(train_ds.classes), args.weights, args.retfound_dir).to(
        device
    )

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    iters = math.ceil(len(train_dl) / args.accum_iter)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * iters, 1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    best = 0

    for ep in range(args.epochs):
        # ---------------- train ----------------
        model.train()
        ls = acc = 0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            with autocast():
                o = model(x)
                loss = crit(o, y) / args.accum_iter
            scaler.scale(loss).backward()
            if (i + 1) % args.accum_iter == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()
            ls += loss.item() * args.accum_iter
            acc += (o.argmax(1) == y).sum().item()
        tr_acc = acc / len(train_ds)

        # ---------------- val ------------------
        model.eval()
        v_acc = v_ls = 0
        with torch.no_grad(), autocast():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                o = model(x)
                v_ls += crit(o, y).item()
                v_acc += (o.argmax(1) == y).sum().item()
        v_acc /= len(val_ds)
        v_ls /= len(val_dl)

        log = dict(
            epoch=ep + 1,
            train_loss=ls / len(train_dl),
            train_acc=tr_acc,
            val_loss=v_ls,
            val_acc=v_acc,
            lr=scheduler.get_last_lr()[0],
        )
        print(json.dumps(log))
        (args.out_dir / "train_log.jsonl").open("a").write(json.dumps(log) + "\n")

        if v_acc > best:
            best = v_acc
            torch.save(model.state_dict(), args.out_dir / "best_model.pth")
    print(f"Done — best val-acc {best:.4f}")


if __name__ == "__main__":
    main()
