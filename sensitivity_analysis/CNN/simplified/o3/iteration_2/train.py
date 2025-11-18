#!/usr/bin/env python3
"""
Fine-tune ConvNeXt-Large on colour fundus photographs.
Author: YOUR_NAME, 2025-07-09
"""
import argparse, json, os, time
from pathlib import Path

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root folder containing train/val/test sub-folders.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./runs/exp"),
        help="Where to save checkpoints and logs.",
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument(
        "--patience", type=int, default=4, help="Early-stop patience on val loss."
    )
    return p.parse_args()


# ---------------- utils ----------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_dataloaders(root: Path, img_size: int, bs: int, workers: int):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_ds = datasets.ImageFolder(root / "train", train_tf)
    val_ds = datasets.ImageFolder(root / "val", val_tf)

    train_dl = DataLoader(
        train_ds, bs, shuffle=True, num_workers=workers, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, bs * 2, shuffle=False, num_workers=workers, pin_memory=True
    )
    return train_dl, val_dl, train_ds.classes


# ---------------- Trainer ----------------
class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scaler,
        train_dl,
        val_dl,
        classes,
        img_size,
        device,
        out_dir: Path,
        patience: int,
    ):
        self.model, self.criterion, self.opt, self.scaler = (
            model,
            criterion,
            optimizer,
            scaler,
        )
        self.train_dl, self.val_dl = train_dl, val_dl
        self.classes, self.img_size = classes, img_size
        self.device, self.out_dir, self.patience = device, out_dir, patience
        out_dir.mkdir(parents=True, exist_ok=True)

        self.metric_acc = MulticlassAccuracy(num_classes=len(classes)).to(device)
        self.metric_auc = MulticlassAUROC(num_classes=len(classes), average="macro").to(
            device
        )
        self.best_acc, self.no_improve = 0.0, 0

    def _run_epoch(self, train: bool):
        dl = self.train_dl if train else self.val_dl
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        self.metric_acc.reset()
        self.metric_auc.reset()

        if train:
            self.opt.zero_grad(set_to_none=True)
        for imgs, labels in dl:
            imgs, labels = imgs.to(self.device, non_blocking=True), labels.to(
                self.device
            )

            with autocast():
                preds = self.model(imgs)
                loss = self.criterion(preds, labels)

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            self.metric_acc.update(preds, labels)
            self.metric_auc.update(preds.softmax(-1), labels)

        n = len(dl.dataset)
        return (
            total_loss / n,
            self.metric_acc.compute().item(),
            self.metric_auc.compute().item(),
        )

    def fit(self, epochs: int):
        log_path = self.out_dir / "log.jsonl"
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_auc = self._run_epoch(train=True)
            with torch.no_grad():
                val_loss, val_acc, val_auc = self._run_epoch(train=False)
            elapsed = time.time() - t0

            log = dict(
                epoch=ep,
                train_loss=tr_loss,
                val_loss=val_loss,
                train_acc=tr_acc,
                val_acc=val_acc,
                train_auc=tr_auc,
                val_auc=val_auc,
                secs=elapsed,
            )
            print(json.dumps(log))
            with open(log_path, "a") as f:
                f.write(json.dumps(log) + "\n")

            # save best checkpoint
            if val_acc > self.best_acc:
                self.best_acc, self.no_improve = val_acc, 0
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "classes": self.classes,
                        "img_size": self.img_size,
                    },
                    self.out_dir / "best.pth",
                )
            else:
                self.no_improve += 1

            if self.no_improve >= self.patience:
                print(f"Early stopping after {ep} epochs without improvement.")
                break


# ---------------- main ----------------
def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, val_dl, class_names = build_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers
    )

    model = timm.create_model(
        "convnext_large_in22k", pretrained=True, num_classes=len(class_names)
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler()

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        scaler,
        train_dl,
        val_dl,
        class_names,
        args.img_size,
        device,
        args.out_dir,
        args.patience,
    )
    trainer.fit(args.epochs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
