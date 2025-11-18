##############################################
# ConvNeXt-Large Fine‑Tuning for Fundus Images
# Two standalone scripts in one canvas
# 1. train_convnext.py – trains & saves a model
# 2. test_convnext.py  – evaluates a saved model
##############################################

# ------------------------------------------------
# === train_convnext.py ===
# Usage example:
#   python train_convnext.py \
#          --data_dir "/path/to/dataset" \
#          --output_dir "./results" \
#          --epochs 30 --batch_size 32
# ------------------------------------------------

import argparse
import os
from pathlib import Path
import time
import copy
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine‑tune ConvNeXt‑L on colour fundus photographs"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root folder with train/val/test subfolders (ImageFolder layout)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save checkpoints & logs",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(data_dir: str, batch_size: int, num_workers: int = 8):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    image_datasets = {
        "train": datasets.ImageFolder(Path(data_dir) / "train", transform=train_tfm),
        "val": datasets.ImageFolder(Path(data_dir) / "val", transform=val_tfm),
    }

    dataloaders = {
        phase: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(phase == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        for phase, ds in image_datasets.items()
    }
    return dataloaders, image_datasets["train"].classes


def build_model(num_classes: int):
    model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def train_model(model, dataloaders, device, epochs, base_lr, weight_decay, output_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_history = []

    for epoch in range(epochs):
        epoch_start = time.time()
        stats = {"epoch": epoch + 1}
        print(f"Epoch {epoch+1}/{epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            n_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )

                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                n_samples += inputs.size(0)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / n_samples
            epoch_acc = running_corrects / n_samples
            stats[f"{phase}_loss"] = epoch_loss
            stats[f"{phase}_acc"] = epoch_acc
            print(f"  {phase}: loss {epoch_loss:.4f} acc {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        epoch_time = time.time() - epoch_start
        stats["time_sec"] = epoch_time
        train_history.append(stats)
        # save checkpoint every epoch
        ckpt_path = Path(output_dir) / f"epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_acc,
                "classes": dataloaders["train"].dataset.classes,
            },
            ckpt_path,
        )
        print(f"  Saved checkpoint to {ckpt_path}")

    # training done
    model.load_state_dict(best_model_wts)
    final_path = Path(output_dir) / "best_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Best model saved to {final_path} with val acc {best_acc:.4f}")

    # save training log
    with open(Path(output_dir) / "training_log.json", "w") as f:
        json.dump(train_history, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, class_names = create_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    model = build_model(num_classes=len(class_names)).to(device)

    train_model(
        model,
        dataloaders,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.output_dir,
    )
