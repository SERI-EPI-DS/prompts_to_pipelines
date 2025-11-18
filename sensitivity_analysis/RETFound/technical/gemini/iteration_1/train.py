# /project/code/train.py

import os
import sys
import argparse
import traceback
import time
from functools import partial  # Import partial for the model definition

# --- Path Setup ---
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
retfound_path = os.path.join(main_path, "RETFound")
if main_path not in sys.path:
    sys.path.insert(0, main_path)
if retfound_path not in sys.path:
    sys.path.insert(0, retfound_path)

# --- Core Imports ---
from RETFound import models_vit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# --- ⭐ FIX: Import the missing utility function directly ---
try:
    # This function is needed to resize position embeddings when loading weights.
    from util.pos_embed import interpolate_pos_embed

    print("✅ Successfully imported 'interpolate_pos_embed' utility.")
except ImportError:
    print("❌ WARNING: Could not import 'interpolate_pos_embed' from 'util.pos_embed'.")
    print(
        "   If your fine-tuning resolution is different from the pre-training resolution, this may cause errors."
    )
    interpolate_pos_embed = None


# A custom Cross-Entropy loss function with label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_args_parser():
    parser = argparse.ArgumentParser(description="RETFound Fine-Tuning Training Script")
    parser.add_argument(
        "--data_path", required=True, type=str, help="Path to the root data directory"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Path to save model weights and logs",
    )
    parser.add_argument(
        "--retfound_weights",
        type=str,
        default="RETFound/RETFound_CFP_weights.pth",
        help="Path to the pre-trained RETFound weights relative to the main project directory",
    )

    # --- Training Hyperparameters ---
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    return parser


def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data Preparation ---
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "val"), transform=val_transform
    )

    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {', '.join(train_dataset.classes)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Model Loading ---
    print(
        "✅ Constructing ViT-Large model directly from the VisionTransformer class..."
    )
    model = models_vit.VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        global_pool=True,
    )

    weights_path = os.path.join(main_path, args.retfound_weights)
    checkpoint = torch.load(weights_path, map_location="cpu")
    print("Loading pre-trained weights from:", weights_path)

    checkpoint_model = checkpoint["model"]

    # --- ⭐ FIX: Call the directly imported function ---
    if interpolate_pos_embed is not None:
        interpolate_pos_embed(model, checkpoint_model)
    else:
        print(
            "Skipping positional embedding interpolation as the function was not found."
        )

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Weights loaded with status: {msg}")

    model.to(device)

    # --- Optimizer, Loss, and Scheduler ---
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    scaler = GradScaler()

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
    )

    # --- Training Loop ---
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_bar.set_postfix(loss=loss.item())

        if epoch >= args.warmup_epochs:
            lr_scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"
            )
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(
                f"✨ New best model saved to {best_model_path} with Val Acc: {best_val_acc:.4f}"
            )

    end_time = time.time()
    print(f"\nTotal training time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
