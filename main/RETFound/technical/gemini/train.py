import argparse
import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from functools import partial

# Add RETFound repository to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RETFound"))
)

# FIX: Import the base VisionTransformer class directly to avoid issues with helper functions
from models_vit import VisionTransformer


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_args_parser():
    parser = argparse.ArgumentParser(description="RETFound Finetuning")
    parser.add_argument(
        "--data_path", required=True, type=str, help="Path to the root data directory."
    )
    parser.add_argument(
        "--output_dir", required=True, type=str, help="Path to save results."
    )
    parser.add_argument(
        "--retfound_weights",
        default="../../RETFound/RETFound_CFP_weights.pth",
        type=str,
        help="Path to RETFound pretrained weights.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--epochs", default=50, type=int, help="Maximum number of training epochs."
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", default=0.05, type=float, help="Weight decay."
    )
    parser.add_argument(
        "--label_smoothing", default=0.1, type=float, help="Label smoothing factor."
    )
    parser.add_argument(
        "--patience", default=10, type=int, help="Patience for early stopping."
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of data loading workers."
    )
    return parser


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available. Training on CPU.")

    # --- Data Preparation ---
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f"Found {num_classes} classes: {class_names}")

    # --- Model Loading ---
    # FIX: Manually define and instantiate the ViT-Large model using the VisionTransformer base class.
    # This is the most robust method and avoids dependency on potentially missing helper functions.
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
    )

    # Load RETFound pre-trained weights
    checkpoint = torch.load(args.retfound_weights, map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in checkpoint["model"].items()
        if k in model_dict and "head" not in k
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.to(device)

    # --- Training Setup ---
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    # --- Training Loop ---
    best_val_loss = float("inf")
    epochs_no_improve = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_pbar.set_postfix(
                    {
                        "val_loss": f"{loss.item():.4f}",
                        "acc": f"{100 * correct / total:.2f}%",
                    }
                )

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(
            f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.2f}%"
        )

        # --- Save Best Model and Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, "best_model.pth")
            )
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # --- Save Class Names ---
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)
    print("Training complete.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
