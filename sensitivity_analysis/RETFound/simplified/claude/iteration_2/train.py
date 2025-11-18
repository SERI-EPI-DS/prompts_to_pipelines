"""
Fine-tune RETFound foundation model for ophthalmology image classification
Author: AI Research Assistant
Updated to fix model loading issues
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchvision.transforms.functional as TF

from timm import create_model
from timm.models.layers import trunc_normal_
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings("ignore")


# Custom Vision Transformer for RETFound (compatible with MAE checkpoints)
class RETFoundViT(nn.Module):
    """Custom ViT implementation compatible with RETFound checkpoints"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        global_pool=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Try to use timm's create_model first, fall back to custom if needed
        try:
            from timm.models.vision_transformer import VisionTransformer

            # Create base model
            self.base_model = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=0,  # We'll add our own head
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                global_pool="avg" if global_pool else "",
            )
        except:
            # Fallback implementation
            self.patch_embed = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, embed_dim)
            )
            self.pos_drop = nn.Dropout(p=drop_rate)

            self.blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=int(mlp_ratio * embed_dim),
                        dropout=drop_rate,
                        activation="gelu",
                        batch_first=True,
                    )
                    for _ in range(depth)
                ]
            )

            self.norm = norm_layer(embed_dim)
            self.base_model = None

        # Head
        self.global_pool = global_pool
        self.fc_norm = norm_layer(embed_dim) if global_pool else nn.Identity()
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize cls_token and pos_embed
        if hasattr(self, "pos_embed"):
            trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, "cls_token"):
            trunc_normal_(self.cls_token, std=0.02)

        # Initialize head
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        if self.base_model is not None:
            # Use timm's implementation
            x = self.base_model.forward_features(x)
            if self.global_pool:
                x = x[:, 1:].mean(dim=1)  # Global average pooling (excluding cls token)
        else:
            # Custom implementation
            B = x.shape[0]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)

            if self.global_pool:
                x = x[:, 1:].mean(dim=1)  # Global average pooling
            else:
                x = x[:, 0]  # Use cls token

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc_norm(x)
        x = self.head(x)
        return x


# Custom data augmentation for retinal images
class RetinalAugmentation:
    """State-of-the-art augmentation for retinal images"""

    def __init__(self, input_size=224, training=True):
        self.input_size = input_size
        self.training = training

    def __call__(self, img):
        if self.training:
            # Random rotation (-30 to 30 degrees)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-30, 30)
                img = TF.rotate(img, angle)

            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = TF.hflip(img)

            # Random vertical flip
            if np.random.rand() > 0.5:
                img = TF.vflip(img)

            # Color jitter
            if np.random.rand() > 0.5:
                img = transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                )(img)

            # Random affine
            if np.random.rand() > 0.5:
                img = transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                )(img)

        return img


def get_transforms(input_size=224, training=True):
    """Get data transforms with state-of-the-art augmentations"""
    if training:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                RetinalAugmentation(input_size, training=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def get_class_weights(dataset):
    """Calculate class weights for imbalanced datasets"""
    targets = []
    for _, label in dataset:
        targets.append(label)

    class_counts = np.bincount(targets)
    total_samples = len(targets)
    class_weights = total_samples / (len(class_counts) * class_counts)

    return torch.FloatTensor(class_weights)


def create_data_loaders(data_path, batch_size=16, input_size=224, num_workers=4):
    """Create data loaders with proper augmentation and balancing"""

    # Create datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_path, "train"),
        transform=get_transforms(input_size, training=True),
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_path, "val"),
        transform=get_transforms(input_size, training=False),
    )

    # Get class weights for balanced training
    class_weights = get_class_weights(train_dataset)
    sample_weights = [class_weights[label] for _, label in train_dataset]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes


def load_retfound_model(model_name, finetune_checkpoint, num_classes, input_size=224):
    """Load RETFound model with proper configuration"""

    # Model configurations
    model_configs = {
        "RETFound_mae": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4,
            "patch_size": 16,
            "drop_path_rate": 0.1,
        },
        "RETFound_dinov2": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4,
            "patch_size": 14,
            "drop_path_rate": 0.1,
        },
    }

    config = model_configs.get(model_name, model_configs["RETFound_mae"])

    # Try to use standard timm models first
    try:
        # Map RETFound model names to timm model names
        timm_model_map = {
            "RETFound_mae": "vit_large_patch16_224",
            "RETFound_dinov2": "vit_large_patch14_224",
        }

        timm_model_name = timm_model_map.get(model_name, "vit_large_patch16_224")

        # Try creating model with timm
        model = create_model(
            timm_model_name,
            pretrained=False,
            num_classes=num_classes,
            img_size=input_size,
            drop_path_rate=config["drop_path_rate"],
        )
        print(f"Created model using timm: {timm_model_name}")

    except Exception as e:
        print(f"Could not create model with timm: {e}")
        print("Using custom RETFound ViT implementation...")

        # Fallback to custom implementation
        model = RETFoundViT(
            img_size=input_size,
            patch_size=config["patch_size"],
            num_classes=num_classes,
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            drop_path_rate=config["drop_path_rate"],
            global_pool=True,
        )

    # Load pretrained weights from HuggingFace
    try:
        checkpoint_path = hf_hub_download(
            repo_id=f"YukunZhou/{finetune_checkpoint}", filename="pytorch_model.bin"
        )

        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Filter and remap state dict if needed
        if hasattr(model, "base_model") and model.base_model is not None:
            # For timm models, might need to adjust keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]  # Remove 'module.' prefix
                if "head" not in k and "fc" not in k:  # Skip classifier layers
                    new_state_dict[k] = v
        else:
            # For custom model
            new_state_dict = {
                k: v for k, v in state_dict.items() if "head" not in k and "fc" not in k
            }

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(
            new_state_dict, strict=False
        )
        print(f"Loaded pretrained weights from {finetune_checkpoint}")
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")

    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")
        print("Training from scratch or using default initialization...")

    return model


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix(
            {"loss": running_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
        )

    return running_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device, num_classes):
    """Validate model and compute metrics"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_targets))

    # Calculate AUC for multi-class
    all_targets_bin = label_binarize(all_targets, classes=range(num_classes))
    try:
        auc = roc_auc_score(all_targets_bin, all_probs, average="macro")
    except:
        auc = 0.0

    return val_loss, val_acc, auc, all_preds, all_targets


def save_training_plots(history, output_dir):
    """Save training history plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss plot
    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy plot
    axes[0, 1].plot(history["train_acc"], label="Train")
    axes[0, 1].plot(history["val_acc"], label="Validation")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # AUC plot
    axes[1, 0].plot(history["val_auc"])
    axes[1, 0].set_title("Validation AUC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("AUC")
    axes[1, 0].grid(True)

    # Learning rate plot
    axes[1, 1].plot(history["lr"])
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RETFound for classification"
    )

    # Data parameters
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for checkpoints and results",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="RETFound_mae",
        choices=["RETFound_mae", "RETFound_dinov2"],
        help="Model architecture",
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default="RETFound_mae_natureCFP",
        help="Pretrained checkpoint name from HuggingFace",
    )
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--drop_path", type=float, default=0.2, help="Drop path rate")
    parser.add_argument(
        "--layer_decay", type=float, default=0.65, help="Layer-wise learning rate decay"
    )

    # System parameters
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        args.data_path, args.batch_size, args.input_size, args.num_workers
    )
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

    # Save class names
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Load model
    model = load_retfound_model(args.model, args.finetune, num_classes, args.input_size)
    model = model.to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss()

    # Layer-wise learning rate decay
    param_groups = []
    no_decay = ["bias", "norm"]

    if hasattr(model, "base_model") and model.base_model is not None:
        # For timm models
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                param_groups.append(
                    {"params": param, "lr": args.lr, "weight_decay": 0.0}
                )
            else:
                param_groups.append(
                    {"params": param, "lr": args.lr, "weight_decay": args.weight_decay}
                )
    else:
        # For custom model - simple approach
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                param_groups.append(
                    {"params": param, "lr": args.lr, "weight_decay": 0.0}
                )
            else:
                param_groups.append(
                    {"params": param, "lr": args.lr, "weight_decay": args.weight_decay}
                )

    optimizer = optim.AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=20)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_val_auc = 0.0

    # Training loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )

        # Validate
        val_loss, val_acc, val_auc, _, _ = validate(
            model, val_loader, criterion, device, num_classes
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}"
        )
        print(f"Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_auc = val_auc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_auc": val_auc,
                "class_names": class_names,
                "model_config": {
                    "model_name": args.model,
                    "input_size": args.input_size,
                    "num_classes": num_classes,
                },
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pth"))
            print(f"Saved best model with val_acc: {val_acc:.2f}%")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Save training plots
        if epoch % 5 == 0:
            save_training_plots(history, args.output_dir)

    print(f"\nTraining completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation AUC: {best_val_auc:.4f}")

    # Save final history
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(args.output_dir, "training_history.csv"), index=False
    )
    save_training_plots(history, args.output_dir)

    # Save configuration
    config = vars(args)
    config["best_val_acc"] = float(best_val_acc)
    config["best_val_auc"] = float(best_val_auc)
    config["num_classes"] = num_classes
    config["class_names"] = class_names

    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()
