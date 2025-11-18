"""
Train a classifier by fine-tuning a RETFound foundation model on a custom
retinal disease dataset.

This script expects your data to be organised as follows:

```
dataset_root/
    train/
        class_0/
            img1.png
            img2.jpg
            ...
        class_1/
            ...
        ...
    val/
        class_0/
            ...
        class_1/
            ...
        ...
    test/
        class_0/
            ...
        class_1/
            ...
        ...
```

The script will fine‑tune the RETFound model on the training set, monitor
performance on the validation set and periodically evaluate on the test set.
It saves the best performing checkpoint (based on validation accuracy) and
produces a CSV file with predictions for the test set.  The model
architecture and pre‑trained weights are loaded from the HuggingFace Hub.
If you do not have access to the HuggingFace weights you can specify
a local `.pth` file via `--pretrained_path`.

Example usage:

```sh
python train_classifier.py \
    --data_dir /path/to/dataset_root \
    --num_classes 5 \
    --model_name RETFound_mae_natureCFP \
    --epochs 50 \
    --batch_size 32 \
    --output_dir ./experiments/retfound_messidor2
```

Notes:
* You must log in to HuggingFace (e.g. via `huggingface-cli login`) prior
  to running this script if you intend to download the weights from the
  Hub.
* The default model uses images resized to 224×224 pixels.  You may
  increase `--input_size` but note that positional embeddings will be
  automatically interpolated to match the new resolution.
"""

import argparse
import csv
import datetime
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    # HuggingFace Hub is used to fetch pretrained weights.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    hf_hub_download = None  # type: ignore


def build_transforms(
    input_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create training and validation/test transforms.

    Args:
        input_size: Target input resolution of the model.

    Returns:
        A tuple containing the training transform and the eval transform.
    """
    # Mean and standard deviation from ImageNet statistics.  RETFound models
    # were pre‑trained on retinal images but reusing ImageNet normalisation
    # has been shown to work well when fine‑tuning ViT models.
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                int(input_size * 256 / 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomResizedCrop(
                input_size,
                scale=(0.8, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                int(input_size * 256 / 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_transform, eval_transform


class VisionTransformer(nn.Module):
    """Vision Transformer with optional global average pooling.

    This implementation closely follows the RETFound repository.  We
    subclass `timm`'s VisionTransformer to provide global pooling and
    expose a classification head.
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int = 0,
        global_pool: bool = True,
        drop_path_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        import timm

        # Create a base ViT model without a classification head.  We
        # disable the default classifier (num_classes=0) so that we can
        # attach our own later.
        self.backbone: nn.Module = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        self.num_features = self.backbone.embed_dim

        self.global_pool = global_pool
        if global_pool:
            # Create a normalisation layer for pooled features.
            self.fc_norm = nn.LayerNorm(self.num_features)
        else:
            # Use the default norm inside timm's ViT if not global pooling.
            # The norm is part of the backbone (self.backbone.norm).
            pass

        # Define a classification head.  It will be re‑initialised after
        # loading pretrained weights.
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through patch embedding, position encoding and transformer blocks.
        x = self.backbone.patch_embed(x)
        cls_tokens = self.backbone.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        for blk in self.backbone.blocks:
            x = blk(x)

        if self.global_pool:
            # Pool over spatial tokens (excluding cls token) and normalise.
            x = x[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)
        else:
            # Use cls token and default norm.
            x = self.backbone.norm(x)
            x = x[:, 0]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


def interpolate_pos_embed(
    model: VisionTransformer, checkpoint_model: Dict[str, torch.Tensor]
) -> None:
    """Interpolate position embeddings when image size changes.

    Adapted from util/pos_embed.py in the RETFound repository.  If the
    checkpoint contains positional embeddings with a different spatial
    resolution, they are resized to match the new model.
    """
    if "pos_embed" not in checkpoint_model:
        return
    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    num_patches = model.backbone.patch_embed.num_patches
    num_extra_tokens = model.backbone.pos_embed.shape[-2] - num_patches
    orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5))
    new_size = int((num_patches) ** 0.5)
    if orig_size == new_size:
        return

    # Separate class/distillation tokens from positional tokens.
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    embed_dim = pos_tokens.shape[-1]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embed_dim).permute(
        0, 3, 1, 2
    )
    # Interpolate.
    pos_tokens = F.interpolate(
        pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
        1, new_size * new_size, embed_dim
    )
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model["pos_embed"] = new_pos_embed


def load_pretrained_weights(
    model: VisionTransformer,
    model_name: str,
    pretrained_path: str = "",
    map_location: str = "cpu",
) -> None:
    """Load RETFound pre‑trained weights into the model.

    The HuggingFace Hub contains checkpoints that store the backbone weights
    under either `model` (for MAE based RETFound) or `teacher` (for DINOv2‑based
    RETFound).  The classification head weights are discarded since the
    number of classes typically does not match the downstream task.

    Args:
        model: VisionTransformer instance to initialise.
        model_name: Name of the RETFound model on HuggingFace, e.g.
            ``RETFound_mae_natureCFP`` or ``RETFound_dinov2_meh``.
        pretrained_path: Optional path to a local `.pth` checkpoint.  If
            provided, the hub will not be queried.
        map_location: Device mapping for `torch.load`.
    """
    if pretrained_path:
        ckpt_path = pretrained_path
    else:
        if hf_hub_download is None:
            raise RuntimeError(
                "huggingface_hub is not installed.  Install it or provide a local"
                " pretrained checkpoint via --pretrained_path."
            )
        try:
            ckpt_path = hf_hub_download(
                repo_id=f"YukunZhou/{model_name}", filename=f"{model_name}.pth"
            )
        except HfHubHTTPError as e:
            raise RuntimeError(
                f"Failed to download pretrained weights for {model_name} from the"
                f" HuggingFace Hub: {e}.  Ensure you have permission and are"
                " logged in with `huggingface-cli login` or provide a local"
                " checkpoint via --pretrained_path."
            )

    checkpoint = torch.load(ckpt_path, map_location=map_location)
    # Determine which key contains the backbone weights.
    if "RETFound_dinov2" in model_name:
        checkpoint_model = checkpoint.get("teacher", checkpoint)
    else:
        # For MAE models the key is usually "model".
        checkpoint_model = checkpoint.get("model", checkpoint)

    # Remove leading strings from keys (e.g. "backbone.") to match our model.
    new_state_dict: Dict[str, torch.Tensor] = {}
    for k, v in checkpoint_model.items():
        new_key = k
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone.") :]
        # The HuggingFace checkpoint for RETFound uses names like mlp.w12 and mlp.w3
        # in some versions; map them to the names used by timm.
        new_key = new_key.replace("mlp.w12.", "mlp.fc1.")
        new_key = new_key.replace("mlp.w3.", "mlp.fc2.")
        new_state_dict[new_key] = v

    # Remove head weights if the shape does not match.
    state_dict = model.state_dict()
    keys_to_delete = []
    for k in ["head.weight", "head.bias"]:
        if (
            k in new_state_dict
            and k in state_dict
            and new_state_dict[k].shape != state_dict[k].shape
        ):
            keys_to_delete.append(k)
    for k in keys_to_delete:
        del new_state_dict[k]

    # Interpolate positional embeddings if necessary.
    interpolate_pos_embed(model, new_state_dict)

    # Load weights into model.
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: missing keys when loading pretrained weights: {missing_keys}")
    if unexpected_keys:
        print(
            f"Warning: unexpected keys when loading pretrained weights: {unexpected_keys}"
        )

    # Reinitialise the classification head weights.  Use a truncated normal
    # distribution as in the original RETFound code.
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        nn.init.trunc_normal_(model.head.weight, std=2e-5)
        if model.head.bias is not None:
            nn.init.zeros_(model.head.bias)


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)
) -> list:
    """Compute the top‑k accuracy for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def train_one_epoch(
    model: VisionTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[float, float]:
    """Train the model for a single epoch.

    Returns the average training loss and top‑1 accuracy.
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = len(loader)
    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        running_loss += loss.item()
        running_acc += acc1
        progress.set_postfix(
            {
                "loss": running_loss / (progress.n + 1),
                "acc": running_acc / (progress.n + 1),
            }
        )
    return running_loss / num_batches, running_acc / num_batches


def evaluate(
    model: VisionTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation or test set.

    Returns the average loss and top‑1 accuracy.
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = len(loader)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            running_loss += loss.item()
            running_acc += acc1
    return running_loss / num_batches, running_acc / num_batches


def predict(
    model: VisionTransformer,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[list, list]:
    """Compute predictions and ground truth labels for an entire dataset."""
    model.eval()
    all_preds: list = []
    all_targets: list = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().tolist())
    return all_preds, all_targets


def main(args: argparse.Namespace) -> None:
    # Prepare output directory.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    print(f"Using device: {device}")

    # Build data transforms and datasets.
    train_tf, eval_tf = build_transforms(args.input_size)
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    test_dir = Path(args.data_dir) / "test"

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_dataset = (
        datasets.ImageFolder(str(val_dir), transform=eval_tf)
        if val_dir.exists()
        else None
    )
    test_dataset = (
        datasets.ImageFolder(str(test_dir), transform=eval_tf)
        if test_dir.exists()
        else None
    )
    # Ensure the number of classes matches the dataset unless explicitly set.
    num_classes = (
        args.num_classes if args.num_classes is not None else len(train_dataset.classes)
    )

    # Create model.
    model = VisionTransformer(
        num_classes=num_classes,
        global_pool=True,
        drop_path_rate=args.drop_path,
    )
    model.to(device)

    # Load pretrained backbone weights.
    if args.pretrained_path or args.model_name:
        load_pretrained_weights(
            model, args.model_name, args.pretrained_path, map_location="cpu"
        )

    # Create data loaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if val_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if test_dataset is not None
        else None
    )

    # Define loss function and optimiser.
    # Label smoothing can help with generalisation when classes are imbalanced.
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine annealing scheduler with warmup.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    scaler = (
        torch.cuda.amp.GradScaler()
        if device.type == "cuda"
        else torch.cuda.amp.GradScaler(enabled=False)
    )

    best_val_acc = 0.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )
        scheduler.step()

        # Evaluate on the validation set if provided.
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                # Save best model checkpoint.
                ckpt_path = output_dir / "best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "args": vars(args),
                        "class_to_idx": train_dataset.class_to_idx,
                    },
                    ckpt_path,
                )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%"
            )

    # Save final model.
    final_ckpt_path = output_dir / "last_model.pth"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args": vars(args),
            "class_to_idx": train_dataset.class_to_idx,
        },
        final_ckpt_path,
    )

    print(
        f"Training complete.  Best val acc = {best_val_acc:.2f}% at epoch {best_epoch}."
    )

    # Evaluate on the test set and save predictions.
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test accuracy: {test_acc:.2f}% (loss {test_loss:.4f})")
        preds, targets = predict(model, test_loader, device)
        # Write predictions to CSV.  Each row contains image_path, ground_truth_label, predicted_label.
        csv_path = output_dir / "test_predictions.csv"
        # Flatten test_dataset samples to get image file names.
        image_paths = [path for path, _ in test_dataset.samples]
        # Map index to class name.
        idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "true_label", "predicted_label"])
            for img_path, tgt, pred in zip(image_paths, targets, preds):
                writer.writerow([img_path, idx_to_class[tgt], idx_to_class[pred]])
        print(f"Saved test predictions to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine‑tune a RETFound model for disease classification"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the dataset (containing train/val/test)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save models and predictions",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of disease classes (inferred if not set)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="RETFound_mae_natureCFP",
        help="Name of the RETFound checkpoint on HuggingFace",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="",
        help="Optional local path to pretrained weights (.pth)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Mini‑batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for AdamW optimiser",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine scheduler",
    )
    parser.add_argument("--drop_path", type=float, default=0.1, help="Drop path rate")
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force use of CPU even if CUDA is available",
    )
    args = parser.parse_args()
    main(args)
