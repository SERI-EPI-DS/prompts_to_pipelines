"""
Fine-tuning script for RETFound_mae ViT-L foundation model
Performs classification on retinal fundus photographs
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Add RETFound path to system
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "RETFound"))

# Import from RETFound repository
import models_vit
from util.pos_embed import interpolate_pos_embed
from util.lr_sched import adjust_learning_rate
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_args():
    parser = argparse.ArgumentParser("RETFound Fine-tuning Script")

    # Paths
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data directory containing train/val/test folders",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save outputs"
    )
    parser.add_argument(
        "--retfound_weights",
        type=str,
        default="../RETFound/RETFound_CFP_weights.pth",
        help="Path to RETFound pretrained weights",
    )

    # Model parameters
    parser.add_argument("--input_size", default=224, type=int, help="Image input size")
    parser.add_argument(
        "--nb_classes", type=int, required=True, help="Number of classes"
    )
    parser.add_argument("--drop_path", type=float, default=0.2, help="Drop path rate")
    parser.add_argument(
        "--model_name",
        type=str,
        default="vit_large_patch16",
        help="Model architecture name",
    )
    parser.add_argument(
        "--global_pool", action="store_true", default=True, help="Use global pooling"
    )

    # Training parameters
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Epochs to warmup LR"
    )

    # Optimizer parameters
    parser.add_argument("--blr", type=float, default=5e-3, help="Base learning rate")
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Lower lr bound for cyclic schedulers",
    )
    parser.add_argument(
        "--layer_decay", type=float, default=0.65, help="Layer-wise lr decay"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing")

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter", type=float, default=0.4, help="Color jitter factor"
    )
    parser.add_argument(
        "--aa", type=str, default="rand-m9-mstd0.5-inc1", help="AutoAugment policy"
    )
    parser.add_argument("--reprob", type=float, default=0.25, help="Random erase prob")
    parser.add_argument("--remode", type=str, default="pixel", help="Random erase mode")
    parser.add_argument("--recount", type=int, default=1, help="Random erase count")

    # Other parameters
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--num_workers", default=8, type=int, help="Number of data loading workers"
    )
    parser.add_argument(
        "--pin_mem", action="store_true", default=True, help="Pin memory"
    )
    parser.add_argument("--resume", default="", type=str, help="Resume from checkpoint")

    return parser.parse_args()


def build_transform(is_train, args):
    """Build image transformations"""
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        # Training augmentation - using standard torchvision transforms
        transform_list = [
            transforms.RandomResizedCrop(
                args.input_size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]

        # Add color jitter if specified
        if args.color_jitter > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=args.color_jitter,
                    contrast=args.color_jitter,
                    saturation=args.color_jitter,
                    hue=args.color_jitter / 4,
                )
            )

        # Add rotation and affine transforms
        transform_list.extend(
            [
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        # Add random erasing if specified
        if args.reprob > 0:
            transform_list.append(
                transforms.RandomErasing(p=args.reprob, scale=(0.02, 0.33))
            )

        transform = transforms.Compose(transform_list)
    else:
        # Validation/Test augmentation
        transform = transforms.Compose(
            [
                transforms.Resize(int(args.input_size * 256 / 224)),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transform


def build_dataset(is_train, args):
    """Build dataset"""
    transform = build_transform(is_train, args)

    if is_train:
        dataset = ImageFolder(
            os.path.join(args.data_path, "train"), transform=transform
        )
    else:
        dataset = ImageFolder(os.path.join(args.data_path, "val"), transform=transform)

    return dataset


def create_model(args):
    """Create ViT-L model with RETFound weights"""
    print(f"Attempting to create model: {args.model_name}")

    # First, let's see what's available in models_vit
    print("Available models in models_vit:")
    available_models = [
        name
        for name in dir(models_vit)
        if not name.startswith("_") and callable(getattr(models_vit, name))
    ]
    print(f"Found: {available_models}")

    # Try different ways to get the model
    model = None

    # Method 1: Try the exact name
    if hasattr(models_vit, args.model_name):
        print(f"Found model function: {args.model_name}")
        model_fn = getattr(models_vit, args.model_name)
        model = model_fn(
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    # Method 2: Try VisionTransformer class directly
    elif hasattr(models_vit, "VisionTransformer"):
        print("Using VisionTransformer class directly")
        model = models_vit.VisionTransformer(
            img_size=args.input_size,
            patch_size=16,
            embed_dim=1024,  # ViT-Large configuration
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            num_classes=args.nb_classes,
            global_pool=args.global_pool,
            drop_path_rate=args.drop_path,
            norm_layer=nn.LayerNorm,
        )
    else:
        # If none of the above work, list what's available and exit
        print(
            f"Error: Could not find model '{args.model_name}' or VisionTransformer class"
        )
        print(f"Available items in models_vit module: {dir(models_vit)}")
        raise ValueError(f"Model {args.model_name} not found in models_vit")

    # Load pretrained weights
    if os.path.exists(args.retfound_weights):
        print(f"Loading RETFound weights from {args.retfound_weights}")
        checkpoint = torch.load(args.retfound_weights, map_location="cpu")

        # Get model state dict
        checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint
        state_dict = model.state_dict()

        # Remove head weights if shape mismatch
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # Remove fc_norm if it exists in checkpoint but not in model
        for k in ["fc_norm.weight", "fc_norm.bias"]:
            if k in checkpoint_model and k not in state_dict:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # Interpolate position embedding if needed
        interpolate_pos_embed(model, checkpoint_model)

        # Load state dict
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"Loaded with msg: {msg}")

        # Initialize head with truncated normal
        if hasattr(model, "head") and hasattr(model.head, "weight"):
            trunc_normal_(model.head.weight, std=2e-5)
            if hasattr(model.head, "bias") and model.head.bias is not None:
                nn.init.zeros_(model.head.bias)
    else:
        print(f"Warning: RETFound weights not found at {args.retfound_weights}")
        print("Training from scratch...")

    return model


def get_params_groups(model, weight_decay, layer_decay=1.0):
    """
    Parameter groups for layer-wise lr decay & weight decay
    """
    param_group_names = {}
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for bias and normalization
        if param.ndim == 1 or name.endswith(".bias"):
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if group_name not in param_group_names:
            param_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }
        param_group_names[group_name]["params"].append(param)

    param_groups = list(param_group_names.values())
    return param_groups


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(model, data_loader, optimizer, criterion, epoch, args):
    """Train for one epoch"""
    model.train()

    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    # Enable automatic mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.device == "cuda" else None

    for i, (images, target) in enumerate(data_loader):
        # Adjust learning rate
        it = len(data_loader) * epoch + i
        for param_group in optimizer.param_groups:
            param_group["lr"] = adjust_learning_rate(
                args.blr,
                it,
                args.warmup_epochs,
                args.epochs,
                len(data_loader),
                args.min_lr,
            )

        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        # Forward pass with mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)

            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward and backward pass
            output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measure accuracy and record loss
        (acc1,) = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        if i % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch: [{epoch}][{i}/{len(data_loader)}]\t"
                f"LR: {lr:.6f}\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})"
            )

    return losses.avg, top1.avg


@torch.no_grad()
def validate(model, data_loader, criterion, args):
    """Validate model"""
    model.eval()

    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    for images, target in data_loader:
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        # Forward pass
        output = model(images)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        (acc1,) = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

    print(f" * Acc@1 {top1.avg:.3f} Loss {losses.avg:.4f}")

    return losses.avg, top1.avg


def save_checkpoint(state, is_best, args):
    """Save checkpoint"""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save latest
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
    torch.save(state, checkpoint_path)

    # Save best
    if is_best:
        best_path = os.path.join(args.output_dir, "checkpoint_best.pth")
        torch.save(state, best_path)
        print(f"Best model saved to {best_path}")


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Build datasets
    print("Building datasets...")
    train_dataset = build_dataset(is_train=True, args=args)
    val_dataset = build_dataset(is_train=False, args=args)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")

    # Calculate class weights for balanced sampling
    train_targets = np.array(train_dataset.targets)
    unique_targets = np.unique(train_targets)
    class_sample_count = np.array(
        [len(np.where(train_targets == t)[0]) for t in unique_targets]
    )
    print(f"Class distribution: {dict(zip(unique_targets, class_sample_count))}")

    # Create weighted sampler for class balance
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in train_targets])
    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    # Create model
    try:
        model = create_model(args)
        model = model.to(args.device)

        # Get number of parameters
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {n_parameters / 1e6:.2f}M")
    except Exception as e:
        print(f"Error creating model: {e}")
        print("\nTrying alternative model creation approach...")

        # Alternative: Import timm and use it to create ViT model
        try:
            import timm

            print("Creating model using timm...")
            model = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                global_pool="avg" if args.global_pool else "token",
            )

            # Load RETFound weights if available
            if os.path.exists(args.retfound_weights):
                print(f"Loading RETFound weights into timm model...")
                checkpoint = torch.load(args.retfound_weights, map_location="cpu")
                checkpoint_model = (
                    checkpoint["model"] if "model" in checkpoint else checkpoint
                )

                # Filter out incompatible keys
                state_dict = model.state_dict()
                filtered_checkpoint = {}
                for k, v in checkpoint_model.items():
                    if k in state_dict and state_dict[k].shape == v.shape:
                        filtered_checkpoint[k] = v
                    else:
                        print(f"Skipping key {k}")

                msg = model.load_state_dict(filtered_checkpoint, strict=False)
                print(f"Loaded with msg: {msg}")

            model = model.to(args.device)
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of parameters: {n_parameters / 1e6:.2f}M")

        except Exception as e2:
            print(f"Failed to create model with timm: {e2}")
            raise

    # Create optimizer
    param_groups = get_params_groups(model, args.weight_decay, args.layer_decay)
    optimizer = optim.AdamW(param_groups, lr=args.blr, betas=(0.9, 0.95))

    # Create loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc1 = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc1 = checkpoint["best_acc1"]
        print(f"Resumed from epoch {start_epoch} with best acc {best_acc1:.2f}%")

    # Training loop
    print("\nStarting training...")
    training_time = AverageMeter("Time", ":6.3f")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # Train for one epoch
        train_loss, train_acc1 = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, args
        )

        # Validate
        val_loss, val_acc1 = validate(model, val_loader, criterion, args)

        # Update time
        epoch_time = time.time() - start_time
        training_time.update(epoch_time)

        # Save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc1": best_acc1,
            "args": args,
        }

        save_checkpoint(checkpoint, is_best, args)

        print(
            f"Epoch {epoch}/{args.epochs-1} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc1:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc1:.2f}% - "
            f"Best Acc: {best_acc1:.2f}% - "
            f"Time: {epoch_time:.1f}s\n"
        )

    total_time = training_time.sum
    print(f"Training completed! Best validation accuracy: {best_acc1:.2f}%")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(
        f"Best model saved to: {os.path.join(args.output_dir, 'checkpoint_best.pth')}"
    )


if __name__ == "__main__":
    args = get_args()

    # Check CUDA availability
    if not torch.cuda.is_available() and args.device == "cuda":
        print("WARNING: CUDA not available, using CPU")
        args.device = "cpu"
    else:
        print(f"Using device: {args.device}")
        if args.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    main(args)
