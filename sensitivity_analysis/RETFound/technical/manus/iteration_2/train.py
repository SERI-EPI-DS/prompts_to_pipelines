#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script conducts fine-tuning of the RETFound_mae ViT-L classifier.
Self-contained version that doesn't require RETFound repository imports.
"""
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
from pathlib import Path
import math
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingLR

# Handle timm compatibility issues
try:
    import timm
    from timm.models.vision_transformer import VisionTransformer
    from timm.models.layers import (
        PatchEmbed,
        Mlp,
        DropPath,
        trunc_normal_,
        lecun_normal_,
    )
except ImportError as e:
    print(f"Error importing timm: {e}")
    print("Please install timm with: pip install timm")
    sys.exit(1)


class NativeScalerWithGradNormCount:
    """Native mixed precision scaler with gradient norm counting"""

    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformerForClassification(VisionTransformer):
    """Vision Transformer for classification with RETFound compatibility"""

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
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        global_pool="token",
        **kwargs,
    ):

        # Initialize parent class
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            **kwargs,
        )

        self.global_pool = global_pool

        # Replace head for new number of classes
        if global_pool == "avg":
            self.head = (
                nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            )
        else:
            self.head = (
                nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            )

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == "avg":
            x = x[:, 1:].mean(dim=1)  # global average pooling, exclude class token
        elif self.global_pool == "token":
            x = x[:, 0]  # class token
        else:
            x = x[:, 0]  # default to class token

        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def interpolate_pos_embed(model, checkpoint_model):
    """Interpolate position embeddings for different image sizes"""
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        # Height and width for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # Height and width for the new position embedding
        new_size = int(num_patches**0.5)

        if orig_size != new_size:
            print(
                f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}"
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def create_model(
    model_name="vit_large_patch16",
    num_classes=1000,
    drop_path_rate=0.1,
    global_pool=True,
):
    """Create Vision Transformer model"""
    if model_name == "vit_large_patch16":
        model = VisionTransformerForClassification(
            img_size=224,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            global_pool="avg" if global_pool else "token",
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

    return model


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    clip_grad=None,
    log_writer=None,
    args=None,
):
    """Train for one epoch"""
    model.train()
    metric_logger = {}

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()

        # This attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=clip_grad,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()

        if data_iter_step % 50 == 0:
            print(
                f"Epoch: [{epoch}][{data_iter_step}/{len(data_loader)}] Loss: {loss_value:.4f}"
            )

    return {"loss": loss_value}


def evaluate(data_loader, model, device):
    """Evaluate model"""
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = {}

    # Switch to evaluation mode
    model.eval()

    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for images, target in data_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc1 = 100 * correct / total
    avg_loss = total_loss / len(data_loader)

    print(f"Test Accuracy: {acc1:.2f}% ({correct}/{total})")

    return {
        "acc1": acc1,
        "acc5": acc1,
        "loss": avg_loss,
    }  # acc5 same as acc1 for simplicity


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    """Save model checkpoint"""
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"checkpoint-{epoch}.pth"
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": args,
    }
    torch.save(to_save, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    """Load model checkpoint"""
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
        print(f"Resumed from checkpoint: {args.resume}")


def get_args_parser():
    parser = argparse.ArgumentParser(
        "RETFound fine-tuning for image classification", add_help=False
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", default=50, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    # Augmentation parameters
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true", default=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    # Dataset parameters
    parser.add_argument("--data_path", required=True, type=str, help="dataset path")
    parser.add_argument(
        "--nb_classes", default=2, type=int, help="number of the classification types"
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        default=True,
        help="Pin CPU memory in DataLoader for more efficient transfer to GPU.",
    )

    return parser


def create_optimizer_with_layer_decay(model, lr, weight_decay, layer_decay):
    """Create optimizer with layer-wise learning rate decay"""
    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine layer index
        layer_id = 0
        if "blocks" in name:
            # Extract block number
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "blocks" and i + 1 < len(parts):
                    try:
                        layer_id = int(parts[i + 1]) + 1
                    except ValueError:
                        layer_id = 0
                    break
        elif "head" in name:
            layer_id = 24  # Head gets full learning rate

        # Calculate layer-wise learning rate
        lr_scale = layer_decay ** (24 - layer_id)

        # Apply weight decay selectively
        wd = weight_decay
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or "pos_embed" in name
            or "cls_token" in name
        ):
            wd = 0.0

        param_groups.append(
            {"params": [param], "lr": lr * lr_scale, "weight_decay": wd}
        )

    return torch.optim.AdamW(param_groups)


def main(args):
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Data augmentation and normalization
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.input_size,
                scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(
                args.input_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), transform=transform_train
    )
    dataset_val = datasets.ImageFolder(
        os.path.join(args.data_path, "val"), transform=transform_val
    )

    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_val)}")
    print(f"Number of classes: {len(dataset_train.classes)}")
    print(f"Classes: {dataset_train.classes}")

    # Create data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        drop_last=False,
    )

    # Create model
    model = create_model(
        model_name=args.model,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load pre-trained weights if specified
    if args.finetune and not args.eval:
        print(f"Loading pre-trained checkpoint from: {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location="cpu")

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

        # Interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # Load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"Load pretrained model with msg: {msg}")

        # Freeze all layers except head for fine-tuning
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params (M): {n_parameters / 1.e6:.2f}")

    # Create optimizer
    optimizer = create_optimizer_with_layer_decay(
        model, args.lr, args.weight_decay, args.layer_decay
    )

    # Create loss scaler
    loss_scaler = NativeScalerWithGradNormCount()

    # Create criterion
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    print(f"Criterion = {str(criterion)}")

    # Load model if resuming
    load_model(args, model, optimizer, loss_scaler)

    # Create log writer
    if args.log_dir and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Evaluation only
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        return

    # Training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args,
        )

        if args.output_dir:
            save_model(args, epoch, model, model, optimizer, loss_scaler)

        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir:
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
