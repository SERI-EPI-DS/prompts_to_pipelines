import os
import sys
import argparse
import json
import datetime
import numpy as np
import time
from pathlib import Path
import csv
import math
from collections import defaultdict, deque
from functools import partial

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler, CosineLRScheduler
from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from timm.data import create_transform
from timm.models.layers import trunc_normal_, DropPath

# Add RETFound directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "RETFound"))
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser("RETFound Fine-tuning", add_help=False)

    # Data parameters
    parser.add_argument("--data_path", default="./data/", type=str, help="dataset path")
    parser.add_argument(
        "--output_dir",
        default="./project/results/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--pretrained_weights",
        default="./RETFound/RETFound_CFP_weights.pth",
        help="pretrained weights path",
    )

    # Model parameters
    parser.add_argument(
        "--model", default="vit_large_patch16", type=str, help="Name of model to train"
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--drop_path", type=float, default=0.2, help="Drop path rate")
    parser.add_argument("--model_ema", action="store_true", default=True)
    parser.add_argument("--model_ema_decay", type=float, default=0.99996)
    parser.add_argument("--model_ema_force_cpu", action="store_true", default=False)
    parser.add_argument("--global_pool", action="store_true", default=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt", default="adamw", type=str, help='Optimizer (default: "adamw"'
    )
    parser.add_argument("--opt_eps", default=1e-8, type=float, help="Optimizer Epsilon")
    parser.add_argument(
        "--opt_betas", default=None, type=float, nargs="+", help="Optimizer Betas"
    )
    parser.add_argument(
        "--clip_grad", type=float, default=None, help="Clip gradient norm"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the weight decay.""",
    )

    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.65,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="lower lr bound for cyclic schedulers",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="epochs to warmup LR"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=-1, help="steps to warmup LR"
    )

    # Learning rate schedule parameters (needed for timm scheduler)
    parser.add_argument("--sched", default="cosine", type=str, help="LR scheduler")
    parser.add_argument(
        "--lr_noise",
        type=float,
        nargs="+",
        default=None,
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr_noise_pct",
        type=float,
        default=0.67,
        help="learning rate noise limit percent",
    )
    parser.add_argument(
        "--lr_noise_std", type=float, default=1.0, help="learning rate noise std-dev"
    )
    parser.add_argument(
        "--lr_cycle_mul",
        type=float,
        default=1.0,
        help="learning rate cycle len multiplier",
    )
    parser.add_argument(
        "--lr_cycle_decay",
        type=float,
        default=0.5,
        help="amount to decay each learning rate cycle",
    )
    parser.add_argument(
        "--lr_cycle_limit",
        type=int,
        default=1,
        help="learning rate cycle limit, cycles enabled if > 1",
    )
    parser.add_argument(
        "--lr_k_decay", type=float, default=1.0, help="learning rate k decay"
    )
    parser.add_argument(
        "--warmup_lr", type=float, default=1e-6, help="warmup learning rate"
    )
    parser.add_argument(
        "--decay_epochs", type=float, default=30, help="epoch interval to decay LR"
    )
    parser.add_argument("--decay_rate", type=float, default=0.1, help="LR decay rate")
    parser.add_argument(
        "--cooldown_epochs", type=int, default=0, help="epochs to cooldown LR at min_lr"
    )
    parser.add_argument(
        "--patience_epochs",
        type=int,
        default=10,
        help="patience epochs for Plateau LR scheduler",
    )
    parser.add_argument(
        "--lr_interval",
        default="epoch",
        type=str,
        help="LR scheduler update interval (epoch or step)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter", type=float, default=0.4, help="Color jitter factor"
    )
    parser.add_argument(
        "--aa", type=str, default="rand-m9-mstd0.5-inc1", help="Use AutoAugment policy"
    )
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help="Training interpolation",
    )

    # Random Erase params
    parser.add_argument("--reprob", type=float, default=0.25, help="Random erase prob")
    parser.add_argument("--remode", type=str, default="pixel", help="Random erase mode")
    parser.add_argument("--recount", type=int, default=1, help="Random erase count")

    # Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0.8, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=1.0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help="How to apply mixup/cutmix params",
    )

    # Dataset parameters
    parser.add_argument("--batch_size", default=16, type=int, help="Per GPU batch size")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        default=True,
        help="Pin CPU memory in DataLoader",
    )

    # Misc
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--save_ckpt_freq", default=20, type=int)
    parser.add_argument("--save_ckpt_num", default=3, type=int)

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)

    # distributed training parameters
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    return parser


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if is_train:
        dataset = datasets.ImageFolder(
            os.path.join(args.data_path, "train"), transform=transform
        )
    else:
        dataset = datasets.ImageFolder(
            os.path.join(args.data_path, "val"), transform=transform
        )

    return dataset


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transform

    # eval transform
    t = []
    if resize_im:
        if args.crop_pct is None:
            size = int((256 / 224) * args.input_size)
        else:
            size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(t)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
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
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        global_pool=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1
        self.global_pool = global_pool
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.fc_norm = None

        # Classifier head(s)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # Weight init
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def main(args):
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Data loading
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    args.nb_classes = len(dataset_train.classes)
    print(f"Number of classes: {args.nb_classes}")
    print(f"Classes: {dataset_train.classes}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    # Model
    print(f"Creating model: {args.model}")

    # Create model using the local function
    model = vit_large_patch16(
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load pretrained weights
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        print("Load pre-trained checkpoint from: %s" % args.pretrained_weights)
        checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    else:
        print(
            "Warning: No pretrained weights found or specified. Training from scratch."
        )

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("number of params:", n_parameters)

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd_params(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # Create scheduler
    num_epochs = args.epochs
    if args.lr_interval == "epoch":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=args.lr_cycle_limit,
            t_in_epochs=True,
        )
    else:
        # step-based scheduler
        steps_per_epoch = len(data_loader_train)
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs * steps_per_epoch,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs * steps_per_epoch,
            cycle_limit=args.lr_cycle_limit,
            t_in_epochs=False,
        )

    # criterion
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # Resume
    if args.auto_resume:
        resume_file = auto_load_model(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            model_ema=model_ema,
        )
        if resume_file:
            args.resume = resume_file

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema:
                model_ema._load_checkpoint(checkpoint["model_ema"])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("Resumed from checkpoint %s" % args.resume)

    # Training
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save class names
    with open(output_dir / "classes.json", "w") as f:
        json.dump(dataset_train.classes, f)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        # Train
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            lr_scheduler=lr_scheduler,
            args=args,
        )

        # Step scheduler
        if args.lr_interval == "epoch":
            lr_scheduler.step(epoch)

        # Evaluate
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on validation images: {test_stats['acc1']:.1f}%"
        )

        if model_ema is not None:
            test_stats_ema = evaluate(data_loader_val, model_ema.ema, device)
            print(
                f"Accuracy of the EMA model on validation images: {test_stats_ema['acc1']:.1f}%"
            )
            max_accuracy = max(max_accuracy, test_stats_ema["acc1"])
        else:
            max_accuracy = max(max_accuracy, test_stats["acc1"])

        print(f"Max accuracy: {max_accuracy:.2f}%")

        # Save checkpoint
        if output_dir:
            checkpoint_path = output_dir / "checkpoint-last.pth"
            save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                model_ema=model_ema,
                checkpoint_path=checkpoint_path,
                lr_scheduler=lr_scheduler,
            )

            if test_stats["acc1"] >= max_accuracy:
                checkpoint_path = output_dir / "checkpoint-best.pth"
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    model_ema=model_ema,
                    checkpoint_path=checkpoint_path,
                    lr_scheduler=lr_scheduler,
                )

            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                checkpoint_path = output_dir / f"checkpoint-{epoch:04d}.pth"
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    model_ema=model_ema,
                    checkpoint_path=checkpoint_path,
                    lr_scheduler=lr_scheduler,
                )

        # Log stats
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    max_norm=None,
    model_ema=None,
    mixup_fn=None,
    lr_scheduler=None,
    args=None,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    for idx, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # Step scheduler if using step-based updates
        if lr_scheduler is not None and args.lr_interval == "step":
            lr_scheduler.step_update(epoch * len(data_loader) + idx)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def lrd_params(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75):
    """
    Layer-wise learning rate decay optimizer parameters
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    """
    if name in ["cls_token", "pos_embed"]:
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    else:
        return num_layers


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not torch.distributed.is_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def save_model(
    args,
    epoch,
    model,
    model_without_ddp,
    optimizer,
    loss_scaler,
    checkpoint_path,
    model_ema=None,
    lr_scheduler=None,
):
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": args,
    }

    if lr_scheduler is not None:
        to_save["lr_scheduler"] = lr_scheduler.state_dict()

    if model_ema is not None:
        to_save["model_ema"] = get_state_dict(model_ema)

    torch.save(to_save, checkpoint_path)


def auto_load_model(
    args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None
):
    output_dir = Path(args.output_dir)
    # auto resume from latest checkpoint
    latest = os.path.join(output_dir, "checkpoint-last.pth")
    if os.path.exists(latest):
        return latest
    return None


def get_state_dict(model):
    if hasattr(model, "module"):
        return model.module.state_dict()
    elif hasattr(model, "ema"):
        return model.ema.state_dict()
    else:
        return model.state_dict()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
