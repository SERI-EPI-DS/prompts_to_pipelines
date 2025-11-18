import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torch.nn import functional as F
import timm
from timm.data import create_transform
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma
import time
import math
import datetime
from tqdm import tqdm
from collections import defaultdict, deque

# Import RETFound model
sys.path.append("../RETFound")
try:
    from models_vit import VisionTransformer
except ImportError:
    # If models_vit is not available, we'll define a basic ViT using timm
    from timm.models.vision_transformer import VisionTransformer


def get_args_parser():
    parser = argparse.ArgumentParser("RETFound fine-tuning", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model", default="vit_large_patch16", type=str, help="Name of model to train"
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--drop_path", type=float, default=0.2, metavar="PCT", help="Drop path rate"
    )
    parser.add_argument(
        "--model_path",
        default="../RETFound/RETFound_CFP_weights.pth",
        type=str,
        help="Path to pretrained RETFound weights",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=5e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
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
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    # Training parameters
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Data parameters
    parser.add_argument(
        "--data_path", default="../data/", type=str, help="dataset path"
    )
    parser.add_argument(
        "--output_dir",
        default="../project/results/",
        type=str,
        help="path where to save",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--nb_classes",
        default=None,
        type=int,
        help="number of classes (automatically detected if not specified)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original"',
    )
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing")

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
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Global pooling
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    return parser


def interpolate_pos_embed(model, checkpoint_model):
    """Interpolate position embeddings for different image sizes."""
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
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


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs)
                / (args.epochs - args.warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def build_dataset(is_train, args):
    """Build dataset with appropriate transforms."""
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
    """Build image transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        # Training transform
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            mean=mean,
            std=std,
        )
    else:
        # Validation transform
        t = []
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
        )
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(t)

    return transform


def create_model(args):
    """Create the Vision Transformer model."""
    try:
        # Try to use RETFound's VisionTransformer
        from models_vit import VisionTransformer

        model = VisionTransformer(
            img_size=args.input_size,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            num_classes=args.nb_classes,
            global_pool=args.global_pool,
            drop_path_rate=args.drop_path,
        )
    except:
        # Fallback to timm's implementation
        print("Using timm's VisionTransformer implementation")
        import timm

        model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool="avg" if args.global_pool else "token",
        )

    return model


def load_pretrained_weights(model, checkpoint_path, args):
    """Load pretrained RETFound weights."""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Proceeding without pretrained weights...")
        return model

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print("Load pre-trained checkpoint from: %s" % checkpoint_path)
    if "model" in checkpoint:
        checkpoint_model = checkpoint["model"]
    else:
        checkpoint_model = checkpoint

    # Interpolate position embedding if needed
    interpolate_pos_embed(model, checkpoint_model)

    # Remove head weights if present
    checkpoint_model = {
        k: v
        for k, v in checkpoint_model.items()
        if not k.startswith("head.") and not k.startswith("fc_norm.")
    }

    # Load state dict
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    return model


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
        # Single GPU training, no need to synchronize
        pass

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


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    max_norm,
    mixup_fn,
    args,
):
    """Train for one epoch."""
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        optimizer.zero_grad()

        # Scale loss and backward
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

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    # Gather stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, args):
    """Evaluate model on validation set."""
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, min(5, args.nb_classes)))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    # Gather stats
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
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


def save_model(args, epoch, model, optimizer, loss_scaler, best_acc, model_ema=None):
    """Save model checkpoint."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"checkpoint-{epoch}.pth"
    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
        "args": args,
        "best_acc": best_acc,
    }

    if model_ema is not None:
        to_save["model_ema"] = model_ema.state_dict()

    torch.save(to_save, checkpoint_path)

    # Save best model
    if best_acc == to_save["best_acc"]:
        best_checkpoint_path = output_dir / "checkpoint-best.pth"
        torch.save(to_save, best_checkpoint_path)


def main():
    args = get_args_parser()
    args = args.parse_args()

    # Set device
    device = torch.device(args.device)

    # Fix random seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Auto-detect number of classes if not specified
    if args.nb_classes is None:
        train_dir = os.path.join(args.data_path, "train")
        args.nb_classes = len(os.listdir(train_dir))
        print(f"Detected {args.nb_classes} classes")

    # Create datasets
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    # Create data loaders
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

    # Create model
    model = create_model(args)
    model = load_pretrained_weights(model, args.model_path, args)
    model.to(device)

    model_ema = None

    # Create optimizer
    eff_batch_size = args.batch_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    param_groups = [{"params": model.parameters(), "weight_decay": args.weight_decay}]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # Create loss function
    if args.mixup > 0 or args.cutmix > 0.0:
        # Smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Create Mixup/Cutmix
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

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        # Update learning rate
        lr = adjust_learning_rate(optimizer, epoch, args)

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
            mixup_fn,
            args,
        )

        # Evaluate
        val_stats = evaluate(model, data_loader_val, device, args)

        print(f"Accuracy of the model on the validation set: {val_stats['acc1']:.1f}%")

        if val_stats["acc1"] > max_accuracy:
            max_accuracy = val_stats["acc1"]
            save_model(
                args, epoch, model, optimizer, loss_scaler, max_accuracy, model_ema
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
            "n_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }

        # Save log
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(
            os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(f"Best accuracy: {max_accuracy:.2f}%")


if __name__ == "__main__":
    main()
