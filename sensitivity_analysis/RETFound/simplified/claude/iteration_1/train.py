"""
Fine-tuning script for RETFound on custom ophthalmology datasets.
This script handles the complete training pipeline with state-of-the-art practices.
Updated version with PyTorch 2.6+ compatibility and fixed metric logging.
"""

import os
import sys
import argparse
import json
import datetime
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import time
from collections import deque
import math
import warnings

warnings.filterwarnings("ignore")

# Add RETFound directory to path (adjust if needed)
sys.path.append("./RETFound_MAE")

try:
    import models_vit
    from util.pos_embed import interpolate_pos_embed
except ImportError:
    print(
        "Error: Could not import RETFound modules. Make sure RETFound_MAE is in the correct path."
    )
    print("You may need to adjust the sys.path.append line above.")
    sys.exit(1)

# Try to import optional modules
try:
    from timm.data.mixup import Mixup
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

    TIMM_AVAILABLE = True
except ImportError:
    print(
        "Warning: timm not available. Using basic versions of mixup and label smoothing."
    )
    TIMM_AVAILABLE = False

# For downloading pretrained weights
try:
    from huggingface_hub import hf_hub_download

    HF_AVAILABLE = True
except ImportError:
    print(
        "Warning: huggingface_hub not installed. Install with: pip install huggingface_hub"
    )
    HF_AVAILABLE = False


# Simple LARS implementation if not available
class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling optimizer.
    Based on: https://arxiv.org/abs/1708.03888
    """

    def __init__(self, params, lr=0.001, weight_decay=0, momentum=0.9, eta=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta)
        super(LARS, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(grad)

                if param_norm != 0 and grad_norm != 0:
                    adaptive_lr = eta * param_norm / grad_norm
                    lr = adaptive_lr * lr

                if not hasattr(p, "momentum_buffer"):
                    p.momentum_buffer = torch.zeros_like(p.data)

                buf = p.momentum_buffer
                buf.mul_(momentum).add_(grad, alpha=-lr)
                p.data.add_(buf)

        return loss


# Simple implementations if timm is not available
if not TIMM_AVAILABLE:

    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing

        def forward(self, pred, target):
            n_classes = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
            log_prb = torch.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
            return loss

    class SoftTargetCrossEntropy(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, pred, target):
            if isinstance(target, tuple):
                # Handle mixup case
                y_a, y_b, lam = target
                loss_a = torch.nn.functional.cross_entropy(pred, y_a)
                loss_b = torch.nn.functional.cross_entropy(pred, y_b)
                return lam * loss_a + (1 - lam) * loss_b
            else:
                log_prb = torch.log_softmax(pred, dim=1)
                loss = -(target * log_prb).sum(dim=1).mean()
                return loss

    class Mixup:
        def __init__(
            self,
            mixup_alpha=1.0,
            cutmix_alpha=0.0,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.1,
            num_classes=1000,
        ):
            self.mixup_alpha = mixup_alpha
            self.cutmix_alpha = cutmix_alpha
            self.cutmix_minmax = cutmix_minmax
            self.mix_prob = prob
            self.switch_prob = switch_prob
            self.label_smoothing = label_smoothing
            self.num_classes = num_classes
            self.mode = mode

        def __call__(self, x, target):
            if np.random.rand() > self.mix_prob:
                return x, target

            if self.mixup_alpha > 0 and np.random.rand() < self.switch_prob:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                batch_size = x.size(0)
                index = torch.randperm(batch_size).to(x.device)
                mixed_x = lam * x + (1 - lam) * x[index]
                y_a, y_b = target, target[index]
                return mixed_x, (y_a, y_b, lam)

            return x, target


# Learning rate scheduling functions
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


class RetinalDataset(datasets.ImageFolder):
    """Extended ImageFolder dataset with additional preprocessing"""

    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path


def build_dataset(args, is_train):
    """Build dataset with appropriate augmentations"""

    # State-of-the-art augmentation strategies
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                ),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(int(args.input_size * 1.15)),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    dataset_path = os.path.join(args.data_path, "train" if is_train else "val")
    dataset = RetinalDataset(dataset_path, transform=transform)

    return dataset


def build_model(args):
    """Build and initialize the RETFound model"""

    # Create model
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load pretrained weights
    if args.finetune:
        print(f"Loading pretrained weights from {args.finetune}...")

        # Map model names to HuggingFace repos
        model_repos = {
            "RETFound_mae_natureCFP": "YukunZhou/RETFound_mae_natureCFP",
            "RETFound_mae_natureOCT": "YukunZhou/RETFound_mae_natureOCT",
            "RETFound_mae_meh": "YukunZhou/RETFound_mae_meh",
            "RETFound_mae_shanghai": "YukunZhou/RETFound_mae_shanghai",
            "RETFound_dinov2_meh": "YukunZhou/RETFound_dinov2_meh",
            "RETFound_dinov2_shanghai": "YukunZhou/RETFound_dinov2_shanghai",
        }

        checkpoint_path = None

        if args.finetune in model_repos and HF_AVAILABLE:
            try:
                # Download from HuggingFace
                checkpoint_path = hf_hub_download(
                    repo_id=model_repos[args.finetune],
                    filename="pytorch_model.bin",
                    cache_dir="./pretrained_weights",
                )
            except Exception as e:
                print(f"Could not download from HuggingFace: {e}")
                print("Please download manually or provide local path.")

        if checkpoint_path is None:
            # Use local path
            checkpoint_path = args.finetune
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint not found at {checkpoint_path}")
                print("Training from scratch...")
                return model

        try:
            # Load with weights_only=False for compatibility
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

            # Load pretrained weights
            checkpoint_model = (
                checkpoint["model"] if "model" in checkpoint else checkpoint
            )
            state_dict = model.state_dict()

            # Remove head weights if number of classes doesn't match
            for k in list(checkpoint_model.keys()):
                if (
                    k.startswith("head")
                    and k in state_dict
                    and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    print(f"Removing {k} from pretrained weights due to size mismatch")
                    del checkpoint_model[k]

            # Interpolate position embedding if needed
            interpolate_pos_embed(model, checkpoint_model)

            # Load weights
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(f"Loaded pretrained weights: {msg}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training from scratch...")

    return model


class SmoothedValue:
    """Track a series of values and provide access to smoothed values"""

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
        """No-op for single GPU"""
        pass

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item() if len(d) > 0 else 0.0

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item() if len(d) > 0 else 0.0

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self):
        return max(self.deque) if len(self.deque) > 0 else 0.0

    @property
    def value(self):
        return self.deque[-1] if len(self.deque) > 0 else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    """Utility class for logging metrics"""

    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)

    def add_meter(self, name, meter):
        """Add a new meter"""
        self.meters[name] = meter

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
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
            f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)"
        )

    def synchronize_between_processes(self):
        """No-op for single GPU"""
        for meter in self.meters.values():
            if hasattr(meter, "synchronize_between_processes"):
                meter.synchronize_between_processes()

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)


def accuracy(output, target, topk=(1,)):
    """Compute accuracy over the k top predictions"""
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


# NativeScaler implementation
class NativeScaler:
    """AMP scaler with gradient norm tracking"""

    def __init__(self):
        self._scaler = (
            torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        )

    def __call__(
        self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False
    ):
        if self._scaler is not None:
            self._scaler.scale(loss).backward(create_graph=create_graph)
            if clip_grad is not None:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            loss.backward(create_graph=create_graph)
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            optimizer.step()

    def state_dict(self):
        return self._scaler.state_dict() if self._scaler is not None else {}

    def load_state_dict(self, state_dict):
        if self._scaler is not None:
            self._scaler.load_state_dict(state_dict)


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    max_norm=None,
    mixup_fn=None,
    args=None,
):
    """Train for one epoch with mixed precision and gradient accumulation"""

    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = f"Epoch: [{epoch}]"

    for data_iter_step, (samples, targets, _) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        # Adjust learning rate
        it = data_iter_step / len(data_loader)
        if args.lr_scheduler == "cosine":
            lr = adjust_learning_rate(optimizer, epoch + it, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Forward pass
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # Backward pass
        optimizer.zero_grad()
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr_value = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr_value)

    # Gather stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    """Evaluate model on validation set"""

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    # Pre-initialize acc1 and acc5 meters
    metric_logger.add_meter("acc1", SmoothedValue(fmt="{global_avg:.3f}"))
    if (
        hasattr(data_loader.dataset, "classes")
        and len(data_loader.dataset.classes) >= 5
    ):
        metric_logger.add_meter("acc5", SmoothedValue(fmt="{global_avg:.3f}"))

    header = "Val:"

    model.eval()

    for samples, targets, _ in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else:
            output = model(samples)
            loss = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, min(5, output.shape[1])))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        if output.shape[1] >= 5:
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    # Gather stats
    metric_logger.synchronize_between_processes()
    if "acc5" in metric_logger.meters:
        print(
            "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                top1=metric_logger.acc1,
                top5=metric_logger.acc5,
                losses=metric_logger.loss,
            )
        )
    else:
        print(
            "* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                top1=metric_logger.acc1, losses=metric_logger.loss
            )
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_checkpoint(state, filename):
    """Save checkpoint with weights_only compatibility"""
    # For maximum compatibility, save only essential state
    torch.save(state, filename, _use_new_zipfile_serialization=True)


def load_checkpoint(filename):
    """Load checkpoint with weights_only compatibility"""
    try:
        # Try loading with weights_only=True first
        return torch.load(filename, map_location="cpu", weights_only=True)
    except:
        # Fall back to weights_only=False if needed
        print("Note: Loading checkpoint with weights_only=False for compatibility")
        return torch.load(filename, map_location="cpu", weights_only=False)


def main(args):
    """Main training function"""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        cudnn.benchmark = True

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Build datasets
    print("Building datasets...")
    dataset_train = build_dataset(args, is_train=True)
    dataset_val = build_dataset(args, is_train=False)

    print(f"Training samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_val)}")
    print(f"Number of classes: {args.nb_classes}")

    # Data loaders
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Mixup augmentation
    mixup_fn = None
    if args.mixup > 0:
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

    # Build model
    print("Building model...")
    model = build_model(args)
    model.to(device)

    # Model info
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters:,}")

    # Calculate effective batch size and learning rate
    eff_batch_size = args.batch_size * args.accum_iter
    args.lr = args.blr * eff_batch_size / 256
    print(f"Base learning rate: {args.blr}")
    print(f"Effective learning rate: {args.lr}")

    # Optimizer
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "bias" not in n]},
        {
            "params": [p for n, p in model.named_parameters() if "bias" in n],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
        )
    elif args.optimizer == "lars":
        optimizer = LARS(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )

    print(f"Using optimizer: {args.optimizer}")

    # Loss function
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Mixed precision scaler
    loss_scaler = NativeScaler()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print(f"\nStart training for {args.epochs} epochs")
    print("=" * 60)
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Adjust learning rate
        if args.lr_scheduler == "cosine":
            lr = adjust_learning_rate(optimizer, epoch, args)
        elif args.lr_scheduler == "step":
            if epoch in args.lr_steps:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= args.lr_gamma

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
        val_stats = evaluate(data_loader_val, model, device)

        # Save checkpoint
        if val_stats["acc1"] > best_acc:
            best_acc = val_stats["acc1"]
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "scaler": loss_scaler.state_dict(),
            }
            save_checkpoint(
                save_dict, os.path.join(args.output_dir, "checkpoint_best.pth")
            )
            print(f"✓ New best model saved with accuracy: {best_acc:.2f}%")

        # Save latest checkpoint every few epochs
        if (epoch + 1) % 10 == 0:
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_stats["acc1"],
                "scaler": loss_scaler.state_dict(),
            }
            save_checkpoint(
                save_dict,
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

        # Save latest
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_stats["acc1"],
            "scaler": loss_scaler.state_dict(),
        }
        save_checkpoint(
            save_dict, os.path.join(args.output_dir, "checkpoint_latest.pth")
        )

        # Log stats
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
        }

        with open(
            os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        print(
            f"\nEpoch {epoch}: Val Acc@1: {val_stats['acc1']:.2f}%, Best: {best_acc:.2f}%"
        )
        print("-" * 60)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("=" * 60)
    print(f"Training completed in {total_time_str}")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RETFound Fine-tuning Script")

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        help="Model architecture: vit_large_patch16, vit_base_patch16, etc.",
    )
    parser.add_argument("--input_size", default=224, type=int, help="Image input size")
    parser.add_argument("--drop_path", type=float, default=0.2, help="Drop path rate")
    parser.add_argument(
        "--global_pool", action="store_true", default=True, help="Use global pooling"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to dataset (with train/val/test folders)",
    )
    parser.add_argument(
        "--nb_classes", required=True, type=int, help="Number of classes"
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to save checkpoints and logs",
    )

    # Training parameters
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument(
        "--accum_iter", default=1, type=int, help="Gradient accumulation steps"
    )
    parser.add_argument("--resume", default="", type=str, help="Resume from checkpoint")

    # Optimizer parameters
    parser.add_argument("--blr", type=float, default=5e-3, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adamw", "sgd", "lars"],
        help="Optimizer",
    )
    parser.add_argument(
        "--clip_grad", type=float, default=None, help="Gradient clipping"
    )

    # Learning rate scheduler
    parser.add_argument(
        "--lr_scheduler",
        default="cosine",
        type=str,
        choices=["cosine", "step"],
        help="LR scheduler",
    )
    parser.add_argument(
        "--lr_steps",
        default=[30, 60, 90],
        nargs="+",
        type=int,
        help="Epochs to decay learning rate (for step scheduler)",
    )
    parser.add_argument(
        "--lr_gamma",
        default=0.1,
        type=float,
        help="LR decay factor (for step scheduler)",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Epochs to warmup LR"
    )
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR")

    # Augmentation parameters
    parser.add_argument(
        "--mixup", type=float, default=0.8, help="Mixup alpha, mixup enabled if > 0"
    )
    parser.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha")
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="CutMix min/max ratio",
    )
    parser.add_argument(
        "--mixup_prob", type=float, default=1.0, help="Probability of mixup/cutmix"
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Switching probability between mixup and cutmix",
    )
    parser.add_argument(
        "--mixup_mode", type=str, default="batch", help="How to apply mixup/cutmix"
    )
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing")

    # Pretrained weights
    parser.add_argument(
        "--finetune",
        default="RETFound_mae_meh",
        type=str,
        help="Pretrained checkpoint name or path",
    )

    # Other parameters
    parser.add_argument(
        "--num_workers", default=8, type=int, help="Number of data loading workers"
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Start training
    main(args)
