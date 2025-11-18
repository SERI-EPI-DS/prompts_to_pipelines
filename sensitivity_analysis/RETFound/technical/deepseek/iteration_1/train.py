import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from timm.data import create_transform
from timm.loss import LabelSmoothingCrossEntropy

# Add the path to the RETFound directory
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../RETFound")
)  # This is the key line
import models_vit  # This should now work
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
import json
import time
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def get_args_parser():
    parser = argparse.ArgumentParser("RETFound fine-tuning", add_help=False)
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--update_freq", default=1, type=int, help="gradient accumulation steps"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="RETFound_mae",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.2,
        metavar="PCT",
        help="Drop path rate (default: 0.2)",
    )

    # RETFound weights
    parser.add_argument(
        "--finetune",
        default="../../RETFound/RETFound_CFP_weights.pth",
        type=str,
        help="RETFound pre-trained weights path",
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer betas",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        metavar="LR",
        help="learning rate (default: 5e-3)",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.65,
        help="layer-wise learning rate decay (default: 0.65)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs (default: 5)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " (default: rand-m9-mstd0.5-inc1)',
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path", default="../../data", type=str, help="dataset path"
    )
    parser.add_argument(
        "--output_dir",
        default="../../project/results",
        type=str,
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default=None, type=str, help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    return parser


def build_transform(is_train, args):
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
        )
        return transform
    else:
        return transforms.Compose(
            [
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def load_retfound_weights(model, checkpoint_path, args):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = checkpoint["model"]

    # Remove head.weight and head.bias as they don't match in size
    for k in ["head.weight", "head.bias"]:
        if (
            k in checkpoint_model
            and checkpoint_model[k].shape != model.state_dict()[k].shape
        ):
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # Interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # Load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Missing keys: {msg.missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")

    return model


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_epochs, num_training_epochs, min_lr_ratio=0.01
):
    """
    Create a schedule with a linear warmup followed by cosine decay.
    """

    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            # Linear warmup
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        else:
            # Cosine decay to min_lr_ratio * initial_lr
            progress = float(current_epoch - num_warmup_epochs) / float(
                max(1, num_training_epochs - num_warmup_epochs)
            )
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def main(args):
    # Initialize distributed mode (simplified for single GPU)
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build dataset
    data_path_train = os.path.join(args.data_path, "train")
    data_path_val = os.path.join(args.data_path, "val")

    # Calculate number of classes
    num_classes = len(
        [
            d
            for d in os.listdir(data_path_train)
            if os.path.isdir(os.path.join(data_path_train, d))
        ]
    )
    print(f"Number of classes: {num_classes}")

    # Build transforms
    transform_train = build_transform(is_train=True, args=args)
    transform_val = build_transform(is_train=False, args=args)

    # Create datasets
    dataset_train = ImageFolder(data_path_train, transform=transform_train)
    dataset_val = ImageFolder(data_path_val, transform=transform_val)

    # Create data loaders
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Build model
    # 1. First, let's see all available models for clarity
    print("\n=== Available models in models_vit ===")
    all_keys = list(models_vit.__dict__.keys())
    all_keys.sort()
    for key in all_keys:
        if not key.startswith("__") and not key.endswith("__"):
            print(f"  - {key}")

    # 2. Use the correct model name - choose ONE of these options:

    # Option A: Use RETFound_mae (specific RETFound model)
    try:
        print("Attempting to load RETFound_mae...")
        model = models_vit.__dict__["RETFound_mae"](
            num_classes=num_classes,
            drop_path_rate=args.drop_path,
            global_pool=True,
        )
        print("✓ Successfully loaded RETFound_mae")

    # Option B: Fallback to VisionTransformer (generic ViT)
    except KeyError:
        print("RETFound_mae not found, falling back to VisionTransformer...")
        model = models_vit.__dict__["VisionTransformer"](
            num_classes=num_classes,
            drop_path_rate=args.drop_path,
            global_pool=True,
        )
        print("✓ Successfully loaded VisionTransformer")

    # 3. Load RETFound weights if available
    if args.finetune and os.path.isfile(args.finetune):
        print(f"Loading pre-trained weights from {args.finetune}")
        model = load_retfound_weights(model, args.finetune, args)
    else:
        print("No pre-trained weights found, training from scratch")

    model.to(device)

    # Build optimizer
    optimizer = create_optimizer(args, model)

    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    # Learning rate schedule
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_epochs=args.warmup_epochs,
        num_training_epochs=args.epochs,
        min_lr_ratio=args.min_lr / args.lr,  # Convert absolute min_lr to ratio
    )
    # Scaler for mixed precision
    loss_scaler = NativeScaler()

    # Training variables
    max_accuracy = 0.0
    start_epoch = args.start_epoch

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Train one epoch
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            scheduler,
            args.update_freq,
        )

        # Validate
        test_stats = validate(model, data_loader_val, device, criterion)

        # Save checkpoint if best model
        if test_stats["acc1"] > max_accuracy:
            max_accuracy = test_stats["acc1"]
            checkpoint_path = os.path.join(args.output_dir, "checkpoint-best.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args,
                },
                checkpoint_path,
            )
            print(f"Best model saved with accuracy: {max_accuracy:.2f}%")

        # Save latest checkpoint
        checkpoint_path = os.path.join(args.output_dir, "checkpoint-latest.pth")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args,
            },
            checkpoint_path,
        )

        # Log stats
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
        }

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    clip_grad,
    scheduler,
    update_freq,
):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        step = data_iter_step // update_freq

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= update_freq
        loss_scaler(loss, optimizer, clip_grad=clip_grad, parameters=model.parameters())

        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        loss_meter.update(loss_value)
        norm_meter.update(0)  # Placeholder for gradient norm
        batch_time.update(time.time() - end)
        end = time.time()

        if data_iter_step % 100 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch: [{epoch}][{data_iter_step}/{num_steps}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"LR {lr:.6f}"
            )

    scheduler.step()
    return {"loss": loss_meter.avg, "lr": optimizer.param_groups[0]["lr"]}


def validate(model, data_loader, device, criterion):
    model.eval()
    criterion_val = torch.nn.CrossEntropyLoss()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    with torch.no_grad():
        for batch in data_loader:
            images, target = batch
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion_val(output, target)

            # Measure accuracy - this will now handle few classes gracefully
            acc_results = accuracy(output, target, topk=(1, 5))
            acc1 = acc_results[0]
            acc5 = (
                acc_results[1] if len(acc_results) > 1 else acc_results[0]
            )  # Handle case with only 1 metric

            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1.item(), images.size(0))
            acc5_meter.update(acc5.item(), images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(f"* Val Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")

    return {"loss": loss_meter.avg, "acc1": acc1_meter.avg, "acc5": acc5_meter.avg}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Safety check: ensure we don't ask for more classes than exist
        num_classes = output.size(1)
        if maxk > num_classes:
            maxk = num_classes
            print(
                f"Warning: Adjusting maxk from {max(topk)} to {num_classes} (number of classes)"
            )

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Adjust k if it's larger than available classes
            k_adj = min(k, num_classes)
            correct_k = (
                correct[:k_adj].contiguous().view(-1).float().sum(0, keepdim=True)
            )
            res.append(correct_k.mul_(100.0 / batch_size))

        # Handle case where original topk values requested more classes than available
        while len(res) < len(topk):
            res.append(torch.tensor([0.0], device=output.device))

        return res


def create_optimizer(args, model):
    # Set weight decay for different parameters
    skip_list = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip_list = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = set_weight_decay(model, skip_list, skip_keywords, args.weight_decay)

    opt_lower = args.opt.lower()
    optimizer = None
    if opt_lower == "sgd":
        optimizer = optim.SGD(
            parameters,
            momentum=args.momentum,
            nesterov=True,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(
            parameters,
            eps=args.opt_eps,
            betas=args.opt_betas if args.opt_betas is not None else (0.9, 0.999),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.05):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or any(skip_keyword in name for skip_keyword in skip_keywords)
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [
        {"params": has_decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


if __name__ == "__main__":
    from datetime import timedelta
    import math

    args = get_args_parser()
    args = args.parse_args()

    # Set additional args needed for training
    args.min_lr = 1e-6
    args.warmup_epochs = 5

    main(args)
