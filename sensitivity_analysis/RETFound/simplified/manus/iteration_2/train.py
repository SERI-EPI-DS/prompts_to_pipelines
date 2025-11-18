#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import time
import numpy as np
from pathlib import Path
import urllib.request

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import timm
from timm.models.vision_transformer import VisionTransformer
from timm.scheduler import CosineLRScheduler


def get_args_parser():
    parser = argparse.ArgumentParser(
        "RETFound Fine-tuning for Image Classification", add_help=False
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        default="RETFound_cfp",
        type=str,
        choices=["RETFound_cfp", "RETFound_oct"],
        help="Name of model to train (RETFound_cfp for color fundus, RETFound_oct for OCT)",
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    # Training parameters
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter", default=1, type=int, help="Gradient accumulation steps"
    )
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
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
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
    parser.add_argument(
        "--finetune_strategy",
        default="full_finetune",
        choices=["linear_probe", "full_finetune", "hybrid"],
        help="Fine-tuning strategy",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path", default="/path/to/your/dataset", type=str, help="dataset path"
    )
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
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


def init_distributed_mode(args):
    """Initialize distributed training if available"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()


def download_retfound_weights(model_name, cache_dir="./retfound_cache"):
    """Download RETFound weights from the correct URLs"""
    os.makedirs(cache_dir, exist_ok=True)

    # Direct download URLs for RETFound weights
    weight_urls = {
        "RETFound_cfp": "https://drive.google.com/uc?export=download&id=1Nt6R3JgkPZdJF8CzeOr2LTS6yKV8XhZb",
        "RETFound_oct": "https://drive.google.com/uc?export=download&id=1l62zbWUFTlp214SvlQVwFNNlpx2rlJhL",
    }

    if model_name not in weight_urls:
        raise ValueError(
            f"Unknown model name: {model_name}. Available: {list(weight_urls.keys())}"
        )

    weight_path = os.path.join(cache_dir, f"{model_name}_weights.pth")

    if os.path.exists(weight_path):
        print(f"Using cached weights: {weight_path}")
        return weight_path

    print(f"Downloading {model_name} weights...")
    try:
        # Try alternative download method for Google Drive
        import gdown

        file_id = weight_urls[model_name].split("id=")[1]
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}", weight_path, quiet=False
        )
    except ImportError:
        print("gdown not available, trying direct download...")
        try:
            urllib.request.urlretrieve(weight_urls[model_name], weight_path)
        except Exception as e:
            print(f"Direct download failed: {e}")
            print("Please manually download the weights from:")
            print(
                f"CFP weights: https://drive.google.com/file/d/1Nt6R3JgkPZdJF8CzeOr2LTS6yKV8XhZb/view"
            )
            print(
                f"OCT weights: https://drive.google.com/file/d/1l62zbWUFTlp214SvlQVwFNNlpx2rlJhL/view"
            )
            print(f"Save as: {weight_path}")
            raise
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please manually download the weights from:")
        print(
            f"CFP weights: https://drive.google.com/file/d/1Nt6R3JgkPZdJF8CzeOr2LTS6yKV8XhZb/view"
        )
        print(
            f"OCT weights: https://drive.google.com/file/d/1l62zbWUFTlp214SvlQVwFNNlpx2rlJhL/view"
        )
        print(f"Save as: {weight_path}")
        raise

    print(f"Downloaded weights to: {weight_path}")
    return weight_path


def filter_checkpoint_keys(checkpoint_state_dict):
    """Filter out MAE decoder keys and other incompatible keys"""
    filtered_state_dict = {}

    # Keys to exclude (MAE decoder components and other incompatible keys)
    exclude_prefixes = [
        "mask_token",
        "decoder_pos_embed",
        "decoder_embed",
        "decoder_blocks",
        "decoder_norm",
        "decoder_pred",
    ]

    for key, value in checkpoint_state_dict.items():
        # Skip decoder-related keys
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            print(f"Skipping decoder key: {key}")
            continue

        # Keep encoder keys
        filtered_state_dict[key] = value

    return filtered_state_dict


def interpolate_pos_embed(model, checkpoint_model):
    """Interpolate position embeddings if needed"""
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


def create_model(model_name, num_classes, drop_path_rate=0.1):
    """Create Vision Transformer model compatible with RETFound weights"""

    # Create a ViT-Large model similar to RETFound architecture
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,  # We'll load RETFound weights manually
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        global_pool="token",  # Use [CLS] token for classification
    )

    # Try to load RETFound pre-trained weights
    try:
        print(f"Attempting to load {model_name} weights...")

        # Download weights
        weight_path = download_retfound_weights(model_name)

        checkpoint = torch.load(weight_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        print(f"Original checkpoint has {len(state_dict)} keys")

        # Filter out MAE decoder keys
        filtered_state_dict = filter_checkpoint_keys(state_dict)
        print(f"Filtered checkpoint has {len(filtered_state_dict)} keys")

        # Remove head weights if number of classes doesn't match
        head_weight_key = "head.weight"
        head_bias_key = "head.bias"

        if head_weight_key in filtered_state_dict:
            if filtered_state_dict[head_weight_key].shape[0] != num_classes:
                print(
                    f"Removing head weights: pretrained classes={filtered_state_dict[head_weight_key].shape[0]}, target classes={num_classes}"
                )
                del filtered_state_dict[head_weight_key]
                del filtered_state_dict[head_bias_key]
        else:
            print("No head weights found in checkpoint - will be randomly initialized")

        # Interpolate position embeddings if needed
        interpolate_pos_embed(model, filtered_state_dict)

        # Load state dict with strict=False to handle missing head weights
        missing_keys, unexpected_keys = model.load_state_dict(
            filtered_state_dict, strict=False
        )

        if missing_keys:
            print(f"Missing keys (will be randomly initialized): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys}")

        print(f"Successfully loaded pre-trained encoder weights from {model_name}")

        # Initialize head weights properly
        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            torch.nn.init.trunc_normal_(model.head.weight, std=2e-5)
            torch.nn.init.constant_(model.head.bias, 0)
            print("Initialized classification head weights")

    except Exception as e:
        print(f"Warning: Could not load pre-trained weights: {e}")
        print("Training from ImageNet pre-trained weights...")

        # Fall back to ImageNet pre-trained weights
        try:
            model = timm.create_model(
                "vit_large_patch16_224",
                pretrained=True,
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                global_pool="token",
            )
            print("Loaded ImageNet pre-trained ViT-Large weights")
        except Exception as e2:
            print(f"Warning: Could not load ImageNet weights either: {e2}")
            print("Training from scratch...")

    return model


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    max_norm=None,
    log_writer=None,
    args=None,
):
    """Training loop for one epoch"""
    model.train(True)

    running_loss = 0.0
    num_batches = len(data_loader)

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= args.accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % args.accum_iter == 0,
        )

        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        running_loss += loss_value

        if data_iter_step % 50 == 0:
            print(
                f"Epoch: {epoch}, Step: {data_iter_step}/{num_batches}, Loss: {loss_value:.4f}"
            )

    avg_loss = running_loss / num_batches
    return {"loss": avg_loss}


def evaluate(data_loader, model, device):
    """Evaluation function"""
    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluation mode
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            images = batch[0]
            target = batch[1]
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            total_correct += correct[:1].reshape(-1).float().sum(0).item()
            total_samples += target.size(0)
            total_loss += loss.item()

    acc1 = 100.0 * total_correct / total_samples
    avg_loss = total_loss / len(data_loader)

    print(f"* Acc@1 {acc1:.3f} Loss {avg_loss:.3f}")

    return {
        "acc1": acc1,
        "acc5": acc1,
        "loss": avg_loss,
    }  # acc5 same as acc1 for simplicity


class NativeScalerWithGradNormCount:
    """Gradient scaler for mixed precision training"""

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
                norm = None
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    """Save model checkpoint"""
    output_dir = Path(args.output_dir)
    checkpoint_paths = [output_dir / ("checkpoint-%s.pth" % epoch)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scaler": loss_scaler._scaler.state_dict(),
            "args": args,
        }
        torch.save(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    """Load model checkpoint for resuming training"""
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Load model state
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            state_dict, strict=False
        )

        if missing_keys:
            print(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading checkpoint: {unexpected_keys}")

        print("Resume checkpoint %s" % args.resume)

        # Load optimizer and other training state if available
        if "optimizer" in checkpoint and "epoch" in checkpoint and not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler._scaler.load_state_dict(checkpoint["scaler"])
            print("Resumed optimizer and training state!")


def main(args):
    init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + (args.rank if hasattr(args, "rank") else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Data augmentation and normalization
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.input_size, scale=(0.2, 1.0), interpolation=3
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(args.input_size, interpolation=3),
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
    print(f"Classes: {dataset_train.classes}")

    # Create data loaders
    if getattr(args, "distributed", False):
        sampler_train = torch.utils.data.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
    else:
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
    model = create_model(
        args.model_name, num_classes=args.nb_classes, drop_path_rate=0.1
    )

    # Apply fine-tuning strategy
    if args.finetune_strategy == "linear_probe":
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False
        print("Linear probing: Only training the classification head")

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of params (M): %.2f" % (n_parameters / 1.0e6))

    # Distributed training setup
    if getattr(args, "distributed", False):
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Calculate effective batch size and learning rate
    eff_batch_size = (
        args.batch_size
        * args.accum_iter
        * (args.world_size if getattr(args, "distributed", False) else 1)
    )

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("Actual lr: %.2e" % args.lr)
    print("Effective batch size: %d" % eff_batch_size)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    loss_scaler = NativeScalerWithGradNormCount()
    criterion = torch.nn.CrossEntropyLoss()

    # Load checkpoint if resuming (this is for resuming training, not loading pretrained weights)
    load_model(args, model_without_ddp, optimizer, loss_scaler)

    # Setup logging
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
        exit(0)

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if getattr(args, "distributed", False):
            data_loader_train.sampler.set_epoch(epoch)

        # Hybrid fine-tuning: switch to full fine-tuning after warmup epochs
        if args.finetune_strategy == "hybrid" and epoch == args.warmup_epochs:
            print(f"Switching to full fine-tuning at epoch {epoch}")
            for name, param in model.named_parameters():
                param.requires_grad = True

            # Update optimizer to include all parameters
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=args.weight_decay,
            )

        # Training
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args,
        )

        # Validation
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        # Save checkpoint
        if args.output_dir:
            save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler)

        # Logging
        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)
            log_writer.add_scalar("perf/train_loss", train_stats["loss"], epoch)

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
    import math
    import sys

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
