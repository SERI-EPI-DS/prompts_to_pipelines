import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import models_vit  # Your RETFound model module
import util.misc as misc

# from util.datasets import build_dataset
from torchvision import datasets, transforms
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import argparse
import os
import time
import json
from pathlib import Path


def build_dataset(is_train, args):
    """
    Build dataset for training or evaluation

    Args:
        is_train (bool): Whether building training dataset
        args: Command line arguments containing data_path
    """
    # Determine the appropriate subdirectory based on is_train
    if is_train:
        split_dir = "train"
    else:
        split_dir = "val"  # or 'test' depending on your use case

    # Construct the full path
    root = os.path.join(args.data_path, split_dir)

    print(f"Loading dataset from: {root}")

    # Create appropriate transform
    transform = build_transform(is_train, args)

    # Create dataset
    dataset = datasets.ImageFolder(root=root, transform=transform)

    return dataset


def build_transform(is_train, args):
    """
    Build data transforms for training or evaluation
    """
    if is_train:
        # Training transforms with data augmentation
        return transforms.Compose(
            [
                transforms.Resize((args.input_size, args.input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        # Validation transforms (no augmentation)
        return transforms.Compose(
            [
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def setup_distributed():
    """Initialize distributed training using environment variables set by torchrun."""
    # These are set by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set the current CUDA device
    torch.cuda.set_device(local_rank)

    # Initialize the process group
    dist.init_process_group(backend="nccl", init_method="env://")


def get_available_models():
    """Check available models in models_vit module."""
    available_models = [
        k
        for k in models_vit.__dict__.keys()
        if not k.startswith("_") and callable(models_vit.__dict__[k])
    ]
    print("Available models in models_vit:")
    for model in available_models:
        print(f"  - {model}")
    return available_models


def get_args_parser():
    parser = argparse.ArgumentParser("RETFound fine-tuning", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="RETFound_mae",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input_size", default=224, type=int, help="image input size")
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.2,
        metavar="PCT",
        help="Drop path rate (default: 0.2)",
    )

    # Training parameters
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", default=50, type=int)
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
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path", default="./dataset/", type=str, help="dataset path"
    )
    parser.add_argument(
        "--nb_classes", default=5, type=int, help="number of the classification types"
    )
    parser.add_argument(
        "--output_dir", default="./output_finetune/", help="path where to save output"
    )
    parser.add_argument(
        "--log_dir", default="./output_finetune/", help="path where to save logs"
    )

    # RETFound weights
    parser.add_argument(
        "--finetune",
        default="./RETFound_cfp_weights.pth",
        help="finetune from checkpoint",
    )

    return parser


def main(args):
    # First, check available models
    available_models = get_available_models()

    # Check if the requested model is available
    if args.model not in available_models:
        print(f"Warning: Model '{args.model}' not found in available models.")
        print(f"Using default model 'vit_base_patch16' instead.")
        args.model = "vit_base_patch16"  # Fallback to a known good model

    # Initialize distributed mode
    setup_distributed()

    # Fix the seed for reproducibility
    seed = 42 + misc.get_rank()
    torch.manual_seed(seed)

    # Build model
    print(f"Creating model: {args.model}")
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=True,
    )

    # Load RETFound pre-trained weights
    if args.finetune and os.path.isfile(args.finetune):
        print(f"Loading pre-trained weights from {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location="cpu")

        checkpoint_model = checkpoint["model"]

        # Interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # Load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"Load result: {msg}")

        # Manually initialize the classification head
        trunc_normal_(model.head.weight, std=2e-5)
        if model.head.bias is not None:
            nn.init.constant_(model.head.bias, 0)
    else:
        print(f"Warning: Pre-trained weights not found at {args.finetune}")
        print("Training from scratch...")

    # Wrap model with DDP
    model = model.cuda()
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # Build datasets
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    # Create data loaders
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
    )

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Build optimizer and loss function
    param_groups = model.parameters()
    optimizer = optim.AdamW(param_groups, lr=args.blr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        # Train for one epoch
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, loss_fn, epoch, args
        )

        # Evaluate
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            eval_stats = evaluate(data_loader_val, model, loss_fn, args)

            # Save checkpoint
            if args.output_dir and misc.is_main_process():
                checkpoint_path = os.path.join(
                    args.output_dir, f"checkpoint-{epoch}.pth"
                )
                misc.save_on_master(
                    {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

    # Save final model
    if args.output_dir and misc.is_main_process():
        final_path = os.path.join(args.output_dir, "checkpoint-final.pth")
        misc.save_on_master(
            {
                "model": model.module.state_dict(),
                "args": args,
            },
            final_path,
        )


def train_one_epoch(model, data_loader, optimizer, loss_fn, epoch, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0].cuda()
        labels = batch[1].cuda()

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        metric_logger.update(loss=loss.item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(data_loader, model, loss_fn, args):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0].cuda()
            labels = batch[1].cuda()

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Calculate accuracy
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean()

            metric_logger.update(loss=loss.item())
            metric_logger.update(acc=acc.item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
