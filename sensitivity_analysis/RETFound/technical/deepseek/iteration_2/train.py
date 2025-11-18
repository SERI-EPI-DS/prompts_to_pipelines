import os
import sys

sys.path.append("../RETFound")

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy
from torchvision.datasets import ImageFolder
from functools import partial

# Import from the local RETFound modules
import models_vit
from util.pos_embed import interpolate_pos_embed
from util.lr_decay import param_groups_lrd
from engine_finetune import train_one_epoch, evaluate


# --- NativeScaler Implementation ---
class NativeScalerWithGradAccum:
    """
    Native scaler for mixed-precision training, adapted for compatibility with RETFound's engine_finetune.py.
    """

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        parameters=None,
        update_grad=True,
        clip_grad=None,
        create_graph=False,
    ):
        # The `create_graph` argument is captured to maintain compatibility but is not used in this implementation.
        self._scaler.scale(loss).backward(create_graph=create_graph)

        if update_grad:
            if clip_grad is not None:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            self._scaler.step(optimizer)
            self._scaler.update()
            optimizer.zero_grad()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


# For compatibility with the original code, alias the class
NativeScaler = NativeScalerWithGradAccum


def get_args_parser():
    parser = argparse.ArgumentParser("RETFound fine-tuning", add_help=False)
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--update_freq", default=1, type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Gradient accumulation steps. Effective batch size = batch_size * accum_iter",
    )

    # Model parameters - UPDATED with correct model name
    # parser.add_argument('--model', default='vit_large_patch16', type=str,
    #                    help='Name of model to train')
    parser.add_argument("--input_size", default=224, type=int, help="image input size")
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.2,
        metavar="PCT",
        help="Drop path rate (default: 0.2)",
    )
    parser.add_argument(
        "--img_size",
        default=224,
        type=int,
        help="Image size (default: 224 for RETFound_CFP)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None)",
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
        metavar="LR",
        help="lower lr bound for cyclic schedulers (default: 1e-6)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="epochs to warmup LR, if scheduler supports",
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
        help="AutoAugment policy (default: rand-m9-mstd0.5-inc1)",
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )

    # Dataset parameters
    parser.add_argument("--data_path", default="../data", type=str, help="dataset path")
    parser.add_argument(
        "--output_dir",
        default="../project/results",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        help="Task name for evaluation output directory",
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin_mem", action="store_true", help="Pin CPU memory in DataLoader"
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):

    # --- Distributed setup is now handled by torchrun ---
    # The environment variables RANK, LOCAL_RANK, and WORLD_SIZE are set automatically by torchrun :cite[2]
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")  # Now the required env vars exist

    # Fix the seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create output directory (only on main process)
    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Data loading and transformations
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = ImageFolder(
        os.path.join(args.data_path, "train"), transform=train_transform
    )
    val_dataset = ImageFolder(
        os.path.join(args.data_path, "val"), transform=val_transform
    )

    # Create data loaders with DistributedSampler
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )
    val_sampler = torch.utils.data.DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Calculate number of classes
    nb_classes = len(train_dataset.classes)
    if dist.get_rank() == 0:
        print(f"Number of classes: {nb_classes}")

    # --- Model creation with error handling ---
    try:
        model = models_vit.__dict__["VisionTransformer"](
            img_size=args.input_size,
            patch_size=16,
            in_chans=3,
            num_classes=nb_classes,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=args.drop_path,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            global_pool=True,
        )
        if dist.get_rank() == 0:
            print("Successfully created VisionTransformer model")

    except KeyError:
        if dist.get_rank() == 0:
            print(f"Error creating model: {e}")
            print("Available models in models_vit:")
            for key in models_vit.__dict__.keys():
                if not key.startswith("_") and isinstance(
                    models_vit.__dict__[key], type
                ):
                    print(f"  - {key}")
        raise

    # Load RETFound pre-trained weights
    checkpoint = torch.load("../RETFound/RETFound_CFP_weights.pth", map_location="cpu")
    checkpoint_model = checkpoint["model"]

    # Interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # Load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    if dist.get_rank() == 0:
        print(msg)

    # Manually initialize classifier head
    if hasattr(model, "head") and model.head is not None:
        torch.nn.init.trunc_normal_(model.head.weight, std=2e-5)

    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Create optimizer with layer-wise learning rate decay
    param_groups = param_groups_lrd(
        model.module,
        args.weight_decay,
        no_weight_decay_list=model.module.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # Loss function with label smoothing
    if args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    # Training loop
    if dist.get_rank() == 0:
        print("Starting training...")
    max_accuracy = 0.0

    import inspect

    if dist.get_rank() == 0:
        sig = inspect.signature(train_one_epoch)
        print("train_one_epoch parameters:", list(sig.parameters.keys()))

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        # Remove the update_freq parameter from the function call
        # Remove the print_freq parameter from the function call
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=args.device,  # or local_rank if that's what your function expects
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,  # maps your 'clip_grad' argument to 'max_norm'
            # mixup_fn=None,  # Uncomment and set if you use mixup
            # log_writer=None,  # Uncomment and set if you have a logger
            args=args,  # Uncomment if your function version requires the full args object
        )

        lr_scheduler.step(epoch)

        # Evaluate on validation set
        # Update the evaluate function call with all required arguments
        val_stats = evaluate(
            data_loader=val_loader,
            model=model,
            device=args.device,
            args=args,  # Your training arguments object
            epoch=epoch,  # Current epoch number
            mode="val",  # Evaluation mode - 'val' for validation
            num_class=nb_classes,  # Number of classes in your dataset
            log_writer=None,  # Can be None if you don't have a logger
        )

        # Debug: Check what val_stats contains
        if dist.get_rank() == 0:
            print(f"Debug - val_stats type: {type(val_stats)}")
            print(f"Debug - val_stats contents: {val_stats}")

        # Extract accuracy based on the actual return type
        if dist.get_rank() == 0:
            if isinstance(val_stats, dict):
                # If it's a dictionary, access as before
                accuracy = val_stats["acc1"]
                print(f"Epoch {epoch}: Val Accuracy: {accuracy:.2f}%")
            else:
                # If it's a tuple, you'll need to figure out which element contains the accuracy
                # This might be val_stats[0] or another index - check the debug output above
                accuracy = val_stats[1]  # Adjust this index based on debug output
                print(f"Epoch {epoch}: Val Accuracy: {accuracy:.2f}%")

        # Save best model
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            checkpoint_path = os.path.join(args.output_dir, "checkpoint-best.pth")
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "max_accuracy": max_accuracy,
                    "args": args,
                },
                checkpoint_path,
            )
            print(f"Saved best model with accuracy: {max_accuracy:.2f}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RETFound training", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
