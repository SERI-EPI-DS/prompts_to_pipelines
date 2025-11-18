#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import numpy as np
import os
import time
import traceback
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

try:
    import timm
    import timm.optim.optim_factory as optim_factory
    from timm.data.mixup import Mixup
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

    TIMM_AVAILABLE = True
    print(f"timm version: {timm.__version__}")
except ImportError as e:
    print(f"Warning: timm not available. Error: {e}")
    TIMM_AVAILABLE = False

import util.misc_fixed as misc
from util.misc_fixed import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune_fixed import train_one_epoch, evaluate
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed


def test_model_creation():
    """Test model creation with various configurations"""
    if not TIMM_AVAILABLE:
        print("timm not available for testing")
        return None

    test_models = [
        "vit_large_patch16_224",
        "vit_base_patch16_224",
        "vit_small_patch16_224",
    ]

    # Test different global_pool values for timm 0.9.16
    global_pool_options = ["avg", "token", ""]

    for model_name in test_models:
        for global_pool in global_pool_options:
            try:
                print(
                    f"Testing model creation: {model_name} with global_pool='{global_pool}'"
                )
                model = timm.create_model(
                    model_name,
                    pretrained=False,
                    num_classes=2,
                    drop_path_rate=0.1,
                    global_pool=global_pool,
                )
                print(
                    f"✓ Successfully created {model_name} with global_pool='{global_pool}'"
                )
                return model_name, global_pool
            except Exception as e:
                print(
                    f"✗ Failed to create {model_name} with global_pool='{global_pool}': {e}"
                )

    return None, None


def get_args_parser():
    parser = argparse.ArgumentParser(
        "RETFound Fine-tuning and Evaluation", add_help=False
    )

    # Basic parameters
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size = batch_size * accum_iter * # gpus)",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Gradient accumulation steps (for gradient accumulation)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16_224",
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
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        "--global_pool",
        type=str,
        default="avg",
        help="Global pooling type: avg, token, or empty string",
    )
    parser.add_argument(
        "--cls_token",
        action="store_true",
        help="Use class token instead of global pool for classification",
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
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)"',
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # Random Erase params
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

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
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

    # Dataset parameters
    parser.add_argument(
        "--data_path", default="/datasets/MESSIDOR2/", type=str, help="dataset path"
    )
    parser.add_argument(
        "--nb_classes", default=5, type=int, help="number of the classification types"
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
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


def main(args):
    if not TIMM_AVAILABLE:
        print(
            "Error: timm is required but not installed. Please install with: pip install timm"
        )
        return

    # Handle cls_token argument (convert to global_pool setting)
    if args.cls_token:
        args.global_pool = "token"

    # Test model creation first
    print("Testing model creation...")
    working_model, working_global_pool = test_model_creation()
    if working_model is None:
        print(
            "Error: Could not create any ViT model. Please check your timm installation."
        )
        return

    # Use the working model and global_pool if the requested ones don't work
    if working_model != args.model:
        print(f"Using {working_model} instead of {args.model}")
        args.model = working_model

    if working_global_pool != args.global_pool:
        print(
            f"Using global_pool='{working_global_pool}' instead of '{args.global_pool}'"
        )
        args.global_pool = working_global_pool

    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % sampler_train)
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

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

    # Create model using timm with correct global_pool parameter
    print(f"Creating model: {args.model}")
    print(
        f"Model parameters: num_classes={args.nb_classes}, drop_path_rate={args.drop_path}, global_pool='{args.global_pool}'"
    )

    try:
        model = timm.create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
        print(f"Model created successfully: {model.__class__.__name__}")
    except Exception as e:
        print(f"Error creating model: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return

    if args.finetune and not args.eval:
        print(f"Loading checkpoint from: {args.finetune}")
        try:
            checkpoint = torch.load(args.finetune, map_location="cpu")
            print("Checkpoint loaded successfully")

            if "model" in checkpoint:
                checkpoint_model = checkpoint["model"]
                print(
                    f"Found 'model' key in checkpoint with {len(checkpoint_model)} parameters"
                )
            else:
                checkpoint_model = checkpoint
                print(
                    f"Using checkpoint directly with {len(checkpoint_model)} parameters"
                )

            state_dict = model.state_dict()
            print(f"Model state dict has {len(state_dict)} parameters")

            # Remove head weights if they don't match
            for k in ["head.weight", "head.bias"]:
                if k in checkpoint_model and k in state_dict:
                    if checkpoint_model[k].shape != state_dict[k].shape:
                        print(
                            f"Removing key {k} from pretrained checkpoint (shape mismatch: {checkpoint_model[k].shape} vs {state_dict[k].shape})"
                        )
                        del checkpoint_model[k]
                elif k in checkpoint_model:
                    print(f"Key {k} in checkpoint but not in model")
                elif k in state_dict:
                    print(f"Key {k} in model but not in checkpoint")

            # Remove fc_norm weights if they don't match (for global pooling)
            for k in ["fc_norm.weight", "fc_norm.bias"]:
                if k in checkpoint_model and k not in state_dict:
                    print(
                        f"Removing key {k} from pretrained checkpoint (not in current model)"
                    )
                    del checkpoint_model[k]

            # interpolate position embedding
            print("Interpolating position embeddings...")
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            print("Loading pre-trained weights...")
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(f"Load pretrained model with msg: {msg}")

            # manually initialize fc layer
            if hasattr(model, "head") and hasattr(model.head, "weight"):
                torch.nn.init.trunc_normal_(model.head.weight, std=2e-5)
                print("Initialized head layer")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            print("Continuing without pre-trained weights...")

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    # Get no_weight_decay list if available
    no_weight_decay_list = set()
    if hasattr(model_without_ddp, "no_weight_decay"):
        no_weight_decay_list = model_without_ddp.no_weight_decay()

    try:
        param_groups = optim_factory.param_groups_lrd(
            model_without_ddp,
            args.weight_decay,
            no_weight_decay_list=no_weight_decay_list,
            layer_decay=args.layer_decay,
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        print("Created optimizer with layer-wise learning rate decay")
    except Exception as e:
        print(f"Error creating optimizer with layer decay: {e}")
        print("Falling back to standard optimizer...")
        optimizer = torch.optim.AdamW(
            model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        acc1_value = test_stats["acc1"]
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_value:.1f}%"
        )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
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
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir:
            misc.save_model(
                args=args,
                epoch=epoch,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
            )

        test_stats = evaluate(data_loader_val, model, device)
        acc1_value = test_stats["acc1"]
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_value:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
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
