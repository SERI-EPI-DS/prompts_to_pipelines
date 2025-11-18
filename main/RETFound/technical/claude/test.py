import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

# Add RETFound directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "RETFound"))
from util.pos_embed import interpolate_pos_embed


def get_args_parser():
    parser = argparse.ArgumentParser("RETFound Testing", add_help=False)

    # Data parameters
    parser.add_argument("--data_path", default="./data/", type=str, help="dataset path")
    parser.add_argument(
        "--output_dir", default="./project/results/", help="path where to save results"
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint path to load")

    # Model parameters
    parser.add_argument(
        "--model", default="vit_large_patch16", type=str, help="Name of model"
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--drop_path", type=float, default=0.2, help="Drop path rate")
    parser.add_argument("--global_pool", action="store_true", default=True)

    # Dataset parameters
    parser.add_argument("--batch_size", default=16, type=int, help="Per GPU batch size")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        default=True,
        help="Pin CPU memory in DataLoader",
    )

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)

    # Misc
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    return parser


def build_dataset_test(args):
    transform = build_transform(False, args)
    dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "test"), transform=transform
    )
    return dataset


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        raise ValueError("This script is for testing only")

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


# Vision Transformer components (same as in train.py)
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


@torch.no_grad()
def test(args):
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Data loading
    dataset_test = build_dataset_test(args)

    # Load class names from training
    output_dir = Path(args.output_dir)
    with open(output_dir / "classes.json", "r") as f:
        classes = json.load(f)

    nb_classes = len(classes)
    print(f"Number of classes: {nb_classes}")
    print(f"Classes: {classes}")

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Model
    print(f"Creating model: {args.model}")
    model = vit_large_patch16(
        num_classes=nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print("Load checkpoint from: %s" % args.checkpoint)

    # Handle different checkpoint formats
    if "model" in checkpoint:
        checkpoint_model = checkpoint["model"]
    elif "model_ema" in checkpoint:
        checkpoint_model = checkpoint["model_ema"]
    else:
        checkpoint_model = checkpoint

    # Load model weights
    msg = model.load_state_dict(checkpoint_model, strict=True)
    print(msg)

    model.to(device)
    model.eval()

    # Prepare results storage
    results = []

    print("Starting evaluation...")

    for batch_idx, (images, targets) in enumerate(data_loader_test):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)

        # Get predictions
        _, predicted = outputs.max(1)

        # Process each sample in the batch
        for i in range(images.size(0)):
            sample_idx = batch_idx * args.batch_size + i
            if sample_idx < len(dataset_test.samples):
                img_path, true_label = dataset_test.samples[sample_idx]
                img_name = os.path.basename(img_path)

                # Get scores for each class
                scores = probabilities[i].cpu().numpy()
                pred_class_idx = predicted[i].item()
                pred_class_name = classes[pred_class_idx]
                true_class_name = classes[true_label]

                # Create result entry
                result_entry = {
                    "image_name": img_name,
                    "image_path": img_path,
                    "true_class": true_class_name,
                    "predicted_class": pred_class_name,
                    "correct": pred_class_idx == true_label,
                }

                # Add class scores
                for j, class_name in enumerate(classes):
                    result_entry[f"score_{class_name}"] = float(scores[j])

                results.append(result_entry)

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Processed {(batch_idx + 1) * args.batch_size} / {len(dataset_test)} images"
            )

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate accuracy
    accuracy = results_df["correct"].mean() * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Calculate per-class accuracy
    class_accuracies = results_df.groupby("true_class")["correct"].agg(
        ["mean", "count"]
    )
    class_accuracies["accuracy"] = class_accuracies["mean"] * 100
    print("\nPer-Class Accuracy:")
    print(class_accuracies[["accuracy", "count"]])

    # Save results
    results_path = output_dir / "test_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Save summary statistics
    summary = {
        "overall_accuracy": accuracy,
        "total_samples": len(results_df),
        "per_class_accuracy": class_accuracies["accuracy"].to_dict(),
        "per_class_counts": class_accuracies["count"].to_dict(),
        "checkpoint_used": args.checkpoint,
        "model": args.model,
        "input_size": args.input_size,
    }

    summary_path = output_dir / "test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to: {summary_path}")

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report

    y_true = results_df["true_class"].values
    y_pred = results_df["predicted_class"].values

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Save confusion matrix
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")

    # Generate classification report
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = output_dir / "classification_report.csv"
    report_df.to_csv(report_path)
    print(f"Classification report saved to: {report_path}")

    print("\nTesting completed successfully!")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    test(args)
