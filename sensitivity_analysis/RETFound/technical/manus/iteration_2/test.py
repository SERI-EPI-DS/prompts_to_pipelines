#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script conducts testing of the final weights of the classifier on a held-out test portion of the dataset.
Self-contained version that doesn't require RETFound repository imports.
"""
import argparse
import csv
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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


def get_args_parser():
    parser = argparse.ArgumentParser(
        "RETFound testing for image classification", add_help=False
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to the fine-tuned model weights",
    )
    parser.add_argument(
        "--data_path", required=True, type=str, help="Path to the test dataset"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Path to save the results CSV file",
    )
    parser.add_argument("--input_size", default=224, type=int, help="Images input size")
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of data loading workers"
    )
    parser.add_argument("--device", default="cuda", help="Device to use for testing")
    parser.add_argument(
        "--nb_classes", default=2, type=int, help="Number of the classification types"
    )
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to test",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument("--global_pool", action="store_true", default=True)

    return parser


def main(args):
    device = torch.device(args.device)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Data transformation for testing
    transform_test = transforms.Compose(
        [
            transforms.Resize(
                args.input_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load test dataset
    dataset_test = ImageFolder(
        os.path.join(args.data_path, "test"), transform=transform_test
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print(f"Test dataset size: {len(dataset_test)}")
    print(f"Number of classes: {len(dataset_test.classes)}")
    print(f"Classes: {dataset_test.classes}")

    # Create model
    model = create_model(
        model_name=args.model,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load trained weights
    print(f"Loading model weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Load state dict
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model with msg: {msg}")

    model.to(device)
    model.eval()

    # Run inference
    results = []
    correct = 0
    total = 0

    print("Starting inference...")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader_test):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            scores = nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # Calculate accuracy
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Store results for each image in the batch
            for i in range(images.size(0)):
                # Get the original image path
                img_idx = batch_idx * args.batch_size + i
                if img_idx < len(dataset_test.samples):
                    image_path, true_label = dataset_test.samples[img_idx]
                    image_name = os.path.basename(image_path)

                    # Create result dictionary
                    result = {"image_name": image_name}

                    # Add scores for each class
                    for j, class_name in enumerate(dataset_test.classes):
                        result[f"{class_name}_score"] = scores[i][j].item()

                    # Add predicted class
                    result["predicted_class"] = dataset_test.classes[predicted[i]]
                    result["true_class"] = dataset_test.classes[true_label]
                    result["correct"] = (predicted[i] == targets[i]).item()

                    results.append(result)

            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(data_loader_test)}")

    # Calculate final accuracy
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")

    # Save results to CSV
    output_file = os.path.join(args.output_dir, "test_results.csv")

    if results:
        fieldnames = (
            ["image_name"]
            + [f"{class_name}_score" for class_name in dataset_test.classes]
            + ["predicted_class", "true_class", "correct"]
        )

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Test results saved to {output_file}")
        print(f"Total images processed: {len(results)}")

        # Print class-wise accuracy
        class_correct = {class_name: 0 for class_name in dataset_test.classes}
        class_total = {class_name: 0 for class_name in dataset_test.classes}

        for result in results:
            true_class = result["true_class"]
            class_total[true_class] += 1
            if result["correct"]:
                class_correct[true_class] += 1

        print("\nClass-wise accuracy:")
        for class_name in dataset_test.classes:
            if class_total[class_name] > 0:
                class_acc = 100 * class_correct[class_name] / class_total[class_name]
                print(
                    f"{class_name}: {class_acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})"
                )
    else:
        print("No results to save!")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
