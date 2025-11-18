import os
import sys

sys.path.append("../RETFound")

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import numpy as np
import models_vit
from util.pos_embed import interpolate_pos_embed


def get_args_parser():
    parser = argparse.ArgumentParser("RETFound testing", add_help=False)
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU")
    parser.add_argument("--input_size", default=224, type=int, help="image input size")
    parser.add_argument("--device", default="cuda", help="device to use for testing")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin_mem", action="store_true", help="Pin CPU memory in DataLoader"
    )

    # Model parameters
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.2,
        metavar="PCT",
        help="Drop path rate (default: 0.2)",
    )

    # Path parameters
    parser.add_argument(
        "--test_data_path", required=True, type=str, help="test dataset path"
    )
    parser.add_argument(
        "--model_path", required=True, type=str, help="path to trained model weights"
    )
    parser.add_argument(
        "--output_csv",
        default="test_results.csv",
        type=str,
        help="output CSV file name",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        help="Task name for compatibility with training script",
    )

    return parser


def main(args):
    # Create model with the same architecture as training
    from functools import partial

    # First, we need to determine the number of classes
    # We'll create a temporary dataset to get the class information
    temp_transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ]
    )

    temp_dataset = ImageFolder(args.test_data_path, transform=temp_transform)
    nb_classes = len(temp_dataset.classes)

    print(f"Number of classes: {nb_classes}")
    print(f"Class names: {temp_dataset.classes}")

    # Create model with the same parameters as training
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

    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location="cpu")

    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    else:
        model_state_dict = checkpoint

    # Remove "module." prefix if present (from distributed training)
    new_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Load state dict
    model.load_state_dict(new_state_dict)
    model = model.to(args.device)
    model.eval()

    # Test data transformations (same as validation transforms in training)
    test_transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create test dataset and loader
    test_dataset = ImageFolder(args.test_data_path, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
    )

    # Get class names
    class_names = test_dataset.classes

    # Collect predictions
    all_filenames = []
    all_predictions = []
    all_probabilities = []
    all_true_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            # Get model predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # Get filenames for this batch
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + len(images), len(test_dataset.samples))
            batch_filenames = [
                test_dataset.samples[i][0] for i in range(start_idx, end_idx)
            ]

            # Store results
            for i in range(len(images)):
                filename = os.path.basename(batch_filenames[i])
                all_filenames.append(filename)
                all_predictions.append(predicted[i].cpu().item())
                all_probabilities.append(probabilities[i].cpu().numpy())
                all_true_labels.append(labels[i].cpu().item())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "filename": all_filenames,
            "true_label": all_true_labels,
            "predicted_label": all_predictions,
        }
    )

    # Add probability scores for each class
    prob_array = np.array(all_probabilities)
    for i, class_name in enumerate(class_names):
        results_df[f"prob_{class_name}"] = prob_array[:, i]

    # Calculate accuracy
    accuracy = (results_df["true_label"] == results_df["predicted_label"]).mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save results to CSV
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = results_df["true_label"] == i
        if class_mask.sum() > 0:
            class_accuracy = (
                results_df.loc[class_mask, "true_label"]
                == results_df.loc[class_mask, "predicted_label"]
            ).mean()
            print(f"{class_name}: {class_accuracy * 100:.2f}%")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RETFound testing", parents=[get_args_parser()])
    args = parser.parse_args()
    results = main(args)
