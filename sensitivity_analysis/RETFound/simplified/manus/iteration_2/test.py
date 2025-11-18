#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import timm


def get_args_parser():
    parser = argparse.ArgumentParser(
        "RETFound Testing for Image Classification", add_help=False
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        default="RETFound_cfp",
        type=str,
        choices=["RETFound_cfp", "RETFound_oct"],
        help="Name of model to test (RETFound_cfp for color fundus, RETFound_oct for OCT)",
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--resume", default="", required=True, help="resume from checkpoint"
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
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    return parser


def create_model(model_name, num_classes, drop_path_rate=0.1):
    """Create Vision Transformer model compatible with RETFound weights"""

    # Create a ViT-Large model similar to RETFound architecture
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        global_pool="token",
    )

    return model


def main(args):
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Data preprocessing
    transform_test = transforms.Compose(
        [
            transforms.Resize(args.input_size, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load test dataset
    dataset_test = datasets.ImageFolder(
        os.path.join(args.data_path, "test"), transform=transform_test
    )
    print(f"Test dataset size: {len(dataset_test)}")
    print(f"Classes: {dataset_test.classes}")

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=32,  # Use larger batch size for faster inference
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Create model
    model = create_model(
        args.model_name, num_classes=args.nb_classes, drop_path_rate=0.1
    )

    # Load checkpoint
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("Loading from training checkpoint format")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("Loading from state_dict format")
        else:
            state_dict = checkpoint
            print("Loading from direct state dict format")

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys}")

        print("Checkpoint loaded successfully")
    else:
        raise ValueError("No checkpoint provided for testing")

    model.to(device)
    model.eval()

    print("Starting evaluation...")

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader_test):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probabilities.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * 32}/{len(dataset_test)} samples")

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    conf_matrix = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-score: {f1:.4f}")
    print(f"Micro Precision: {precision_micro:.4f}")
    print(f"Micro Recall: {recall_micro:.4f}")
    print(f"Micro F1-score: {f1_micro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall: {recall_weighted:.4f}")
    print(f"Weighted F1-score: {f1_weighted:.4f}")

    print("\nPer-class Results:")
    for i, class_name in enumerate(dataset_test.classes):
        print(f"Class {class_name}:")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall: {recall_per_class[i]:.4f}")
        print(f"  F1-score: {f1_per_class[i]:.4f}")
        print(f"  Support: {support_per_class[i]}")

    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=dataset_test.classes, zero_division=0
        )
    )

    # Prepare results dictionary
    results = {
        "overall_metrics": {
            "accuracy": float(accuracy),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "micro_precision": float(precision_micro),
            "micro_recall": float(recall_micro),
            "micro_f1": float(f1_micro),
            "weighted_precision": float(precision_weighted),
            "weighted_recall": float(recall_weighted),
            "weighted_f1": float(f1_weighted),
        },
        "per_class_metrics": {
            dataset_test.classes[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1_score": float(f1_per_class[i]),
                "support": int(support_per_class[i]),
            }
            for i in range(len(dataset_test.classes))
        },
        "confusion_matrix": conf_matrix.tolist(),
        "class_names": dataset_test.classes,
    }

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save detailed results
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        # Save predictions
        predictions_data = {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_probs": y_probs.tolist(),
            "class_names": dataset_test.classes,
        }

        with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
            json.dump(predictions_data, f, indent=4)

        print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
