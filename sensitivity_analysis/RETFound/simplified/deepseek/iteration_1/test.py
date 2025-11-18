import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import models_vit

# from util.datasets import build_dataset
from torchvision import datasets, transforms
from util.pos_embed import interpolate_pos_embed
import argparse
import os
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
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
        split_dir = "test"  # or 'test' depending on your dataset structure

    # Construct the full path using the directory name, not the boolean
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


def validate_model_choice(available_models, requested_model):
    """Validate and return the correct model name to use."""
    # List of preferred RETFound models in priority order
    retfound_models = ["RETFound_mae", "RETFound_dinov2", "VisionTransformer"]

    # If the requested model is available, use it
    if requested_model in available_models:
        return requested_model

    # Otherwise, try to find a suitable RETFound model
    for model_name in retfound_models:
        if model_name in available_models:
            print(f"Using available RETFound model: {model_name}")
            return model_name

    # If no RETFound models are available, use the first available model
    if available_models:
        print(f"No RETFound models found. Using available model: {available_models[0]}")
        return available_models[0]

    raise ValueError("No valid models found in models_vit module")


def get_args_parser():
    parser = argparse.ArgumentParser("RETFound evaluation", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="RETFound_mae",
        type=str,
        metavar="MODEL",
        help="Name of model to evaluate",
    )
    parser.add_argument("--input_size", default=224, type=int, help="image input size")
    parser.add_argument(
        "--nb_classes", default=5, type=int, help="number of the classification types"
    )

    # Evaluation parameters
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument(
        "--data_path",
        default="./dataset/",
        type=str,
        help="dataset path for evaluation",
    )
    parser.add_argument(
        "--resume",
        default="./output_finetune/checkpoint-final.pth",
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default="./evaluation_results/",
        help="path where to save evaluation results",
    )

    return parser


def main(args):
    # First, check available models and validate our choice
    available_models = get_available_models()
    model_name = validate_model_choice(available_models, args.model)
    args.model = model_name

    # Build model
    print(f"Creating model: {args.model}")

    # Handle different model types with appropriate parameters
    if "RETFound" in args.model or "VisionTransformer" in args.model:
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=0.0,  # No drop path during evaluation
            global_pool=True,
        )
    else:
        # Fallback for other model types
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=0.0,
        )

    # Load checkpoint
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location="cpu")

        if "model" in checkpoint:
            # Remove module prefix if present (from DDP training)
            state_dict = checkpoint["model"]
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v

            # Load state dict
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"Load result: {msg}")
        else:
            # Load directly if no 'model' key
            model.load_state_dict(checkpoint)

        print(
            f"Loaded checkpoint '{args.resume}' (epoch {checkpoint.get('epoch', 'unknown')})"
        )
    else:
        raise FileNotFoundError(f"No checkpoint found at '{args.resume}'")

    model.cuda()
    model.eval()

    # Build dataset - this should now work with the corrected build_dataset function
    print("Building evaluation dataset...")
    dataset_eval = build_dataset(is_train=False, args=args)
    print(f"Evaluation samples: {len(dataset_eval)}")
    print(f"Dataset classes: {dataset_eval.classes}")

    data_loader_eval = DataLoader(
        dataset_eval,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    # Evaluation
    print("Starting evaluation...")
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader_eval):
            images = batch[0].cuda()
            labels = batch[1].cuda()

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    # Convert to numpy arrays and calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calculate comprehensive metrics
    accuracy = (all_predictions == all_labels).mean()
    precision = precision_score(
        all_labels, all_predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        all_labels, all_predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

    # Classification report and confusion matrix
    class_report = classification_report(all_labels, all_predictions, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Save results
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix.tolist(),
            "per_class_metrics": {
                k: v
                for k, v in class_report.items()
                if k not in ["accuracy", "macro avg", "weighted avg"]
            },
            "macro_avg": class_report["macro avg"],
            "weighted_avg": class_report["weighted avg"],
        }

        with open(os.path.join(args.output_dir, "evaluation_metrics.json"), "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {args.output_dir}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
