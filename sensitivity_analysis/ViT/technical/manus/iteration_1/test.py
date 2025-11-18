#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def main():
    parser = argparse.ArgumentParser(
        description="Test a Swin-V2-B classifier for ophthalmology image classification."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the root data directory."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save the results CSV.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Define data transforms
    test_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )

    # Load the checkpoint to get model information
    checkpoint = torch.load(args.model_path, map_location="cpu")

    # Get class information from checkpoint if available, otherwise from train dataset
    if "class_names" in checkpoint and "num_classes" in checkpoint:
        class_names = checkpoint["class_names"]
        num_classes = checkpoint["num_classes"]
        print(f"Loaded class information from checkpoint:")
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")
    else:
        # Fallback: Load train dataset to get class names
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"))
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes
        print(f"Loaded class information from train dataset:")
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")

    # Create test dataset and dataloader
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Test dataset size: {len(test_dataset)} images")

    # Create the model
    model = timm.create_model(
        "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        pretrained=False,  # Set to False as we are loading our own weights
        num_classes=num_classes,
    )

    # Load the trained weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")
        print(
            f"Best validation accuracy: {checkpoint.get('best_val_acc', 'unknown'):.4f}"
        )
    else:
        # Fallback for older checkpoint format
        model.load_state_dict(checkpoint)
        print("Loaded model weights (legacy format)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    print("Starting testing...")

    # Testing loop
    results = []
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            scores = torch.softmax(outputs, dim=1)

            _, predicted_indices = torch.max(scores, 1)

            # Calculate accuracy
            correct_predictions += (predicted_indices == labels).sum().item()
            total_predictions += labels.size(0)

            for j in range(images.size(0)):
                # Get the actual image path and filename
                image_path, true_label = test_loader.dataset.samples[
                    batch_idx * args.batch_size + j
                ]
                image_name = os.path.basename(image_path)

                result = {"filename": image_name}

                # Add prediction scores for each class
                for k, class_name in enumerate(class_names):
                    result[f"{class_name}_score"] = scores[j, k].item()

                # Add predicted class and true class
                result["predicted_class"] = class_names[predicted_indices[j]]
                result["true_class"] = class_names[true_label]
                result["correct_prediction"] = (
                    predicted_indices[j] == labels[j]
                ).item()

                results.append(result)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * args.batch_size} images...")

    # Calculate overall accuracy
    test_accuracy = correct_predictions / total_predictions
    print(
        f"Test accuracy: {test_accuracy:.4f} ({correct_predictions}/{total_predictions})"
    )

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    output_path = os.path.join(args.output_dir, "test_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total test images: {len(results)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Per-class accuracy if available
    if len(results) > 0:
        class_accuracy = results_df.groupby("true_class")["correct_prediction"].mean()
        print("\nPer-class accuracy:")
        for class_name, accuracy in class_accuracy.items():
            class_count = (results_df["true_class"] == class_name).sum()
            print(f"  {class_name}: {accuracy:.4f} ({class_count} images)")


if __name__ == "__main__":
    main()
