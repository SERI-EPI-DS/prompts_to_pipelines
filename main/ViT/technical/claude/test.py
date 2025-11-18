import os
import argparse
import csv
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast
import timm
from tqdm import tqdm
import numpy as np
from pathlib import Path


def get_test_transforms(input_size=384):
    """Get test-time transforms"""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_model_from_checkpoint(checkpoint, device):
    """Create model based on checkpoint configuration"""

    # Get model configuration from checkpoint
    num_classes = len(checkpoint["class_names"])
    input_size = checkpoint.get("input_size", 192)  # Default to 192 if not specified

    # Try to get model name from args if available
    model_name = None
    if "args" in checkpoint and isinstance(checkpoint["args"], dict):
        # Try to infer model from training args
        train_input_size = checkpoint["args"].get("input_size", input_size)
        input_size = train_input_size

    # Model configurations based on input size
    if input_size == 192:
        model_configs = [
            "swinv2_base_window12_192",
            "swinv2_base_window12_192_22k",
        ]
    elif input_size == 224:
        model_configs = [
            "swin_base_patch4_window7_224",
            "swin_base_patch4_window7_224_in22k",
        ]
    elif input_size == 256:
        model_configs = [
            "swinv2_base_window16_256",
            "swinv2_base_window8_256",
        ]
    else:  # 384 or other sizes
        model_configs = [
            "swin_large_patch4_window12_384",
            "swin_large_patch4_window12_384_in22k",
            "swin_base_patch4_window12_384",
            "swin_base_patch4_window12_384_in22k",
        ]

    # Add fallback options
    model_configs.extend(
        [
            "swin_base_patch4_window7_224",
            "swinv2_base_window12_192_22k",
            "resnet50",
        ]
    )

    model = None
    model_name_used = None

    for model_name in model_configs:
        try:
            model = timm.create_model(
                model_name, pretrained=False, num_classes=num_classes
            )

            # Try to load the state dict
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
                model_name_used = model_name
                print(
                    f"Successfully loaded model: {model_name} with input size {input_size}"
                )
                break
            except:
                # If loading fails, try next model
                model = None
                continue

        except:
            continue

    if model is None:
        raise RuntimeError(
            f"Could not find compatible model for checkpoint with input size {input_size}"
        )

    return model.to(device), input_size, model_name_used


def test_time_augmentation(model, image, device, num_augmentations=5):
    """Apply test-time augmentation for more robust predictions"""
    model.eval()

    # Base prediction
    with torch.no_grad():
        with autocast():
            base_output = model(image.unsqueeze(0).to(device))
            predictions = F.softmax(base_output, dim=1)

    # TTA transforms
    tta_transforms = [
        lambda x: x,  # Original
        lambda x: torch.flip(x, dims=[2]),  # Horizontal flip
        lambda x: torch.flip(x, dims=[3]),  # Vertical flip
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # 90 degree rotation
        lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # 270 degree rotation
    ]

    # Apply TTA
    for i in range(1, min(num_augmentations, len(tta_transforms))):
        augmented = tta_transforms[i](image.unsqueeze(0))
        with torch.no_grad():
            with autocast():
                output = model(augmented.to(device))
                predictions += F.softmax(output, dim=1)

    # Average predictions
    predictions = predictions / min(num_augmentations, len(tta_transforms))

    return predictions.squeeze(0)


def test_model(model, test_loader, device, class_names, use_tta=True):
    """Test the model and collect predictions"""
    model.eval()

    all_predictions = []
    all_scores = []
    all_paths = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                batch_size = images.size(0)

                for i in range(batch_size):
                    image = images[i]

                    if use_tta:
                        # Use test-time augmentation
                        scores = test_time_augmentation(model, image, device)
                    else:
                        # Standard prediction
                        with autocast():
                            output = model(image.unsqueeze(0).to(device))
                            scores = F.softmax(output, dim=1).squeeze(0)

                    predicted_class = scores.argmax().item()

                    # Get file path
                    idx = len(all_predictions)
                    if hasattr(test_loader.dataset, "samples"):
                        file_path, _ = test_loader.dataset.samples[idx]
                        relative_path = os.path.relpath(
                            file_path, test_loader.dataset.root
                        )
                    else:
                        # Fallback for different dataset structures
                        relative_path = f"image_{idx}.png"

                    all_predictions.append(predicted_class)
                    all_scores.append(scores.cpu().numpy())
                    all_paths.append(relative_path)

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

    return all_predictions, all_scores, all_paths


def save_results_csv(predictions, scores, paths, class_names, output_path):
    """Save test results to CSV file"""
    with open(output_path, "w", newline="") as csvfile:
        fieldnames = ["filename", "predicted_class"] + [
            f"score_{cls}" for cls in class_names
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for path, pred, score in zip(paths, predictions, scores):
            row = {"filename": path, "predicted_class": class_names[pred]}

            for i, cls in enumerate(class_names):
                row[f"score_{cls}"] = f"{score[i]:.6f}"

            writer.writerow(row)


def calculate_metrics(predictions, labels, num_classes):
    """Calculate accuracy metrics"""
    correct = sum(p == l for p, l in zip(predictions, labels))
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0

    # Per-class accuracy
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for pred, label in zip(predictions, labels):
        if label < num_classes:  # Ensure label is valid
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

    class_accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies.append(class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)

    return accuracy, class_accuracies


def main():
    parser = argparse.ArgumentParser(
        description="Test Swin-V2-B model on fundus images"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to data root folder"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to results folder"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: best_model.pth in results_dir)",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--input_size",
        type=int,
        default=None,
        help="Input image size (if None, will use size from checkpoint)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--use_tta", action="store_true", help="Use test-time augmentation"
    )
    parser.add_argument(
        "--output_csv", type=str, default="test_results.csv", help="Output CSV filename"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    if args.checkpoint is None:
        checkpoint_path = os.path.join(args.results_dir, "best_model.pth")
    else:
        checkpoint_path = args.checkpoint

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get class names from checkpoint
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

    # Create model and get input size from checkpoint
    model, actual_input_size, model_name = create_model_from_checkpoint(
        checkpoint, device
    )

    # Override input size if specified
    if args.input_size is not None:
        print(
            f"Warning: Overriding checkpoint input size {actual_input_size} with {args.input_size}"
        )
        actual_input_size = args.input_size

    print(f"Using input size: {actual_input_size}x{actual_input_size}")

    # Create test dataset with correct input size
    test_dataset = ImageFolder(
        os.path.join(args.data_root, "test"),
        transform=get_test_transforms(actual_input_size),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 2),  # Limit workers
        pin_memory=True,
    )

    print(f"\nTotal test images: {len(test_dataset)}")
    print(f"Using test-time augmentation: {args.use_tta}")
    print(f"Model architecture: {model_name}")

    # Test the model
    predictions, scores, paths = test_model(
        model, test_loader, device, class_names, use_tta=args.use_tta
    )

    # Get true labels
    true_labels = []
    for i in range(len(test_dataset)):
        _, label = test_dataset[i]
        true_labels.append(label)

    # Calculate metrics
    accuracy, class_accuracies = calculate_metrics(
        predictions, true_labels, num_classes
    )

    print(f"\nTest Results:")
    print(
        f"Overall Accuracy: {accuracy:.4f} ({sum(p == l for p, l in zip(predictions, true_labels))}/{len(predictions)})"
    )
    print("\nPer-class Accuracy:")
    for i, cls in enumerate(class_names):
        print(f"{cls}: {class_accuracies[i]:.4f}")

    # Save results to CSV
    output_path = os.path.join(args.results_dir, args.output_csv)
    save_results_csv(predictions, scores, paths, class_names, output_path)
    print(f"\nResults saved to: {output_path}")

    # Save summary
    summary = {
        "total_images": len(test_dataset),
        "overall_accuracy": float(accuracy),
        "per_class_accuracy": {
            cls: float(acc) for cls, acc in zip(class_names, class_accuracies)
        },
        "checkpoint_used": checkpoint_path,
        "test_time_augmentation": args.use_tta,
        "input_size": actual_input_size,
        "model_architecture": model_name,
    }

    with open(os.path.join(args.results_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(
        f'Test summary saved to: {os.path.join(args.results_dir, "test_summary.json")}'
    )


if __name__ == "__main__":
    main()
