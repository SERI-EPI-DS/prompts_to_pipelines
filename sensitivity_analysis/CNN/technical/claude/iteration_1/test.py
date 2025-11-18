import os
import argparse
import json
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import convnext_large
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def get_transforms(input_size=384):
    """Get image transformations for testing"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.1)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def create_model(num_classes, checkpoint_path):
    """Create and load trained ConvNext-L model"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint["args"]

    # Create model with same architecture
    model = convnext_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Sequential(
        torch.nn.Dropout(p=args.dropout), torch.nn.Linear(in_features, num_classes)
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint


def test_model(model, dataloader, device, num_classes):
    """Test model and collect predictions"""
    model.eval()

    all_paths = []
    all_predictions = []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        for inputs, labels in pbar:
            inputs = inputs.to(device)

            # Get predictions
            outputs = model(inputs)
            scores = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # Store results
            batch_paths = [
                dataloader.dataset.samples[i][0]
                for i in range(len(all_paths), len(all_paths) + inputs.size(0))
            ]
            all_paths.extend(batch_paths)
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    return all_paths, all_predictions, all_scores, all_labels


def calculate_metrics(predictions, labels, num_classes):
    """Calculate accuracy metrics"""
    correct = sum(p == l for p, l in zip(predictions, labels))
    total = len(labels)
    accuracy = 100.0 * correct / total

    # Per-class accuracy
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for p, l in zip(predictions, labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    class_accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0.0)

    return accuracy, class_accuracies


def main(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class mapping
    with open(os.path.join(args.model_dir, "class_mapping.json"), "r") as f:
        mapping = json.load(f)
        idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
        class_to_idx = mapping["class_to_idx"]

    num_classes = len(idx_to_class)
    print(f"Number of classes: {num_classes}")

    # Data preparation
    transform = get_transforms(args.input_size)
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), transform=transform
    )

    # Verify class consistency
    if test_dataset.class_to_idx != class_to_idx:
        print("Warning: Test dataset classes don't match training classes!")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load model
    checkpoint_path = os.path.join(args.model_dir, "best_model.pth")
    model, checkpoint = create_model(num_classes, checkpoint_path)
    model = model.to(device)
    print(
        f'Loaded model from epoch {checkpoint["epoch"]} with validation accuracy: {checkpoint["best_val_acc"]:.2f}%'
    )

    # Test model
    paths, predictions, scores, labels = test_model(
        model, test_loader, device, num_classes
    )

    # Calculate metrics
    accuracy, class_accuracies = calculate_metrics(predictions, labels, num_classes)

    print(f"\nTest Results:")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"\nPer-class Accuracies:")
    for i, acc in enumerate(class_accuracies):
        print(f"{idx_to_class[i]}: {acc:.2f}%")

    # Save results to CSV
    csv_path = os.path.join(args.output_dir, "test_results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        # Create header
        fieldnames = ["filename", "true_class", "predicted_class"]
        for i in range(num_classes):
            fieldnames.append(f"score_{idx_to_class[i]}")

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write rows
        for path, pred, score_list, label in zip(paths, predictions, scores, labels):
            row = {
                "filename": os.path.basename(path),
                "true_class": idx_to_class[label],
                "predicted_class": idx_to_class[pred],
            }

            # Add scores for each class
            for i in range(num_classes):
                row[f"score_{idx_to_class[i]}"] = f"{score_list[i]:.6f}"

            writer.writerow(row)

    print(f"\nResults saved to: {csv_path}")

    # Save summary
    summary = {
        "overall_accuracy": accuracy,
        "per_class_accuracy": {
            idx_to_class[i]: acc for i, acc in enumerate(class_accuracies)
        },
        "total_samples": len(labels),
        "model_epoch": checkpoint["epoch"],
        "validation_accuracy": checkpoint["best_val_acc"],
    }

    summary_path = os.path.join(args.output_dir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fine-tuned ConvNext-L model")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to directory containing trained model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
