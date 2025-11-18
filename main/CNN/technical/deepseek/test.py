import os
import argparse
import csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="ConvNeXt-L Testing")
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root directory of test dataset"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model weights"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Path to output CSV file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for testing (default: 32)",
    )
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test transformations
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Create test dataset with file paths
    class ImageFolderWithPaths(datasets.ImageFolder):
        def __getitem__(self, index):
            original_tuple = super().__getitem__(index)
            path = self.samples[index][0]
            return original_tuple + (path,)

    test_dataset = ImageFolderWithPaths(args.data_root, test_transform)

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    model = models.convnext_large(weights=None)

    # Reconstruct classifier - handle different checkpoint formats
    num_ftrs = model.classifier[2].in_features

    # Determine number of classes from checkpoint
    if "class_to_idx" in checkpoint:
        num_classes = len(checkpoint["class_to_idx"])
    elif "classes" in checkpoint:
        num_classes = len(checkpoint["classes"])
    else:
        raise KeyError("Checkpoint must contain either 'class_to_idx' or 'classes' key")

    model.classifier[2] = nn.Linear(num_ftrs, num_classes)

    # Load state dict - handle both regular and DataParallel models
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Create class mappings
    if "class_to_idx" in checkpoint:
        class_to_idx = checkpoint["class_to_idx"]
        classes = list(class_to_idx.keys())
    elif "classes" in checkpoint:
        classes = checkpoint["classes"]
        # Create class_to_idx mapping from classes list
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    else:
        # Fallback to dataset classes if not in checkpoint
        class_to_idx = test_dataset.class_to_idx
        classes = list(class_to_idx.keys())

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"Loaded model with {len(classes)} classes")

    # Prepare CSV output
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", newline="") as csvfile:
        fieldnames = (
            ["file_path", "true_class", "predicted_class"]
            + [f"prob_{cls}" for cls in classes]
            + ["correct"]
        )
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Inference loop
        with torch.no_grad():
            for inputs, labels, paths in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                # Write results for each sample in batch
                for i in range(inputs.size(0)):
                    # Get relative path from data_root
                    rel_path = os.path.relpath(paths[i], args.data_root)

                    row = {
                        "file_path": rel_path,
                        "true_class": idx_to_class[labels[i].item()],
                        "predicted_class": idx_to_class[preds[i].item()],
                        "correct": int(preds[i] == labels[i]),
                    }

                    # Add probabilities for each class
                    for j, cls in enumerate(classes):
                        row[f"prob_{cls}"] = probs[i, j].item()

                    writer.writerow(row)

    print(f"Test results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
