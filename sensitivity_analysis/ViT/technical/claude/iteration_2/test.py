import os
import argparse
import csv
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class FundusDataset(torch.utils.data.Dataset):
    """Custom dataset for fundus images."""

    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.samples = self.dataset.samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = self.dataset.loader(img_path)

        if self.transform:
            img = self.transform(img)

        # Get relative path for the image
        rel_path = os.path.relpath(img_path, self.dataset.root)

        return img, label, rel_path


def get_transforms(input_size=384):
    """Get test-time transforms."""
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


def create_model(num_classes):
    """Create Swin-V2-B model with custom head."""
    model = models.swin_v2_b(weights=None)

    # Recreate the same classifier head structure as in training
    num_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


def test_time_augmentation(model, image, device, n_augmentations=5):
    """Apply test-time augmentation to improve predictions."""
    tta_transforms = [
        transforms.Compose([]),  # Original
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
        transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
        transforms.Compose([transforms.RandomRotation(degrees=10)]),
        transforms.Compose([transforms.RandomRotation(degrees=-10)]),
    ]

    predictions = []

    for i in range(min(n_augmentations, len(tta_transforms))):
        augmented = tta_transforms[i](image)
        with torch.no_grad():
            with autocast():
                output = model(augmented.unsqueeze(0).to(device))
                prob = torch.softmax(output, dim=1)
        predictions.append(prob.cpu().numpy())

    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction[0]


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading model from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Get model info from checkpoint
    classes = checkpoint["classes"]
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(classes)

    print(f"Number of classes: {num_classes}")
    print(f"Classes: {classes}")

    # Create model
    model = create_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Get input size from training args if available
    input_size = checkpoint["args"].input_size if "args" in checkpoint else 384

    # Create test dataset
    test_transform = get_transforms(input_size)
    test_dataset = FundusDataset(
        os.path.join(args.data_dir, "test"), transform=test_transform
    )

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for detailed results
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Prepare CSV file
    csv_path = os.path.join(args.results_dir, "test_results.csv")

    # Create header for CSV
    header = ["image_path", "true_class", "predicted_class"]
    for cls in classes:
        header.append(f"prob_{cls}")
    header.extend(["confidence", "correct"])

    results = []
    correct = 0
    total = 0

    # Test loop
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            if args.use_tta:
                # Test-time augmentation
                probs = test_time_augmentation(model, images[0].cpu(), device)
                probs = torch.from_numpy(probs).unsqueeze(0)
            else:
                # Standard inference
                with autocast():
                    outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu()

            # Get predictions
            confidence, predicted = torch.max(probs, 1)

            # Prepare row for CSV
            row = {
                "image_path": paths[0],
                "true_class": classes[labels[0].item()],
                "predicted_class": classes[predicted[0].item()],
                "confidence": confidence[0].item(),
                "correct": int(predicted[0].item() == labels[0].item()),
            }

            # Add probability for each class
            for i, cls in enumerate(classes):
                row[f"prob_{cls}"] = probs[0][i].item()

            results.append(row)

            # Update accuracy
            total += 1
            if predicted[0].item() == labels[0].item():
                correct += 1

    # Calculate overall accuracy
    accuracy = 100.0 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})")

    # Write results to CSV
    print(f"\nSaving results to: {csv_path}")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    # Calculate per-class metrics
    class_correct = {cls: 0 for cls in classes}
    class_total = {cls: 0 for cls in classes}

    for result in results:
        true_class = result["true_class"]
        pred_class = result["predicted_class"]
        class_total[true_class] += 1
        if true_class == pred_class:
            class_correct[true_class] += 1

    # Print per-class accuracy
    print("\nPer-class accuracy:")
    print("-" * 40)
    for cls in classes:
        if class_total[cls] > 0:
            cls_acc = 100.0 * class_correct[cls] / class_total[cls]
            print(
                f"{cls:15s}: {cls_acc:6.2f}% ({class_correct[cls]}/{class_total[cls]})"
            )
        else:
            print(f"{cls:15s}: No samples")

    # Save summary
    summary = {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class_accuracy": {
            cls: (
                (100.0 * class_correct[cls] / class_total[cls])
                if class_total[cls] > 0
                else 0
            )
            for cls in classes
        },
        "per_class_samples": class_total,
        "per_class_correct": class_correct,
        "checkpoint_used": args.checkpoint_path,
        "test_time_augmentation": args.use_tta,
    }

    summary_path = os.path.join(args.results_dir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nTest summary saved to: {summary_path}")
    print("\nTesting completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Swin-V2-B classifier on fundus images"
    )

    # Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing test folder",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (best_model.pth)",
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to save test results"
    )

    # Test parameters
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Use test-time augmentation for better accuracy",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )

    args = parser.parse_args()
    main(args)
