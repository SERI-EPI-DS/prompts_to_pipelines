import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class TestDataset(torch.utils.data.Dataset):
    """Dataset for test images."""

    def __init__(self, test_dir, transform=None):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Collect all images with their class labels
        classes = sorted([d.name for d in self.test_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for class_name in classes:
            class_dir = self.test_dir / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".bmp",
                    ".tiff",
                ]:
                    self.samples.append(
                        {
                            "path": img_path,
                            "class_name": class_name,
                            "class_idx": self.class_to_idx[class_name],
                            "relative_path": str(img_path.relative_to(self.test_dir)),
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"])

        # Convert to RGB if grayscale
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "class_idx": sample["class_idx"],
            "class_name": sample["class_name"],
            "relative_path": sample["relative_path"],
        }


def create_model(num_classes):
    """Create ConvNext-L model with custom head."""
    model = models.convnext_large(weights=None)

    # Modify the classifier head to match training
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model


def get_test_transform(img_size=384):
    """Get test time transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transform


def test_time_augmentation(model, image, device, num_augmentations=5):
    """Apply test time augmentation for more robust predictions."""
    model.eval()

    # Different augmentations
    tta_transforms = [
        transforms.Compose([]),  # Original
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
        transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
        transforms.Compose([transforms.RandomRotation(degrees=10)]),
        transforms.Compose([transforms.RandomRotation(degrees=-10)]),
    ]

    predictions = []

    with torch.no_grad():
        for i in range(min(num_augmentations, len(tta_transforms))):
            # Apply augmentation
            if i > 0:  # Skip for original image
                # Convert tensor back to PIL for augmentation
                to_pil = transforms.ToPILImage()
                to_tensor = transforms.ToTensor()
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

                # Denormalize
                denorm = transforms.Normalize(
                    mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
                )
                img_denorm = denorm(image.squeeze(0))
                img_pil = to_pil(img_denorm)

                # Apply augmentation and normalize again
                img_aug = tta_transforms[i](img_pil)
                img_tensor = to_tensor(img_aug)
                normalize = transforms.Normalize(mean=mean, std=std)
                image_aug = normalize(img_tensor).unsqueeze(0).to(device)
            else:
                image_aug = image

            output = model(image_aug)
            predictions.append(torch.softmax(output, dim=1))

    # Average predictions
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    return avg_prediction


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Create model and load weights
    model = create_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create test dataset
    test_dir = Path(args.data_dir) / "test"
    test_transform = get_test_transform(img_size=args.img_size)
    test_dataset = TestDataset(test_dir, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for TTA
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Prepare results storage
    results = {
        "filename": [],
        "true_class": [],
        "predicted_class": [],
        "confidence": [],
    }

    # Add columns for each class probability
    for class_name in class_names:
        results[f"prob_{class_name}"] = []

    # Test the model
    print("\nTesting model...")
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            image = data["image"].to(device)
            true_class_idx = data["class_idx"].item()
            true_class_name = data["class_name"][0]
            relative_path = data["relative_path"][0]

            # Apply test time augmentation if enabled
            if args.use_tta:
                output = test_time_augmentation(
                    model, image, device, num_augmentations=5
                )
            else:
                output = model(image)
                output = torch.softmax(output, dim=1)

            # Get predictions
            probabilities = output.cpu().numpy()[0]
            predicted_idx = np.argmax(probabilities)
            predicted_class = class_names[predicted_idx]
            confidence = probabilities[predicted_idx]

            # Update accuracy
            if predicted_idx == true_class_idx:
                correct += 1
            total += 1

            # Store results
            results["filename"].append(relative_path)
            results["true_class"].append(true_class_name)
            results["predicted_class"].append(predicted_class)
            results["confidence"].append(float(confidence))

            # Store class probabilities
            for i, class_name in enumerate(class_names):
                results[f"prob_{class_name}"].append(float(probabilities[i]))

    # Calculate overall accuracy
    accuracy = 100.0 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})")

    # Create results dataframe
    df = pd.DataFrame(results)

    # Sort by filename
    df = df.sort_values("filename")

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "test_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Results saved to {csv_path}")

    # Calculate and save per-class metrics
    class_metrics = []
    for class_name in class_names:
        class_df = df[df["true_class"] == class_name]
        if len(class_df) > 0:
            class_correct = len(class_df[class_df["predicted_class"] == class_name])
            class_total = len(class_df)
            class_accuracy = 100.0 * class_correct / class_total

            class_metrics.append(
                {
                    "class": class_name,
                    "total_samples": class_total,
                    "correct_predictions": class_correct,
                    "accuracy": class_accuracy,
                }
            )

    class_metrics_df = pd.DataFrame(class_metrics)
    class_metrics_path = results_dir / "class_metrics.csv"
    class_metrics_df.to_csv(class_metrics_path, index=False, float_format="%.2f")
    print(f"Class metrics saved to {class_metrics_path}")

    # Save summary
    summary = {
        "test_accuracy": accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "num_classes": num_classes,
        "class_names": class_names,
        "model_path": str(args.model_path),
        "use_tta": args.use_tta,
    }

    summary_path = results_dir / "test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_path}")

    # Print class-wise results
    print("\nClass-wise Results:")
    print(class_metrics_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test ConvNext-L model on fundus images"
    )

    # Required arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing test folder",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to save test results"
    )

    # Optional arguments
    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="Input image size (should match training)",
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Use test time augmentation for more robust predictions",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    args = parser.parse_args()
    main(args)
