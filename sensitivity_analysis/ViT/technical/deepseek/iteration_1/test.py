import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class TestFundusDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.image_names = []

        # Support both structured (with class folders) and flat test directories
        if any(
            os.path.isdir(os.path.join(root_dir, item)) for item in os.listdir(root_dir)
        ):
            # Structured directory (with class folders)
            for class_name in os.listdir(root_dir):
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                        ):
                            self.image_paths.append(os.path.join(class_dir, img_name))
                            self.image_names.append(img_name)
        else:
            # Flat directory
            for img_name in os.listdir(root_dir):
                if img_name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                ):
                    self.image_paths.append(os.path.join(root_dir, img_name))
                    self.image_names.append(img_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = self.image_names[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_name


def load_model(model_path, num_classes, device):
    checkpoint = torch.load(model_path, map_location="cpu")

    if "classes" in checkpoint:
        classes = checkpoint["classes"]
        class_to_idx = checkpoint["class_to_idx"]
        num_classes = len(classes)
    else:
        classes = None
        class_to_idx = None

    model = timm.create_model(
        "swinv2_base_window8_256", pretrained=False, num_classes=num_classes
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, classes, class_to_idx


def get_test_transforms(img_size=256):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test Swin-V2-B for Ophthalmology Diagnosis"
    )
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
        "--batch_size", type=int, default=16, help="Batch size for testing"
    )
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Load model
    print("Loading model...")
    model, classes, class_to_idx = load_model(
        args.model_path, num_classes=None, device=device
    )

    if classes is None:
        print("Warning: Class names not found in model checkpoint.")
        classes = [f"class_{i}" for i in range(getattr(model, "num_classes", 2))]

    print(f"Loaded model for {len(classes)} classes: {classes}")

    # Test transforms
    test_transform = get_test_transforms(args.img_size)

    # Test dataset and dataloader
    test_dataset = TestFundusDataset(args.data_root, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Testing
    print("Running inference...")
    all_predictions = []
    all_probabilities = []
    all_filenames = []

    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Testing"):
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_filenames.extend(filenames)

    # Create results DataFrame
    results = []
    for filename, pred_idx, probs in zip(
        all_filenames, all_predictions, all_probabilities
    ):
        result_row = {"filename": filename}

        # Add probability scores for each class
        for i, class_name in enumerate(classes):
            result_row[f"score_{class_name}"] = probs[i]

        # Add final prediction
        result_row["prediction"] = classes[pred_idx]
        result_row["prediction_index"] = pred_idx

        results.append(result_row)

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to: {args.output_csv}")

    # Print summary statistics
    print(f"\nTest Summary:")
    print(f"Total images processed: {len(results_df)}")
    print(f"Class distribution:")
    pred_counts = results_df["prediction"].value_counts()
    for class_name in classes:
        count = pred_counts.get(class_name, 0)
        percentage = 100 * count / len(results_df)
        print(f"  {class_name}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
