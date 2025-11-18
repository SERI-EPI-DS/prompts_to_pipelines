import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class FundusTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []

        # Collect all images with their paths and labels
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_names.append(class_name)
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                    ):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


def get_test_transforms(input_size=384):
    """
    Test time augmentation transforms
    """
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_model(num_classes):
    """
    Create Swin-V2-B model with custom head
    """
    model = models.swin_v2_b(weights=None)

    # Get the number of features in the last layer
    num_features = model.head.in_features

    # Replace the head with the same architecture used in training
    model.head = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )

    return model


def test_time_augmentation(model, image, device, num_augmentations=5):
    """
    Apply test time augmentation for more robust predictions
    """
    model.eval()

    predictions = []

    # Original image
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        predictions.append(F.softmax(output, dim=1).cpu())

    # Augmented versions
    for _ in range(num_augmentations - 1):
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            aug_image = torch.flip(image, dims=[2])
        else:
            aug_image = image

        # Random vertical flip
        if torch.rand(1) > 0.5:
            aug_image = torch.flip(aug_image, dims=[1])

        with torch.no_grad():
            output = model(aug_image.unsqueeze(0).to(device))
            predictions.append(F.softmax(output, dim=1).cpu())

    # Average predictions
    avg_prediction = torch.stack(predictions).mean(dim=0)
    return avg_prediction


def main():
    parser = argparse.ArgumentParser(
        description="Test Swin-V2-B for Fundus Image Classification"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to data root folder"
    )
    parser.add_argument(
        "--results_folder", type=str, required=True, help="Path to results folder"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--input_size", type=int, default=384, help="Input image size")
    parser.add_argument(
        "--use_tta", action="store_true", help="Use test time augmentation"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model weights (default: best_model.pth in results folder)",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class names
    class_names_path = os.path.join(args.results_folder, "class_names.json")
    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
    else:
        print(
            "Warning: class_names.json not found. Using class names from test directory."
        )
        class_names = sorted(os.listdir(os.path.join(args.data_root, "test")))

    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

    # Create test dataset
    test_dir = os.path.join(args.data_root, "test")
    test_transform = get_test_transforms(input_size=args.input_size)
    test_dataset = FundusTestDataset(test_dir, transform=test_transform)

    # Create model
    model = create_model(num_classes)

    # Load model weights
    if args.model_path is None:
        model_path = os.path.join(args.results_folder, "best_model.pth")
    else:
        model_path = args.model_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {model_path}")

    # Results storage
    results = []
    all_predictions = []
    all_labels = []

    # Test the model
    if args.use_tta:
        print("Using Test Time Augmentation...")
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset)), desc="Testing"):
                image, label, img_path = test_dataset[i]

                # Apply TTA
                pred_probs = test_time_augmentation(model, image, device)
                pred_probs = pred_probs.squeeze(0).numpy()

                # Get prediction
                pred_class = np.argmax(pred_probs)

                # Store results
                result = {
                    "file_name": os.path.relpath(img_path, args.data_root),
                    "true_label": class_names[label],
                    "predicted_label": class_names[pred_class],
                }

                # Add probability scores for each class
                for i, class_name in enumerate(class_names):
                    result[f"prob_{class_name}"] = float(pred_probs[i])

                results.append(result)
                all_predictions.append(pred_class)
                all_labels.append(label)
    else:
        # Standard testing without TTA
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        with torch.no_grad():
            for images, labels, img_paths in tqdm(test_loader, desc="Testing"):
                images = images.to(device)

                outputs = model(images)
                probs = F.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)

                for i in range(len(images)):
                    result = {
                        "file_name": os.path.relpath(img_paths[i], args.data_root),
                        "true_label": class_names[labels[i]],
                        "predicted_label": class_names[predicted[i].item()],
                    }

                    # Add probability scores for each class
                    for j, class_name in enumerate(class_names):
                        result[f"prob_{class_name}"] = float(probs[i, j].cpu().numpy())

                    results.append(result)
                    all_predictions.append(predicted[i].item())
                    all_labels.append(labels[i].item())

    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_predictions == all_labels) * 100

    # Calculate per-class accuracy
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_labels == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(all_predictions[class_mask] == i) * 100
            class_accuracies[class_name] = class_acc

    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.results_folder, "test_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    # Save summary statistics
    summary = {
        "overall_accuracy": accuracy,
        "class_accuracies": class_accuracies,
        "num_test_samples": len(test_dataset),
        "test_configuration": vars(args),
    }

    summary_path = os.path.join(args.results_folder, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nTest Results:")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"\nPer-class Accuracy:")
    for class_name, acc in class_accuracies.items():
        print(f"{class_name}: {acc:.2f}%")

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_folder, "confusion_matrix.png"))
    plt.close()

    # Save classification report
    report = classification_report(
        all_labels, all_predictions, target_names=class_names, output_dict=True
    )
    report_path = os.path.join(args.results_folder, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nAll results saved to: {args.results_folder}")


if __name__ == "__main__":
    main()
