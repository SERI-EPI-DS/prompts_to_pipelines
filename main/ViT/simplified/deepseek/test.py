import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from sklearn.metrics import classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Test Swin-V2-B Classifier")
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Path to testing dataset"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument("--input_size", type=int, default=256, help="Input image size")
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformation
    test_transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )

    # Dataset and Loader
    test_dataset = datasets.ImageFolder(args.test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Load model
    model = timm.create_model(
        "swinv2_base_window8_256",
        pretrained=False,
        num_classes=len(test_dataset.classes),
    )
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    # Test the model
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Generate reports
    class_names = test_dataset.classes
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Calculate accuracy
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\nTest Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
