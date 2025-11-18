import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import functional  # Corrected import
import pandas as pd
from tqdm import tqdm
import numpy as np


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_dim = max(w, h)
        hp = (max_dim - w) // 2
        vp = (max_dim - h) // 2
        padding = (
            hp,
            vp,
            hp + (0 if (max_dim - w) % 2 == 0 else 1),
            vp + (0 if (max_dim - h) % 2 == 0 else 1),
        )
        return functional.pad(image, padding, 0, "constant")


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transform
    test_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create custom dataset to get image paths
    class ImageFolderWithPaths(datasets.ImageFolder):
        def __getitem__(self, index):
            original_tuple = super().__getitem__(index)
            path = self.samples[index][0]
            return (*original_tuple, path)

    # Load dataset
    test_dataset = ImageFolderWithPaths(
        root=os.path.join(args.data_root, "test"), transform=test_transform
    )

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize model
    model = models.convnext_large()
    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features, len(test_dataset.classes)
    )
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    # Get class names
    class_names = test_dataset.classes

    # Run inference
    results = []
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            for i in range(len(paths)):
                rel_path = os.path.relpath(paths[i], start=args.data_root)
                result = {
                    "image_path": rel_path,
                    "predicted_class": class_names[preds[i]],
                    "true_class": class_names[labels[i].item()],
                }
                # Add probabilities for each class
                for cls_idx, cls_name in enumerate(class_names):
                    result[f"prob_{cls_name}"] = probs[i, cls_idx]
                results.append(result)

    # Save results
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Test results saved to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test ConvNeXt-L model for ophthalmology"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root directory of dataset"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model weights"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Output CSV file path"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loader workers"
    )

    args = parser.parse_args()
    main(args)
