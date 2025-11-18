import os
import sys
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

# Import RETFound utilities
sys.path.append("../RETFound")
from models_vit import VisionTransformer
from util.pos_embed import interpolate_pos_embed


def get_args_parser():
    parser = argparse.ArgumentParser("RETFound testing", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model", default="vit_large_patch16", type=str, help="Name of model to test"
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--nb_classes",
        default=None,
        type=int,
        help="number of classes (automatically detected if not specified)",
    )

    # Data parameters
    parser.add_argument(
        "--data_path", default="../data/", type=str, help="dataset path"
    )
    parser.add_argument(
        "--output_dir",
        default="../project/results/",
        type=str,
        help="path where to save results",
    )
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda", help="device to use for testing")

    # Global pooling
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    return parser


class TestDataset(Dataset):
    """Custom dataset to keep track of image paths."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.paths = []

        # Walk through directory structure
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                    ):
                        img_path = os.path.join(class_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(class_idx)
                        self.paths.append(os.path.relpath(img_path, root_dir))

        self.class_names = sorted(os.listdir(root_dir))
        print(f"Found {len(self.images)} images in {len(self.class_names)} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, self.paths[idx]


def build_transform(args):
    """Build test image transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))

    return transforms.Compose(t)


def create_model(args):
    """Create the Vision Transformer model."""
    model = VisionTransformer(
        img_size=args.input_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
    )

    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load fine-tuned model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Handle potential module prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return model


@torch.no_grad()
def test_model(model, data_loader, device, output_path, class_names):
    """Test model and save results to CSV."""
    model.eval()

    # Prepare CSV file
    csv_path = os.path.join(output_path, "test_results.csv")

    # Prepare headers
    headers = (
        ["image_path"]
        + [f"{class_name}_score" for class_name in class_names]
        + ["predicted_class"]
    )

    all_results = []

    print("Starting evaluation...")
    for batch in tqdm(data_loader, desc="Testing"):
        images, labels, paths = batch
        images = images.to(device, non_blocking=True)

        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)

        # Get predictions
        _, predicted = torch.max(outputs, 1)

        # Process each sample in batch
        for i in range(images.size(0)):
            result = {
                "image_path": paths[i],
                "scores": probabilities[i].cpu().numpy(),
                "predicted_class": class_names[predicted[i].cpu().item()],
            }
            all_results.append(result)

    # Write results to CSV
    print(f"\nSaving results to {csv_path}")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for result in all_results:
            row = [result["image_path"]]
            row.extend(result["scores"].tolist())
            row.append(result["predicted_class"])
            writer.writerow(row)

    # Calculate and print accuracy if labels are available
    correct = 0
    total = len(all_results)

    # Create confusion matrix
    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)

    # Rerun to calculate accuracy (since we have ground truth labels from folder structure)
    for i, (images, labels, paths) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()

        # Update confusion matrix
        for j in range(labels.size(0)):
            confusion_matrix[labels[j].cpu().item(), predicted[j].cpu().item()] += 1

    accuracy = 100.0 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # Save confusion matrix
    cm_path = os.path.join(output_path, "confusion_matrix.csv")
    np.savetxt(cm_path, confusion_matrix, delimiter=",", fmt="%d")
    print(f"Confusion matrix saved to {cm_path}")

    # Save summary statistics
    summary_path = os.path.join(output_path, "test_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Test Results Summary\n")
        f.write(f"===================\n\n")
        f.write(f"Total test samples: {total}\n")
        f.write(f"Correct predictions: {correct}\n")
        f.write(f"Test accuracy: {accuracy:.2f}%\n\n")
        f.write(f"Class-wise accuracy:\n")
        for i, class_name in enumerate(class_names):
            class_total = confusion_matrix[i].sum()
            class_correct = confusion_matrix[i, i]
            class_acc = 100.0 * class_correct / class_total if class_total > 0 else 0
            f.write(f"{class_name}: {class_correct}/{class_total} ({class_acc:.2f}%)\n")

    print(f"Summary saved to {summary_path}")


def main():
    args = get_args_parser()
    args = args.parse_args()

    # Set device
    device = torch.device(args.device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect number of classes if not specified
    test_dir = os.path.join(args.data_path, "test")
    if args.nb_classes is None:
        args.nb_classes = len(
            [
                d
                for d in os.listdir(test_dir)
                if os.path.isdir(os.path.join(test_dir, d))
            ]
        )
        print(f"Detected {args.nb_classes} classes")

    # Create dataset and dataloader
    transform = build_transform(args)
    dataset = TestDataset(test_dir, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Create and load model
    print(f"Creating model: {args.model}")
    model = create_model(args)
    model = load_checkpoint(model, args.checkpoint_path, device)
    model.to(device)

    # Run testing
    test_model(model, data_loader, device, args.output_dir, dataset.class_names)

    print("\nTesting completed successfully!")


if __name__ == "__main__":
    main()
