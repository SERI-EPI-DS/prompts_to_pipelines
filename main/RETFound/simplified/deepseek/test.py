import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm
from sklearn.metrics import confusion_matrix, classification_report

# Configuration
parser = argparse.ArgumentParser(description="Test RETFound classifier")
parser.add_argument(
    "--data_dir", type=str, required=True, help="Root dataset directory"
)
parser.add_argument(
    "--model_path", type=str, required=True, help="Path to trained model"
)
parser.add_argument("--batch_size", type=int, default=32, help="Input batch size")
parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers")
args = parser.parse_args()

# Test transformation
test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ]
)

# Load dataset
test_dataset = datasets.ImageFolder(
    root=os.path.join(args.data_dir, "test"), transform=test_transform
)

# Data loader
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

# Load model
checkpoint = torch.load(args.model_path)
class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Use ViT-Large to match training configuration
model = timm.create_model(
    "vit_large_patch16_224", pretrained=False, num_classes=len(class_to_idx)
)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Testing loop
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"Test Accuracy: {accuracy:.2f}%")
print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
