import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.models import convnext_large
import os
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Argument parser
parser = argparse.ArgumentParser(description="ConvNext-L Testing")
parser.add_argument(
    "--test_dir", type=str, required=True, help="Path to testing dataset"
)
parser.add_argument(
    "--model_path", type=str, required=True, help="Path to trained model"
)
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = torch.load(args.model_path)
model = convnext_large(pretrained=False, num_classes=len(checkpoint["classes"]))
model.load_state_dict(checkpoint["model"])
model = model.to(device).eval()

# Data transformations
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create dataset and dataloader
test_dataset = datasets.ImageFolder(args.test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Ensure class mappings match training
assert (
    test_dataset.classes == checkpoint["classes"]
), "Class mismatch between test data and trained model"

# Testing loop
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calculate metrics
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(
    all_labels, all_preds, target_names=test_dataset.classes
)

# Print results
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Save results
with open("test_results.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(class_report)
