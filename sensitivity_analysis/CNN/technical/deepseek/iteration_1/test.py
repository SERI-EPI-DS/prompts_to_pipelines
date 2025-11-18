import os
import argparse
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Argument parser
parser = argparse.ArgumentParser(description="ConvNext-L Testing")
parser.add_argument(
    "--data_root", type=str, default="../data", help="Root directory for dataset"
)
parser.add_argument(
    "--results_dir", type=str, default="../results", help="Directory to save results"
)
parser.add_argument(
    "--model_path", type=str, default="best_model.pth", help="Path to trained model"
)
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test dataset
test_dir = os.path.join(args.data_root, "test")
test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

# Load model
model = models.convnext_large()
num_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, len(test_dataset.classes))

# Load weights with map_location to handle device differences
model_path = os.path.join(args.results_dir, args.model_path)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Create results file
results_file = os.path.join(args.results_dir, "test_results.csv")
class_names = test_dataset.classes

# Testing loop
with open(results_file, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["image_path", "true_label", "predicted_label"] + [
        f"score_{cls}" for cls in class_names
    ]
    writer.writerow(header)

    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

        # Get original image paths
        batch_paths = [
            test_dataset.samples[idx][0]
            for idx in range(
                i * args.batch_size, min((i + 1) * args.batch_size, len(test_dataset))
            )
        ]

        for j in range(inputs.size(0)):
            row = [batch_paths[j], class_names[labels[j]], class_names[preds[j]]] + [
                f"{probs[j][k]:.6f}" for k in range(len(class_names))
            ]
            writer.writerow(row)

print(f"Test results saved to {results_file}")
