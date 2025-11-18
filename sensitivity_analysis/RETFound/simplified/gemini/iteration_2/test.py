import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import argparse
import json
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Suppress timm warnings about hub
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='timm.models._hub')

def test_model(data_dir, model_path, output_dir, batch_size=32):
    """
    Tests the fine-tuned RETFound classifier.

    Args:
        data_dir (str): Path to the test data directory.
        model_path (str): Path to the saved fine-tuned model weights.
        output_dir (str): Directory to save the evaluation results.
        batch_size (int): Batch size for testing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading ---
    # IMPORTANT: Use the same normalization as in validation
    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load class mapping saved during training
    class_map_path = os.path.join(os.path.dirname(model_path), 'class_map.json')
    try:
        with open(class_map_path, 'r') as f:
            class_map = json.load(f)
        class_names = list(class_map['class_to_idx'].keys())
        num_classes = len(class_names)
        print(f"Loaded {num_classes} classes: {class_names}")
    except FileNotFoundError:
        print("Error: class_map.json not found. Using class names from ImageFolder.")
        class_names = test_dataset.classes
        num_classes = len(class_names)


    # --- Model Loading ---
    model = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded in evaluation mode.")

    # --- Evaluation ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Results Generation ---
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)
    
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a RETFound-based classifier.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the test data directory.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.pth) file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the evaluation results.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for testing.")
    
    args = parser.parse_args()
    test_model(args.data_dir, args.model_path, args.output_dir, args.batch_size)
