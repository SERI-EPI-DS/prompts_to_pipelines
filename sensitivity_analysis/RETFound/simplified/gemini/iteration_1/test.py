# test.py (Updated)

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial # Import partial

# Import the RETFound model definition
import models_vit 

def get_model_for_testing(num_classes, model_path):
    """
    Loads the fine-tuned model architecture and weights for evaluation.
    """
    # --- CHANGE ---
    # Directly instantiate the VisionTransformer with parameters for 'vit_large_patch16'
    model = models_vit.VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    # The head will be replaced by the loaded state_dict
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    # --- END CHANGE ---

    # Load the fine-tuned weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Transformations for Testing ---
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- Dataset and DataLoader ---
    train_dir = os.path.join(args.data_dir, 'train')
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}. It's needed to infer class names.")
    
    class_names = sorted([d.name for d in os.scandir(train_dir) if d.is_dir()])
    num_classes = len(class_names)
    print(f"Inferred {num_classes} classes: {', '.join(class_names)}")
    
    test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), data_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Test set size: {len(test_dataset)}")

    # --- Model Loading ---
    model = get_model_for_testing(num_classes=num_classes, model_path=args.model_path)
    model = model.to(device)
    model.eval()

    # --- Evaluation ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Metrics and Reporting ---
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(args.output_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    print("\nClassification Report:")
    print(report_df)

    # 2. Overall Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")
    with open(os.path.join(args.output_dir, 'test_accuracy.txt'), 'w') as f:
        f.write(f"Overall Test Accuracy: {accuracy:.4f}\n")

    # 3. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a fine-tuned RETFound classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory (containing test folder)')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the test results')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model weights (.pth file)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    
    args = parser.parse_args()
    main(args)
