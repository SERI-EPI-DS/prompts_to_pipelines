import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def test_model(model, dataloader, device, class_names, results_dir="results"):
    """
    Tests a PyTorch model.

    Args:
        model (torch.nn.Module): The trained model to test.
        dataloader (torch.utils.data.DataLoader): The dataloader for the test set.
        device (torch.device): The device to test on ('cuda:0' or 'cpu').
        class_names (list): A list of the class names.
        results_dir (str): The directory to save the confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a ConvNext-L classifier for ophthalmic diagnosis."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where the trained model and results are stored.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing."
    )
    args = parser.parse_args()

    # Data transformation for the test set
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), data_transform
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Load class names
    class_names_path = os.path.join(args.results_dir, "class_names.txt")
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(
            f"Class names file not found at {class_names_path}. Please run the training script first."
        )
    with open(class_names_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    num_classes = len(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    model_path = os.path.join(args.results_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please run the training script first."
        )

    model = models.convnext_large(weights=None)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # Test the model
    test_model(model, test_dataloader, device, class_names, args.results_dir)
