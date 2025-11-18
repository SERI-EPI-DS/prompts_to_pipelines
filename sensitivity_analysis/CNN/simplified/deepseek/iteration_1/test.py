import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Test ConvNeXt-L model")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Root directory of dataset"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Input batch size")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loader workers"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save results"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)  # Correct function here too
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test transformation
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create test dataset
    test_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_dir, "test"), transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Load model with metadata
    checkpoint = torch.load(args.model_path)
    class_to_idx = checkpoint.get("class_to_idx", None)

    model = models.convnext_large()
    num_classes = (
        len(test_dataset.classes) if class_to_idx is None else len(class_to_idx)
    )
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)

    # Handle state dict (for both single and multi-GPU models)
    state_dict = checkpoint["state_dict"]
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # Evaluation
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Generate reports
    class_names = list(class_to_idx.keys()) if class_to_idx else test_dataset.classes
    cr = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    cm = confusion_matrix(all_labels, all_preds)

    print("Classification Report:")
    print(cr)

    # Save results
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(cr)

    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), cm)
    np.save(os.path.join(args.output_dir, "probabilities.npy"), np.vstack(all_probs))

    # Save per-class accuracy
    class_acc = {}
    for i, class_name in enumerate(class_names):
        acc = np.diag(cm)[i] / cm[i].sum() if cm[i].sum() > 0 else 0
        class_acc[class_name] = acc

    with open(os.path.join(args.output_dir, "class_accuracy.json"), "w") as f:
        json.dump(class_acc, f, indent=2)

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
