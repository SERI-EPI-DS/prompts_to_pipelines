# ------------------------------------------------
# === test_convnext.py ===
# Usage example:
#   python test_convnext.py \
#          --test_dir "/path/to/dataset/test" \
#          --model_path "./results/best_model.pth" \
#          --class_map "./results/training_log.json"
# ------------------------------------------------

import argparse
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ConvNeXt‑L fundus classifier on a test set"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Folder containing test images organised in class subfolders",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model .pth file (state_dict)",
    )
    parser.add_argument(
        "--class_map",
        type=str,
        required=False,
        help="Optional JSON log containing class order (from training)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def load_classes(class_map_path, dataset_classes):
    if class_map_path and Path(class_map_path).exists():
        try:
            with open(class_map_path) as f:
                train_log = json.load(f)
            # first log entry has classes list
            return (
                train_log[0]["classes"]
                if isinstance(train_log, list)
                else dataset_classes
            )
        except Exception:
            pass
    return dataset_classes


def build_model(num_classes):
    model = models.convnext_large(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


if __name__ == "__main__":
    args = parse_args()

    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_ds = datasets.ImageFolder(args.test_dir, transform=tfm)
    class_names = load_classes(args.class_map, test_ds.classes)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names))
    state_dict = torch.load(args.model_path, map_location=device)
    # handle state_dict whether full checkpoint or just model state
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("Confusion Matrix:")
    print(cm_df)

    # optional: save metrics to CSV
    out_dir = Path(args.model_path).parent
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)
    print(f"Metrics saved to {out_dir}")
