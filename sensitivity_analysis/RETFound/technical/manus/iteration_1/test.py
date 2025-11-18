import argparse
import os
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys


def main():
    parser = argparse.ArgumentParser(description="RETFound Testing")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to the results directory"
    )
    parser.add_argument(
        "--retfound_dir",
        type=str,
        required=True,
        help="Path to the RETFound repository directory",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model weights",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    args = parser.parse_args()

    # Add RETFound to the Python path
    sys.path.append(args.retfound_dir)

    from models_vit import RETFound_mae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image transformations
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Dataset and Dataloader
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes

    # Model
    model = RETFound_mae(num_classes=num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            for j in range(images.size(0)):
                image_path, _ = test_loader.dataset.samples[i * args.batch_size + j]
                filename = os.path.basename(image_path)
                result = {"filename": filename}
                for k, class_name in enumerate(class_names):
                    result[f"{class_name}_score"] = scores[j, k].item()
                result["prediction"] = class_names[predicted[j]]
                results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.results_dir, "test_results.csv"), index=False)


if __name__ == "__main__":
    main()
