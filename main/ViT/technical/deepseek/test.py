import os
import argparse
import csv
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import swin_v2_b


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Swin-V2-B Ophthalmology Classifier"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of dataset (contains test folder)",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model weights"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Path to output CSV file"
    )
    parser.add_argument(
        "--img_size", type=int, default=256, help="Input image size (default: 256)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and class mappings
    checkpoint = torch.load(args.model_path)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Initialize model
    model = swin_v2_b()
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features, len(class_to_idx))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    # Create test dataset and loader
    test_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = ImageFolder(
        root=os.path.join(args.data_root, "test"), transform=test_transform
    )

    # Ensure consistent class mapping
    test_dataset.class_to_idx = class_to_idx

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    print(f"Found {len(test_dataset)} test images")

    # Run inference
    results = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            for i in range(images.size(0)):
                img_path = test_dataset.samples[i][0]
                file_name = os.path.basename(img_path)
                scores = probs[i].cpu().numpy()
                predicted_class = idx_to_class[torch.argmax(probs[i]).item()]

                # Add to results
                results.append(
                    {
                        "image": file_name,
                        "scores": scores,
                        "prediction": predicted_class,
                    }
                )

    # Save results to CSV
    with open(args.output_csv, "w", newline="") as csvfile:
        fieldnames = ["image", "prediction"] + [
            f"score_{idx_to_class[i]}" for i in range(len(idx_to_class))
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for res in results:
            row = {"image": res["image"], "prediction": res["prediction"]}
            for i, score in enumerate(res["scores"]):
                row[f"score_{idx_to_class[i]}"] = score
            writer.writerow(row)

    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
