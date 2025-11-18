import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from transformers import ViTForImageClassification


def get_args_parser():
    parser = argparse.ArgumentParser("ViT Testing", add_help=False)
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vit-large-patch16-224",
        help="Hugging Face ViT model name",
    )
    parser.add_argument(
        "--num_classes", type=int, required=True, help="Number of diagnostic classes"
    )
    return parser


def main():
    args = get_args_parser().parse_args()

    # Data loading
    test_dir = os.path.join(args.data_dir, "test")

    transform_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset_test = ImageFolder(test_dir, transform=transform_test)
    dataloader_test = DataLoader(
        dataset_test, batch_size=16, shuffle=False, num_workers=4
    )

    # Load fine-tuned model
    print(f"Loading fine-tuned model from: {args.model_path}")
    model = ViTForImageClassification.from_pretrained(
        args.model_name, num_labels=args.num_classes, ignore_mismatched_sizes=True
    )

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()
    model.eval()

    # Testing
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader_test:
            inputs = inputs.cuda()
            outputs = model(inputs)
            _, predicted = outputs.logits.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Generate reports
    class_names = [dataset_test.classes[i] for i in range(len(dataset_test.classes))]

    print("Classification Report:")
    print(
        classification_report(
            all_targets, all_predictions, target_names=class_names, digits=4
        )
    )

    cm = confusion_matrix(all_targets, all_predictions)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("\nConfusion Matrix:")
    print(cm_df)

    accuracy = (
        (torch.tensor(all_predictions) == torch.tensor(all_targets)).float().mean()
    )
    print(f"\nOverall Test Accuracy: {accuracy.item()*100:.2f}%")


if __name__ == "__main__":
    main()
