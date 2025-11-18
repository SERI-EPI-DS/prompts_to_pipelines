"""
test.py
=======

This script evaluates a fine‑tuned ViT‑Large model on a held‑out test
set of colour fundus images.  It loads the model weights saved by
``train.py``, runs inference on each test image, computes softmax
probabilities for every class and writes a comma‑separated values
(CSV) file containing the file name, per‑class scores and the final
predicted class.  The dataset is expected to follow the same
directory structure described in ``train.py``::

    data/
      test/
        class_a/
          img_001.png
          img_002.png
          ...
        class_b/
        ...

Although ground‑truth labels exist in the directory names, the script
does not rely on them for computing predictions; they are used only
to determine the mapping between class indices and human‑readable
class names.  The predictions are saved as a CSV with the following
columns::

    file_name,score_class_a,score_class_b,...,prediction

where ``score_class_*`` are the softmax probabilities for each class
and ``prediction`` is the class name with the highest probability.

Usage example:

    python test.py \
        --data_path /path/to/data \
        --model_path ./project/results/best_model.pth \
        --output_csv ./project/results/test_predictions.csv

The implementation reuses the same normalisation statistics and
transforms as in training【341698127874478†L329-L365】.  It is designed to run on
GPU if available.
"""

import argparse
import csv
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import torchvision
    from torchvision import transforms, datasets
except ImportError as exc:
    raise ImportError(
        "torchvision is required for test.py; please install it in your environment"
    ) from exc


class ImageFolderWithPaths(datasets.ImageFolder):
    """Subclass of ImageFolder that returns the file path along with the sample.

    This allows us to record the original file names for the CSV output
    while still leveraging the directory structure for class labels.
    """

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return sample, target, path


def build_test_loader(
    root_dir: str, input_size: int, batch_size: int, num_workers: int
) -> Tuple[DataLoader, List[str]]:
    """Prepare a DataLoader for the test split and return the class names."""
    test_dir = os.path.join(root_dir, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Could not find a 'test' directory under {root_dir}.")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    dataset = ImageFolderWithPaths(test_dir, transform=test_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    class_names = dataset.classes
    return dataloader, class_names


def build_model(num_classes: int, checkpoint_path: str) -> nn.Module:
    """Recreate the ViT‑Large/16 model and load fine‑tuned weights."""
    model = torchvision.models.vit_l_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered_state_dict[k] = v
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    if missing:
        print(
            f"Warning: {len(missing)} keys were missing when loading the model; the corresponding layers were initialised randomly."
        )
    if unexpected:
        print(
            f"Warning: {len(unexpected)} unexpected keys in the checkpoint were ignored."
        )
    return model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the test script."""
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned RETFound model on a test set"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset root containing a 'test' subdirectory",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine‑tuned model checkpoint (best_model.pth)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the CSV file where predictions will be saved",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input image size (height and width)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Prepare DataLoader and class names
    test_loader, class_names = build_test_loader(
        args.data_path, args.input_size, args.batch_size, args.num_workers
    )

    # Rebuild model and load the fine‑tuned weights
    model = build_model(num_classes=len(class_names), checkpoint_path=args.model_path)
    model.to(device)
    model.eval()

    # We'll compute softmax probabilities for each test image
    results: List[List] = []

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs, _, paths in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = softmax(outputs)
            for i in range(inputs.size(0)):
                file_name = os.path.basename(paths[i])
                prob_vector = probs[i].cpu().tolist()
                pred_idx = int(torch.argmax(probs[i]).item())
                pred_class = class_names[pred_idx]
                row = [file_name] + prob_vector + [pred_class]
                results.append(row)

    # Create header: 'file_name', 'score_<class1>', ..., 'prediction'
    header = ["file_name"] + [f"score_{cls}" for cls in class_names] + ["prediction"]
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    print(f"Saved predictions to {args.output_csv}.")


if __name__ == "__main__":
    main()
