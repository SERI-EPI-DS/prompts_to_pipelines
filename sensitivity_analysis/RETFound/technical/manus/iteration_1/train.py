import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothingCrossEntropy class.
        :param smoothing: The smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def main():
    parser = argparse.ArgumentParser(description="RETFound Finetuning")
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
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train for"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing")
    args = parser.parse_args()

    # Add RETFound to the Python path
    sys.path.append(args.retfound_dir)

    from models_vit import RETFound_mae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image transformations
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Datasets and Dataloaders
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"), transform=transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"), transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    num_classes = len(train_dataset.classes)

    # Model
    model = RETFound_mae(num_classes=num_classes)
    model.to(device)

    # Load pre-trained weights
    checkpoint_path = os.path.join(args.retfound_dir, "RETFound_CFP_weights.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    # Loss and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_acc = correct / total

        print(
            f"Epoch {epoch+1}/{args.epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(args.results_dir, "best_model.pth")
            )


if __name__ == "__main__":
    main()
