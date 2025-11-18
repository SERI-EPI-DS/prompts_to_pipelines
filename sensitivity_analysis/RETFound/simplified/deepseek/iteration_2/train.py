import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import os
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTForImageClassification, ViTImageProcessor


def get_args_parser():
    parser = argparse.ArgumentParser("ViT Fine-tuning", add_help=False)
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Path to save models and logs",
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    return parser


def build_transform():
    """Build data transformations for ViT"""
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # ViT default normalization
        ]
    )


def main():
    args = get_args_parser().parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, f"logs_{current_time}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Data loading
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    transform = build_transform()

    dataset_train = ImageFolder(train_dir, transform=transform)
    dataset_val = ImageFolder(val_dir, transform=transform)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model setup - Using Hugging Face ViT
    print(f"Loading ViT model: {args.model_name}")
    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True,  # Crucial for replacing the classification head:cite[7]
    )
    model = model.cuda()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader_train):
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch: {epoch} | Batch: {batch_idx}/{len(dataloader_train)} | Loss: {loss.item():.4f}"
                )

        train_acc = 100.0 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader_val:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs.logits, targets)

                val_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_acc = 100.0 * correct / total

        scheduler.step()

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss / len(dataloader_train), epoch)
        writer.add_scalar("Loss/val", val_loss / len(dataloader_val), epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(
            f"Epoch: {epoch} | Train Loss: {train_loss/len(dataloader_train):.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(dataloader_val):.4f} | Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "class_to_idx": dataset_train.class_to_idx,
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )

    writer.close()
    print(f"Training completed. Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
