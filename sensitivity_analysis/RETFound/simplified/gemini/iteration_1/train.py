# train.py (Updated)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
from sklearn.metrics import accuracy_score
from functools import partial

# Import the RETFound model definition
import models_vit


# --- NEW FUNCTION ---
# The missing interpolate_pos_embed function is now defined directly in this script.
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (h) & width (w) for interpolation
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}"
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = F.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


# --- END NEW FUNCTION ---


def get_retfound_model(num_classes, pretrained_weights_path):
    """
    Loads the RETFound model, freezes its backbone, and adds a new classification head.
    """
    model = models_vit.VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    checkpoint = torch.load(
        pretrained_weights_path, map_location="cpu", weights_only=False
    )
    print("Loading pre-trained model from %s" % pretrained_weights_path)
    checkpoint_model = checkpoint["model"]

    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # --- CHANGE ---
    # Call the local version of the function instead of the one from models_vit
    interpolate_pos_embed(model, checkpoint_model)
    # --- END CHANGE ---

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    for param in model.head.parameters():
        param.requires_grad = True

    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    num_classes = len(class_names)

    print(f"Found {num_classes} classes: {', '.join(class_names)}")
    print(
        f"Training set size: {dataset_sizes['train']}, Validation set size: {dataset_sizes['val']}"
    )

    model = get_retfound_model(
        num_classes=num_classes, pretrained_weights_path=args.model_path
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.05,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_model_weights = model.state_dict()
    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = accuracy_score(all_labels, all_preds)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = model.state_dict()
                torch.save(
                    model.state_dict(), os.path.join(args.output_dir, "best_model.pth")
                )
                print(f"New best validation accuracy: {best_acc:.4f}. Model saved.")

        if phase == "train":
            scheduler.step()

    print(f"Best val Acc: {best_acc:4f}")
    model.load_state_dict(best_model_weights)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RETFound Fine-tuning for Image Classification"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset directory (containing train/val folders)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the trained model and results",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained RETFound weights (.pth file)",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate"
    )

    args = parser.parse_args()
    main(args)
