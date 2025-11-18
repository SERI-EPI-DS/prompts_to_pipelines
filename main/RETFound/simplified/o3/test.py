#!/usr/bin/env python
"""
Evaluate a fine-tuned RETFound/timm ViT on a fundus dataset.

Example – evaluate the checkpoint you just trained:
python eval_retfound.py \
    --data_dir /path/to/dataset \
    --checkpoint /path/to/runs/best_RETFound_mae.pth \
    --backbone RETFound_mae

Example – evaluate a public RETFound weight from HuggingFace:
python eval_retfound.py \
    --data_dir /path/to/dataset \
    --ckpt_url https://huggingface.co/open-eye/RETFound_MAE/resolve/main/RETFound_mae_natureCFP.pth \
    --backbone RETFound_mae
"""
# --------------------------------------------------------------------------- #
import argparse, json, torch
from pathlib import Path
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import timm

# repo-local utilities
import models_vit  # RETFound repo
from util.pos_embed import interpolate_pos_embed  # RETFound util


# --------------------------------------------------------------------------- #
def build_model(backbone: str, nb_classes: int):
    """Return a ViT from RETFound (if present) or from timm."""
    if hasattr(models_vit, backbone):  # e.g. RETFound_mae
        model_fn = getattr(models_vit, backbone)
        model = model_fn(num_classes=nb_classes, global_pool=True, drop_path_rate=0.0)
    else:  # any timm ViT name
        model = timm.create_model(
            backbone, pretrained=False, num_classes=nb_classes, global_pool="avg"
        )
    return model


def load_checkpoint(model, ckpt_path=None, ckpt_url=None):
    """Load weights, interpolating positional embeddings when needed."""
    if ckpt_path is None and ckpt_url is None:
        return
    state = (
        torch.hub.load_state_dict_from_url(ckpt_url, map_location="cpu")
        if ckpt_url
        else torch.load(ckpt_path, map_location="cpu")
    )
    state = state.get("model", state)  # unwrap
    # strip head if shape mismatch
    for k in ["head.weight", "head.bias"]:
        if k in state and state[k].shape != model.state_dict()[k].shape:
            del state[k]
    interpolate_pos_embed(model, state)
    _ = model.load_state_dict(state, strict=False)


# --------------------------------------------------------------------------- #
def get_loader(split_dir, batch, workers=8, img_size=224):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    ds = datasets.ImageFolder(split_dir, transform=tf)
    ld = torch.utils.data.DataLoader(
        ds, batch, False, num_workers=workers, pin_memory=True
    )
    return ds, ld


# --------------------------------------------------------------------------- #
def evaluate(args):
    ds, loader = get_loader(
        Path(args.data_dir) / "test", args.batch, img_size=args.img_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, len(ds.classes)).to(device)
    load_checkpoint(model, args.checkpoint, args.ckpt_url)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            y_hat = model(x.to(device)).argmax(1).cpu()
            y_true.extend(y.numpy())
            y_pred.extend(y_hat.numpy())

    print(classification_report(y_true, y_pred, target_names=ds.classes))
    print("Confusion-matrix:\n", confusion_matrix(y_true, y_pred))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir", required=True, help="root with train/val/test sub-folders"
    )
    ap.add_argument(
        "--backbone",
        default="RETFound_mae",
        help="RETFound_mae | RETFound_dinov2 | any timm ViT",
    )
    ap.add_argument(
        "--checkpoint", default=None, help="(optional) local .pth file to load"
    )
    ap.add_argument(
        "--ckpt_url",
        default=None,
        help="(optional) remote URL to .pth file (e.g. HuggingFace)",
    )
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    evaluate(args)
