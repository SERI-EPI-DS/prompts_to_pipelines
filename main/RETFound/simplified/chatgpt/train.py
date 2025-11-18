"""
Fine‑tune the RETFound foundation model on a custom dataset.

This script is an improved version of the training wrapper provided
previously.  It adds flexibility for locating the RETFound_MAE
repository by either searching parent directories, reading an
environment variable (`RETFOND_DIR`) or accepting an explicit
command‑line argument (`--retfound_dir`).  These mechanisms prevent
`ModuleNotFoundError: No module named 'main_finetune'` when the
script is not co‑located with `main_finetune.py`.

See `train_classifier.py` for detailed usage instructions.  In
general, the recommended way to run this script is:

```
python train.py \
    --data_dir /path/to/dataset \
    --output_dir /path/to/results \
    --num_classes 5 \
    --finetune RETFound_mae_meh \
    --epochs 100 \
    --batch_size 16 \
    --retfound_dir /path/to/RETFound_MAE
```

If you omit `--retfound_dir`, the script will try to locate
`main_finetune.py` in your current directory and its parents.  If the
RETFound repository lives elsewhere, set `RETFOND_DIR` or use
`--retfound_dir`.
"""

import argparse
import os
import sys
from pathlib import Path
import torch


def add_retfound_to_sys_path(retfound_dir: str | None = None) -> None:
    """Ensure that `main_finetune.py` from the RETFound repository is importable.

    Parameters
    ----------
    retfound_dir : str | None, optional
        A user‑supplied path pointing to the root of the cloned
        `RETFound_MAE` repository.  If provided and valid, this
        directory will be appended to `sys.path`.  If not provided,
        the function will try an environment variable (`RETFOND_DIR`)
        and then search for `main_finetune.py` in the current
        directory and its parents.

    Raises
    ------
    ImportError
        If `main_finetune.py` cannot be located via any of the
        mechanisms.
    """
    candidates: list[Path] = []
    # 1. Use explicit argument if provided
    if retfound_dir:
        candidate = Path(retfound_dir).expanduser().resolve()
        if (candidate / "main_finetune.py").exists():
            candidates.append(candidate)
    # 2. Fall back to environment variable
    if not candidates:
        env_dir = os.environ.get("RETFOND_DIR")
        if env_dir:
            candidate = Path(env_dir).expanduser().resolve()
            if (candidate / "main_finetune.py").exists():
                candidates.append(candidate)
    # 3. Search upwards from the script location
    if not candidates:
        here = Path(__file__).resolve()
        for parent in [here.parent] + list(here.parents):
            if (parent / "main_finetune.py").exists():
                candidates.append(parent)
                break
    # 4. Search current working directory and its parents
    if not candidates:
        cwd = Path.cwd().resolve()
        for parent in [cwd] + list(cwd.parents):
            if (parent / "main_finetune.py").exists():
                candidates.append(parent)
                break
    if not candidates:
        raise ImportError(
            "Unable to locate main_finetune.py. Please run this script inside "
            "the RETFound_MAE repository or specify --retfound_dir/RETFOND_DIR."
        )
    # Add the first valid candidate to sys.path
    sys.path.append(str(candidates[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine‑tune the RETFound foundation model on a custom dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Path to dataset
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root folder of your dataset. Must contain train/val/test subfolders with class‑specific directories.",
    )
    # Output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_finetune",
        help="Directory where checkpoints and logs will be saved.",
    )
    # Number of classes
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of target classes in your classification task.",
    )
    # Pre‑trained weight name on HuggingFace
    parser.add_argument(
        "--finetune",
        type=str,
        required=True,
        help="Name of the RETFound pre‑trained weight to use (e.g. RETFound_mae_meh).",
    )
    # Hyper‑parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size per GPU."
    )
    parser.add_argument(
        "--input_size", type=int, default=224, help="Input image resolution."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RETFound_mae",
        help="Model architecture to fine‑tune.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Optional descriptive name for this fine‑tuning run. Automatically derived if omitted.",
    )
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=None,
        help="Optional HuggingFace access token used to download the pre‑trained weight.",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Absolute learning rate override."
    )
    parser.add_argument("--blr", type=float, default=5e-3, help="Base learning rate.")
    parser.add_argument(
        "--layer_decay", type=float, default=0.65, help="Layer‑wise LR decay factor."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay."
    )
    # Path to RETFound repository
    parser.add_argument(
        "--retfound_dir",
        type=str,
        default=None,
        help="Path to the cloned RETFound_MAE repository. If provided, this is used to locate main_finetune.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Augment sys.path to find main_finetune
    add_retfound_to_sys_path(args.retfound_dir)
    # Now that the path is set up, import main_finetune
    import main_finetune as mf  # type: ignore

    try:
        from huggingface_hub import login as hf_login  # type: ignore
    except Exception:
        hf_login = None
    # Authenticate with HuggingFace if token provided
    if args.huggingface_token and hf_login is not None:
        hf_login(token=args.huggingface_token)
    # Build default arguments from RETFound fine‑tuner
    finetune_parser = mf.get_args_parser()
    finetune_args = finetune_parser.parse_args([])
    # Populate arguments
    finetune_args.data_path = args.data_dir
    finetune_args.output_dir = args.output_dir
    finetune_args.nb_classes = args.num_classes
    finetune_args.epochs = args.epochs
    finetune_args.batch_size = args.batch_size
    finetune_args.input_size = args.input_size
    finetune_args.model = args.model
    finetune_args.finetune = args.finetune
    finetune_args.layer_decay = args.layer_decay
    finetune_args.weight_decay = args.weight_decay
    finetune_args.lr = args.lr
    finetune_args.blr = args.blr
    # Derive task name if not provided
    if args.task_name:
        finetune_args.task = args.task_name
    else:
        dataset_name = os.path.basename(os.path.abspath(args.data_dir.rstrip("/")))
        finetune_args.task = f"{args.finetune}-{dataset_name}"
    # Use global pooling
    finetune_args.global_pool = True
    # Create criterion
    criterion = torch.nn.CrossEntropyLoss()
    # Start training
    mf.main(finetune_args, criterion)


if __name__ == "__main__":
    main()
