"""
Evaluate a fine‑tuned RETFound model on a test dataset.

This script addresses issues locating the `main_finetune` module when the
script is not placed within the RETFound_MAE repository.  It can
accept a `--retfound_dir` argument pointing directly to the
RETFound_MAE clone, fall back on a `RETFOND_DIR` environment
variable or search upwards from the current location and the current
working directory for `main_finetune.py`.  If the module still cannot
be found, a clear error message is raised.

Example usage:

```
python test.py \
    --data_dir /path/to/dataset \
    --checkpoint /path/to/output_dir/task/checkpoint-best.pth \
    --num_classes 5 \
    --output_dir /path/to/eval_results \
    --retfound_dir /path/to/RETFound_MAE
```

See the accompanying documentation in `test_classifier.py` for more
details about the expected dataset structure and outputs.
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
    retfound_dir : str | None
        Optional path to the cloned RETFound_MAE repository.  If provided
        and contains `main_finetune.py`, it will be appended to
        `sys.path`.  If not provided, the function tries an environment
        variable (`RETFOND_DIR`) and then searches up the filesystem
        hierarchy from both the script location and the current
        working directory.

    Raises
    ------
    ImportError
        If `main_finetune.py` cannot be located.
    """
    candidates: list[Path] = []
    # 1. Use explicit argument
    if retfound_dir:
        candidate = Path(retfound_dir).expanduser().resolve()
        if (candidate / "main_finetune.py").exists():
            candidates.append(candidate)
    # 2. Environment variable
    if not candidates:
        env_dir = os.environ.get("RETFOND_DIR")
        if env_dir:
            candidate = Path(env_dir).expanduser().resolve()
            if (candidate / "main_finetune.py").exists():
                candidates.append(candidate)
    # 3. Search from script location
    if not candidates:
        here = Path(__file__).resolve()
        for parent in [here.parent] + list(here.parents):
            if (parent / "main_finetune.py").exists():
                candidates.append(parent)
                break
    # 4. Search from current working directory
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
    sys.path.append(str(candidates[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‑tuned RETFound classifier on the test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root folder of your dataset containing train/val/test subdirectories.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved checkpoint (.pth) from training (e.g. checkpoint-best.pth).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of target classes; must match the value used during training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_eval",
        help="Directory where evaluation results will be saved.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input image resolution used during training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RETFound_mae",
        help="Model architecture used during training.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Optional task name for organising evaluation outputs. Derived if omitted.",
    )
    parser.add_argument(
        "--retfound_dir",
        type=str,
        default=None,
        help="Path to the cloned RETFound_MAE repository containing main_finetune.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Ensure main_finetune is importable
    add_retfound_to_sys_path(args.retfound_dir)
    # Import after path modification
    import main_finetune as mf  # type: ignore

    # Build default arguments
    finetune_parser = mf.get_args_parser()
    eval_args = finetune_parser.parse_args([])
    # Populate evaluation parameters
    eval_args.data_path = args.data_dir
    eval_args.output_dir = args.output_dir
    eval_args.nb_classes = args.num_classes
    eval_args.input_size = args.input_size
    eval_args.model = args.model
    eval_args.resume = args.checkpoint
    eval_args.eval = True
    # Set task name
    if args.task_name:
        eval_args.task = args.task_name
    else:
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        dataset_name = os.path.basename(os.path.abspath(args.data_dir.rstrip("/")))
        eval_args.task = f"eval-{checkpoint_name}-{dataset_name}"
    # Criterion (unused in eval but required by API)
    criterion = torch.nn.CrossEntropyLoss()
    # Run evaluation
    mf.main(eval_args, criterion)


if __name__ == "__main__":
    main()
