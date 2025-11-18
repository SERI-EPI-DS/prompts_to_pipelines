"""
RETFound Fine-tuning Script for Ophthalmology Classification
Author: AI Assistant
Description: Fine-tunes RETFound foundation model for multi-class classification
Updated to handle different PyTorch versions
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import subprocess
import yaml
from typing import Dict, Any
import shutil
import pkg_resources


class RETFoundTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_directories()
        self.save_config()
        self.check_pytorch_version()

    def check_pytorch_version(self):
        """Check PyTorch version and set appropriate torchrun syntax"""
        try:
            torch_version = pkg_resources.get_distribution("torch").version
            major, minor = map(int, torch_version.split(".")[:2])

            # PyTorch 2.0+ uses hyphens in torchrun arguments
            if major >= 2:
                self.torchrun_style = "hyphen"
            else:
                self.torchrun_style = "underscore"

            print(f"Detected PyTorch version: {torch_version}")
            print(f"Using torchrun style: {self.torchrun_style}")
        except:
            # Default to newer style
            self.torchrun_style = "hyphen"
            print(
                "Could not detect PyTorch version, defaulting to newer torchrun syntax"
            )

    def setup_directories(self):
        """Create necessary directories for outputs"""
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)

    def save_config(self):
        """Save configuration for reproducibility"""
        config_path = self.output_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"Configuration saved to {config_path}")

    def get_num_classes(self):
        """Automatically detect number of classes from data directory"""
        train_dir = Path(self.config["data_path"]) / "train"
        if not train_dir.exists():
            raise ValueError(f"Training directory not found: {train_dir}")

        classes = [d for d in train_dir.iterdir() if d.is_dir()]
        num_classes = len(classes)

        # Save class names
        class_names = sorted([c.name for c in classes])
        class_mapping = {name: idx for idx, name in enumerate(class_names)}

        with open(self.output_dir / "class_mapping.json", "w") as f:
            json.dump(class_mapping, f, indent=2)

        print(f"Detected {num_classes} classes: {class_names}")
        return num_classes, class_names

    def detect_torchrun_command(self):
        """Detect the correct torchrun command to use"""
        # Try different torchrun commands
        commands_to_try = [
            "torchrun",
            "python -m torch.distributed.run",
            "python -m torch.distributed.launch",
        ]

        for cmd in commands_to_try:
            try:
                # Test if command exists
                test_cmd = cmd.split() + ["--help"]
                result = subprocess.run(test_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Using distributed launcher: {cmd}")
                    return cmd.split()
            except:
                continue

        raise RuntimeError(
            "Could not find a working distributed launcher (torchrun). "
            "Please ensure PyTorch is properly installed."
        )

    def create_training_script(self):
        """Generate the actual training command"""
        num_classes, _ = self.get_num_classes()

        # Detect the correct launcher
        launcher_cmd = self.detect_torchrun_command()

        # Base command
        cmd = launcher_cmd.copy()

        # Add distributed training arguments with correct syntax
        if self.torchrun_style == "hyphen" or "torch.distributed.run" in " ".join(
            launcher_cmd
        ):
            cmd.extend(
                [
                    "--nproc-per-node",
                    str(self.config["num_gpus"]),
                    "--master-port",
                    str(self.config["master_port"]),
                ]
            )
        else:
            cmd.extend(
                [
                    "--nproc_per_node",
                    str(self.config["num_gpus"]),
                    "--master_port",
                    str(self.config["master_port"]),
                ]
            )

        # Add the main script and arguments
        cmd.extend(
            [
                "main_finetune.py",
                "--model",
                self.config["model_type"],
                "--savemodel",
                "--global_pool",
                "--batch_size",
                str(self.config["batch_size"]),
                "--world_size",
                str(self.config["num_gpus"]),
                "--epochs",
                str(self.config["epochs"]),
                "--blr",
                str(self.config["base_lr"]),
                "--layer_decay",
                str(self.config["layer_decay"]),
                "--weight_decay",
                str(self.config["weight_decay"]),
                "--drop_path",
                str(self.config["drop_path"]),
                "--nb_classes",
                str(num_classes),
                "--data_path",
                self.config["data_path"],
                "--input_size",
                str(self.config["input_size"]),
                "--task",
                self.config["task_name"],
                "--finetune",
                self.config["pretrained_model"],
                "--output_dir",
                str(self.output_dir / "checkpoints"),
            ]
        )

        # Add optional parameters
        if self.config.get("warmup_epochs"):
            cmd.extend(["--warmup_epochs", str(self.config["warmup_epochs"])])

        if self.config.get("min_lr"):
            cmd.extend(["--min_lr", str(self.config["min_lr"])])

        if self.config.get("mixup"):
            cmd.extend(["--mixup", str(self.config["mixup"])])

        if self.config.get("cutmix"):
            cmd.extend(["--cutmix", str(self.config["cutmix"])])

        if self.config.get("reprob"):
            cmd.extend(["--reprob", str(self.config["reprob"])])

        return cmd

    def verify_retfound_installation(self):
        """Verify that RETFound is properly installed"""
        # Check if we're in the right directory
        main_finetune_path = Path("main_finetune.py")
        if not main_finetune_path.exists():
            # Try to find it in parent directories
            possible_paths = [
                Path("../RETFound_MAE/main_finetune.py"),
                Path("RETFound_MAE/main_finetune.py"),
                Path("../main_finetune.py"),
            ]

            for path in possible_paths:
                if path.exists():
                    print(f"Found main_finetune.py at: {path}")
                    # Change to the correct directory
                    os.chdir(path.parent)
                    return True

            raise FileNotFoundError(
                "Could not find main_finetune.py. Please ensure you're running this script "
                "from the RETFound_MAE directory or adjust the path."
            )
        return True

    def run_training(self):
        """Execute the training process"""
        print("\n" + "=" * 50)
        print("Starting RETFound Fine-tuning")
        print("=" * 50 + "\n")

        # Verify RETFound installation
        self.verify_retfound_installation()

        # Create and save the command
        cmd = self.create_training_script()
        cmd_str = " ".join(cmd)

        with open(self.output_dir / "training_command.txt", "w") as f:
            f.write(cmd_str)

        print(f"Training command: {cmd_str}\n")

        # Test the command first with --help to ensure it's valid
        print("Testing command validity...")
        test_cmd = cmd[:3] + ["--help"]  # Just test the launcher
        test_result = subprocess.run(test_cmd, capture_output=True, text=True)
        if test_result.returncode != 0:
            print(f"Command test failed. Error: {test_result.stderr}")
            raise RuntimeError(
                "The distributed launcher command appears to be invalid."
            )

        # Log output
        log_file = (
            self.output_dir
            / "logs"
            / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )

        # Execute training
        try:
            with open(log_file, "w") as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                )

                # Real-time output
                for line in iter(process.stdout.readline, ""):
                    print(line, end="")
                    log.write(line)
                    log.flush()

                process.wait()

                if process.returncode != 0:
                    raise RuntimeError(
                        f"Training failed with return code {process.returncode}"
                    )

            print(f"\nTraining completed successfully!")
            print(f"Logs saved to: {log_file}")
            print(f"Checkpoints saved to: {self.output_dir / 'checkpoints'}")

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            print("\nTroubleshooting tips:")
            print("1. Ensure you're in the RETFound_MAE directory")
            print("2. Check that all dependencies are installed")
            print("3. Verify CUDA is available if using GPU")
            print("4. Try running with --num_gpus 0 for CPU training")
            raise

    def create_evaluation_script(self):
        """Create a script for model evaluation"""
        eval_script = """#!/usr/bin/env python
\"\"\"
Evaluation script for RETFound model
\"\"\"
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def evaluate_model(checkpoint_path, data_path, output_dir):
    # This would be implemented based on RETFound's evaluation code
    print(f"Evaluating model: {checkpoint_path}")
    print(f"Test data: {data_path}/test")
    # Add actual evaluation code here
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python evaluate_model.py <checkpoint_path> <data_path> <output_dir>")
        sys.exit(1)
    evaluate_model(sys.argv[1], sys.argv[2], sys.argv[3])
"""

        eval_path = self.output_dir / "evaluate_model.py"
        with open(eval_path, "w") as f:
            f.write(eval_script)

        # Make it executable
        eval_path.chmod(0o755)

        print(f"Evaluation script saved to: {eval_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RETFound for classification"
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset (contains train/val/test folders)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save outputs"
    )
    parser.add_argument(
        "--task_name", type=str, required=True, help="Name for this training task"
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="RETFound_mae",
        choices=["RETFound_mae", "RETFound_dinov2"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="RETFound_mae_natureCFP",
        choices=[
            "RETFound_mae_natureCFP",
            "RETFound_mae_natureOCT",
            "RETFound_mae_meh",
            "RETFound_mae_shanghai",
            "RETFound_dinov2_meh",
            "RETFound_dinov2_shanghai",
        ],
        help="Pretrained model to use",
    )
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument(
        "--base_lr", type=float, default=5e-3, help="Base learning rate"
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="Minimum learning rate"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Number of warmup epochs"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument(
        "--layer_decay", type=float, default=0.65, help="Layer-wise learning rate decay"
    )
    parser.add_argument("--drop_path", type=float, default=0.2, help="Drop path rate")

    # Augmentation arguments
    parser.add_argument(
        "--mixup", type=float, default=0.8, help="Mixup alpha, enabled if > 0"
    )
    parser.add_argument(
        "--cutmix", type=float, default=1.0, help="CutMix alpha, enabled if > 0"
    )
    parser.add_argument(
        "--reprob", type=float, default=0.25, help="Random erase probability"
    )

    # System arguments
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--master_port",
        type=int,
        default=48798,
        help="Master port for distributed training",
    )

    args = parser.parse_args()

    # Create configuration
    config = vars(args)

    # Validate paths
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise ValueError(f"Data path does not exist: {data_path}")

    # Check for required subdirectories
    for split in ["train", "val", "test"]:
        split_path = data_path / split
        if not split_path.exists():
            raise ValueError(f"Required directory not found: {split_path}")

    # Initialize trainer
    trainer = RETFoundTrainer(config)

    # Run training
    try:
        trainer.run_training()

        # Create evaluation script
        trainer.create_evaluation_script()

        print("\n" + "=" * 50)
        print("Training pipeline completed!")
        print(f"All outputs saved to: {args.output_dir}")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
