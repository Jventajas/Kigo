#!/usr/bin/env python3
"""Print the largest micro-batch that fits on this GPU.

Uses Lightning's built-in batch-size finder (a power search of real training
steps). The micro-batch is per-device, so this probes a single GPU; run it once
per (model config, GPU type), then put the value in the config's batch_size or
pass it to train.py via --batch-size.

Usage:
    python scripts/find_batch.py --config config/kigo-162m.yaml --data-dir DATA_DIR
"""

import argparse
from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.tuner.tuning import Tuner

from accelerator import get_platform
from config import load_config
from data_module import KigoDataModule
from nn.lightning_module import KigoLightningModule


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the largest micro-batch that fits.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to the data directory containing the train/ split.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    platform = get_platform(prefer=config.device)
    datamodule = KigoDataModule(config, platform, data_dir=Path(args.data_dir))
    model = KigoLightningModule(config, platform)

    # Single device, no logging/checkpointing: this only measures what fits.
    trainer = Trainer(
        accelerator=platform.name,
        devices=1,
        precision=config.dtype or platform.default_precision,
        logger=False,
        enable_checkpointing=False,
    )
    suggested = Tuner(trainer).scale_batch_size(model, datamodule=datamodule, mode="binsearch")
    print(f"Suggested batch_size = {suggested} (back off ~10% for val/sampling/compile headroom)")


if __name__ == "__main__":
    main()
