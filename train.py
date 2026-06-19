#!/usr/bin/env python3
"""Main training loop for Kigo using PyTorch Lightning."""

import argparse
from pathlib import Path
from typing import cast

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from accelerator import get_platform
from callbacks import (
    SampleGenerationCallback,
    ThroughputMemoryCallback,
)
from config import load_config
from data_module import KigoDataModule
from nn.lightning_module import KigoLightningModule


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Kigo with PyTorch Lightning.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to the data directory containing train/, val/, and test/ splits.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Path to the directory where checkpoints are saved and resumed from.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed, workers=True)
    platform = get_platform(prefer=config.device)

    # Resume detection
    last_ckpt = Path(args.checkpoint_dir) / "last.ckpt"

    # Data
    datamodule = KigoDataModule(config, platform, data_dir=Path(args.data_dir))

    # Model
    model = KigoLightningModule(config, platform)
    if platform.should_compile:
        if platform.name == "tpu":
            model = cast(KigoLightningModule, torch.compile(model, backend="openxla"))
        else:
            model = cast(KigoLightningModule, torch.compile(model))

    # Logger
    logger: WandbLogger | None = None
    if not args.no_wandb:
        logger = WandbLogger(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={k: str(v) if isinstance(v, Path) else v for k, v in vars(config).items()},
        )

    # Callbacks
    accumulation_steps = config.global_batch_size // config.batch_size

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.checkpoint_dir),
        filename="checkpoint_step_{step:05d}",
        auto_insert_metric_name=False,
        every_n_train_steps=config.checkpoint_interval,
        save_last=True,
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        save_weights_only=False,
        enable_version_counter=False,
    )

    callbacks: list[Callback] = [
        checkpoint_callback,
        ThroughputMemoryCallback(config, platform),
        SampleGenerationCallback(config),
    ]

    # Trainer
    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator=platform.name,
        devices=1,
        precision=config.dtype or platform.default_precision,
        accumulate_grad_batches=accumulation_steps,
        gradient_clip_val=config.grad_clip,
        val_check_interval=config.eval_interval,
        check_val_every_n_epoch=1,
        log_every_n_steps=config.log_interval,
        enable_progress_bar=True,
        logger=logger,
        callbacks=callbacks,
    )

    print(f"Platform : {platform}")
    print(f"Model    : {config.n_layer} layers, {config.d_model} dim, {config.n_head} heads")
    print(f"Accumulation : {accumulation_steps} micro-steps")
    print(f"Tokens/step  : {config.global_batch_size * config.block_size:,}")

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=last_ckpt if last_ckpt.exists() else None,
    )


if __name__ == "__main__":
    main()
