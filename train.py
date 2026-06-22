#!/usr/bin/env python3
"""Main training loop for Kigo using PyTorch Lightning."""

import argparse
import os
from pathlib import Path
from typing import cast

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger

from accelerator import get_platform
from callbacks import (
    HubModelCheckpoint,
    SampleGenerationCallback,
    ThroughputMemoryCallback,
)
from config import load_config
from data_module import KigoDataModule
from nn.lightning_module import KigoLightningModule
from nn.model import GPT


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
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HF model repo (e.g. user/kigo) to sync checkpoints with; omit for local-only.",
    )
    parser.add_argument(
        "--auto-batch",
        action="store_true",
        help="Probe the device for the largest micro-batch and override config.batch_size.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed, workers=True)
    platform = get_platform(prefer=config.device)
    num_devices = platform.device_count()
    # Under multi-device DDP the script re-runs per rank; only rank 0 touches the Hub.
    is_rank_zero = os.environ.get("LOCAL_RANK", "0") == "0"

    # Auto-tune the micro-batch in a subprocess so its CUDA context -- and the
    # OOMs it deliberately triggers -- are fully torn down before we train.
    if args.auto_batch and platform.name == "cuda":
        import subprocess
        import sys
        import tempfile

        probe = Path(__file__).parent / "scripts" / "find_batch_size.py"
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            emit_path = f.name
        subprocess.run(
            [sys.executable, str(probe), "--config", args.config, "--emit", emit_path],
            check=True,
        )
        config.batch_size = int(Path(emit_path).read_text().strip())
        os.unlink(emit_path)
        print(f"Auto-tuned batch_size = {config.batch_size}")

    # Pull the latest checkpoints from the Hub so a run can resume on any machine.
    if args.hf_repo and is_rank_zero:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import RepositoryNotFoundError

        try:
            snapshot_download(
                repo_id=args.hf_repo, repo_type="model", local_dir=args.checkpoint_dir
            )
        except RepositoryNotFoundError:
            print(f"No Hub repo {args.hf_repo} yet; using local checkpoints if present.")

    # Resume detection
    last_ckpt = Path(args.checkpoint_dir) / "last.ckpt"

    # Data
    datamodule = KigoDataModule(config, platform, data_dir=Path(args.data_dir))

    # Model
    model = KigoLightningModule(config, platform)
    if platform.should_compile:
        # Compile the inner GPT, not the LightningModule: compiling the whole
        # module makes Dynamo trace Lightning's self.log() and crash.
        model.model = cast(GPT, torch.compile(model.model))

    # Logger
    logger: WandbLogger | None = None
    if not args.no_wandb:
        import secrets

        # Persist the W&B run id next to the checkpoints so a resumed run continues
        # the same run; a clean start gets a new one. Only rank 0 logs, so only it
        # owns the id (non-zero ranks pass None and Lightning keeps them silent).
        run_id = None
        if is_rank_zero:
            run_id_file = Path(args.checkpoint_dir) / "wandb_run_id"
            if last_ckpt.exists() and run_id_file.exists():
                run_id = run_id_file.read_text().strip()
            else:
                run_id = secrets.token_hex(4)
                run_id_file.parent.mkdir(parents=True, exist_ok=True)
                run_id_file.write_text(run_id)

        logger = WandbLogger(
            project=config.wandb_project,
            name=config.wandb_run_name,
            id=run_id,
            resume="allow",
            config={k: str(v) if isinstance(v, Path) else v for k, v in vars(config).items()},
        )

    # Callbacks
    # Effective batch = num_devices * batch_size * accumulation, so divide by the
    # device count to keep global_batch_size fixed regardless of how many we use.
    accumulation_steps = max(1, config.global_batch_size // (config.batch_size * num_devices))

    checkpoint_callback = HubModelCheckpoint(
        repo_id=args.hf_repo,
        dirpath=Path(args.checkpoint_dir),
        filename="checkpoint_step_{step:05d}",
        auto_insert_metric_name=False,
        every_n_train_steps=config.checkpoint_interval,
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
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
        devices="auto",
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
