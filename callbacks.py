"""PyTorch Lightning callbacks for Kigo training."""

import time
from pathlib import Path
from typing import Any

import torch
import wandb
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from accelerator import Platform
from config import Config

DEFAULT_PROMPTS = [
    "The capital of France is",
    "The quick brown fox",
    "Once upon a time",
    "def factorial(n):",
    "In Spanish, 'hello' is",
    "The theory of relativity states that",
    "To make a cake, you need",
    "The Great Depression began in",
    "import numpy as np\n",
    "The mitochondria is the powerhouse of",
]


class ThroughputMemoryCallback(Callback):
    """Log throughput (tok/s), learning rate, and device memory at intervals."""

    def __init__(self, cfg: Config, platform: Platform) -> None:
        super().__init__()
        self.cfg = cfg
        self.platform = platform
        self.tokens_per_step = cfg.global_batch_size * cfg.block_size
        self.log_interval = cfg.log_interval
        self._log_time = time.time()
        self._log_start_step = 0

    def on_train_start(self, trainer: Trainer, pl_module: Any) -> None:
        self._log_time = time.time()
        self._log_start_step = 0
        pl_module.tokens_processed = 0

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        step = trainer.global_step
        pl_module.tokens_processed = step * self.tokens_per_step

        if step == 0 or step % self.log_interval != 0:
            return

        # Synchronize for accurate timing where a sync API exists.
        if self.platform.name == "cuda":
            torch.cuda.synchronize(self.platform.device)
        elif self.platform.name == "mps":
            torch.mps.synchronize()

        now = time.time()
        steps_elapsed = step - self._log_start_step
        dt = now - self._log_time
        tokens_sec = (steps_elapsed * self.tokens_per_step) / dt if dt > 0 else 0.0

        lr = trainer.optimizers[0].param_groups[0]["lr"]
        memory_gb = self.platform.memory_allocated() / 1e9
        self.platform.reset_peak_memory()

        pl_module.log("train/tokens_per_sec", tokens_sec, on_step=True, on_epoch=False)
        pl_module.log("train/lr", lr, on_step=True, on_epoch=False)
        pl_module.log("train/memory_gb", memory_gb, on_step=True, on_epoch=False)

        self._log_time = now
        self._log_start_step = step


class SampleGenerationCallback(Callback):
    """Generate and log text samples at configured intervals."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.sample_interval = config.sample_interval

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        step = trainer.global_step
        if step == 0 or step % self.sample_interval != 0:
            return

        samples = pl_module.generate_samples(
            prompts=DEFAULT_PROMPTS,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
        )

        rank_zero_info("  → Generating samples...")
        for sample in samples[:3]:
            rank_zero_info(f"    PROMPT: {sample['prompt']!r}")
            rank_zero_info(f"    OUTPUT: {sample['output'][:200]!r}")
            rank_zero_info("")

        # Log the full set to W&B if a logger is attached.
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                table = wandb.Table(columns=["step", "prompt", "output"])
                for sample in samples:
                    table.add_data(step, sample["prompt"], sample["output"])
                logger.experiment.log({"samples": table}, step=step)
