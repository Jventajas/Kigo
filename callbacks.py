"""PyTorch Lightning callbacks for Kigo training."""

import time
from concurrent.futures import Future
from typing import Any, TYPE_CHECKING, cast

import torch
import wandb
from huggingface_hub import CommitInfo, HfApi
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

from accelerator import Platform
from config import Config

if TYPE_CHECKING:
    from nn.lightning_module import KigoLightningModule

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

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_time = time.time()
        self._log_start_step = 0
        cast("KigoLightningModule", pl_module).tokens_processed = 0

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        module = cast("KigoLightningModule", pl_module)
        step = trainer.global_step
        module.tokens_processed = step * self.tokens_per_step

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

        module.log("train/tokens_per_sec", tokens_sec, on_step=True, on_epoch=False)
        module.log("train/lr", lr, on_step=True, on_epoch=False)
        module.log("train/memory_gb", memory_gb, on_step=True, on_epoch=False)

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
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        step = trainer.global_step
        if step == 0 or step % self.sample_interval != 0:
            return

        samples = cast("KigoLightningModule", pl_module).generate_samples(
            prompts=DEFAULT_PROMPTS,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
        )

        # Log the samples to W&B if a logger is attached (not to the console).
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                table = wandb.Table(columns=["step", "prompt", "output"])
                for sample in samples:
                    table.add_data(step, sample["prompt"], sample["output"])
                logger.experiment.log({"samples": table}, step=step)


class HubModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that mirrors each saved checkpoint to a Hugging Face repo.

    CommitScheduler is the wrong tool here: it is append-only (overwriting a file
    "might corrupt your repository") and uploads via binary IO buffers, which Xet
    Storage rejects. Instead we push the checkpoint directory by path right after
    Lightning writes one, in the background via the SDK's documented `run_as_future`
    executor -- the same pattern as Lightning's own `LitModelCheckpoint`.
    """

    def __init__(self, repo_id: str | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.repo_id = repo_id
        self._api = HfApi()
        self._pending: Future[CommitInfo] | None = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if self.repo_id is not None and trainer.is_global_zero:
            self._api.create_repo(self.repo_id, repo_type="model", private=True, exist_ok=True)

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        if self.repo_id is None or not trainer.is_global_zero or self.dirpath is None:
            return
        self._wait()  # surface the prior upload's outcome before queuing the next
        self._pending = self._api.upload_folder(
            repo_id=self.repo_id,
            repo_type="model",
            folder_path=self.dirpath,
            run_as_future=True,
        )

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self._wait()

    def _wait(self) -> None:
        if self._pending is None:
            return
        try:
            self._pending.result()
        except Exception as exc:  # keep training alive; CommitScheduler would hide this
            print(f"[hub] checkpoint upload failed: {exc!r}")
        self._pending = None
