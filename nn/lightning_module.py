"""PyTorch Lightning module for Kigo training and inference."""

import math
from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from accelerator import Platform
from config import Config
from nn.model import GPT
from optimizers import Muon, split_params
from tokenizer import build_tokenizer


class KigoLightningModule(LightningModule):
    """A ``LightningModule`` wrapping the Kigo GPT model."""

    def __init__(self, config: Config, platform: Platform) -> None:
        super().__init__()
        self.config = config
        self.platform = platform
        self.model = GPT(config)
        self.tokenizer = build_tokenizer(config)

        # Runtime state persisted across checkpoints.
        self.best_val_loss = float("inf")
        self.tokens_processed = 0

        # Validation accumulators (reset each epoch).
        self._val_loss_sum = 0.0
        self._val_batches = 0

    def setup(self, stage: str) -> None:
        """Apply platform-specific PyTorch optimizations once."""
        self.platform.optimize()

    def transfer_batch_to_device(
        self, batch: torch.Tensor, device: torch.device, dataloader_idx: int
    ) -> torch.Tensor:
        # Transfer uint32 tokens to GPU, then cast to int64 for Embedding.
        return batch.to(device, non_blocking=self.platform.non_blocking).long()

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.model(idx, targets)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        _, loss = self(inputs, targets)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        _, loss = self(inputs, targets)
        self._val_loss_sum += loss.item()
        self._val_batches += 1
        return loss

    def on_validation_epoch_end(self) -> None:
        if self._val_batches == 0:
            return
        avg_loss = self._val_loss_sum / self._val_batches
        perplexity = torch.exp(torch.tensor(avg_loss))
        self.log("val/loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", perplexity, prog_bar=True, sync_dist=True)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        self._val_loss_sum = 0.0
        self._val_batches = 0

    def configure_optimizers(self) -> OptimizerLRScheduler:
        muon_params, adamw_params = split_params(self.model)
        optimizer = Muon(
            muon_params,
            adamw_params,
            muon_lr=self.config.muon_lr,
            adamw_lr=self.config.adamw_lr,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

        min_lr_ratio = self.config.min_lr / self.config.adamw_lr
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.config.warmup_steps

        def lr_lambda(step: int) -> float | int:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Strip torch.compile's "_orig_mod." prefix so checkpoints stay portable
        # across compiled/uncompiled platforms (e.g. resuming a GPU run on CPU).
        checkpoint["state_dict"] = {
            key.replace("_orig_mod.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["tokens_processed"] = self.tokens_processed

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Match the checkpoint's keys to this process: drop any "_orig_mod."
        # prefix, then re-add it only if our model is currently compiled.
        compiled = hasattr(self.model, "_orig_mod")
        remapped: dict[str, Any] = {}
        for key, value in checkpoint["state_dict"].items():
            key = key.replace("_orig_mod.", "")
            if compiled:
                key = key.replace("model.", "model._orig_mod.", 1)
            remapped[key] = value
        checkpoint["state_dict"] = remapped

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.tokens_processed = checkpoint.get("tokens_processed", 0)

    @torch.no_grad()
    def generate_samples(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> list[dict[str, str]]:
        """Generate text samples from the model."""
        self.eval()
        results: list[dict[str, str]] = []

        for prompt_text in prompts:
            token_ids = self.tokenizer.encode(prompt_text)
            tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            generated = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=self.tokenizer.eos_id,
            )
            ids = [i for i in generated[0].tolist() if i != self.tokenizer.eos_id]
            output_text = self.tokenizer.decode(ids)
            results.append({"prompt": prompt_text, "output": output_text})

        self.train()
        return results
