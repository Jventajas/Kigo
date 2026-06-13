"""PyTorch Lightning module for Kigo training and inference."""

import math
from typing import Any, Literal

import tiktoken
import torch
from lightning.pytorch import LightningModule

from accelerator import Platform
from config import Config
from eval import DEFAULT_PROMPTS
from nn.model import GPT

Stage = Literal["fit", "validate", "test", "predict"]


class KigoLightningModule(LightningModule):
    """A ``LightningModule`` wrapping the Kigo GPT model."""

    def __init__(self, config: Config, platform: Platform, wandb_run_id: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.platform = platform
        self.model = GPT(config)
        self.wandb_run_id = wandb_run_id

        # Runtime state persisted across checkpoints.
        self.best_val_loss = float("inf")
        self.tokens_processed = 0

        # Validation accumulators (reset each epoch).
        self._val_loss_sum = 0.0
        self._val_batches = 0

    def setup(self, stage: Stage | None = None) -> None:
        """Apply platform-specific PyTorch optimizations once."""
        self.platform.optimize()

    def transfer_batch_to_device(
        self, batch: torch.Tensor, device: torch.device, dataloader_idx: int
    ) -> torch.Tensor:
        # Keep CPU→GPU transfer at 2 bytes/token, then cast to int64 for Embedding.
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
        self.log("val/loss", avg_loss, prog_bar=True)
        self.log("val/perplexity", perplexity, prog_bar=True)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        self._val_loss_sum = 0.0
        self._val_batches = 0

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

        min_lr_ratio = self.config.min_lr / self.config.learning_rate

        def lr_lambda(step: int) -> float:
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            progress = (step - self.config.warmup_steps) / (
                    self.config.max_steps - self.config.warmup_steps
            )
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["tokens_processed"] = self.tokens_processed
        checkpoint["wandb_run_id"] = self.wandb_run_id

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.tokens_processed = checkpoint.get("tokens_processed", 0)
        self.wandb_run_id = checkpoint.get("wandb_run_id")

    @torch.no_grad()
    def generate_samples(
        self,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        prompts: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Generate text samples from the model."""
        if prompts is None:
            prompts = DEFAULT_PROMPTS

        tokenizer = tiktoken.get_encoding("gpt2")
        self.eval()
        results: list[dict[str, str]] = []

        for prompt_text in prompts:
            token_ids = tokenizer.encode(prompt_text, allowed_special="all")
            tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            generated = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            output_text = tokenizer.decode(generated[0].tolist())
            results.append({"prompt": prompt_text, "output": output_text})

        self.train()
        return results
