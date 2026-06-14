"""GPT model assembly with modern architectural improvements."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from nn.init import init_weights
from nn.layers import TransformerBlock
from nn.norm import RMSNorm


class GPT(nn.Module):
    """~162M-parameter GPT with RoPE, SwiGLU, RMSNorm, and SDPA.

    Follows the Config dataclass for all hyperparameters.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model=config.d_model,
                n_head=config.n_head,
                d_ff=config.d_ff,
                dropout=config.dropout,
                bias=config.bias,
                rope_max_length=config.rope_max_length,
                rope_base=config.rope_base,
                layer_idx=i,
            )
            for i in range(config.n_layer)
        )

        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=config.bias)

        # Weight tying: LM head shares weights with token embedding.
        # This is standard for GPT-style models and keeps the parameter count
        # at ~162M with the cl100k_base vocab.
        self.token_embedding.weight = self.lm_head.weight

        # Base initialization
        self.apply(lambda m: init_weights(m, std=0.02))

        # Scale residual output projections for depth stability (nanoGPT style)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            idx: Token indices of shape (B, T).
            targets: Target indices of shape (B, T) for loss computation.

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross-entropy loss if targets provided, else None.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        x = self.token_embedding(idx)  # (B, T, d_model)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            idx: Initial token sequence of shape (B, T).
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature. Lower = more deterministic.
            top_k: If set, restrict sampling to the top-k logits.
            eos_token_id: If set, stop generation per-row as soon as a
                sequence samples this token. Already-finished rows are
                padded with ``eos_token_id`` so the output stays rectangular.

        Returns:
            Generated sequence of shape (B, T + n) where n <= max_new_tokens.
        """
        B = idx.size(0)
        finished = torch.zeros((B, 1), dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            if finished.all():
                break

            # Crop to block_size context window
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[
                :, -self.config.block_size :
            ]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Force already-finished rows to keep emitting EOS so the tensor
            # stays rectangular and downstream code can mask them out.
            if eos_token_id is not None:
                idx_next = torch.where(
                    finished,
                    torch.tensor([[eos_token_id]], device=idx.device),
                    idx_next,
                )

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None:
                finished = finished | idx_next.eq(eos_token_id)

        return idx

        return idx

    @property
    def num_params(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())
