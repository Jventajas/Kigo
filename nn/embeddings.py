"""Rotary Position Embedding (RoPE)."""

import torch
import torch.nn as nn


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor x.

    Args:
        x: Tensor of shape (B, n_head, T, d_head).
        cos: Cosine cache of shape (1, 1, T, d_head // 2).
        sin: Sine cache of shape (1, 1, T, d_head // 2).

    Returns:
        Rotated tensor of shape (B, n_head, T, d_head).
    """
    # Extract the two coordinates of each 2D rotation plane: (B, n_head, T, d_head//2)
    x1 = x[..., ::2]  # even indices: first axis of each pair
    x2 = x[..., 1::2]  # odd indices: second axis of each pair

    # Rotate each pair: [y1, y2] = [cos -sin; sin cos] @ [x1, x2]
    y1 = x1 * cos - x2 * sin          # (B, n_head, T, d_head//2)
    y2 = x1 * sin + x2 * cos          # (B, n_head, T, d_head//2)

    # Stack into (B, n_head, T, d_head//2, 2) — last dim is [y1, y2] per pair
    paired = torch.stack([y1, y2], dim=-1)

    # Flatten last two dims to interleave back to original layout: (B, n_head, T, d_head)
    x_rotated = paired.flatten(-2)

    return x_rotated


class RoPE(nn.Module):
    """Rotary Position Embedding cache.

    Pre-computes cosine and sine tables for all positions up to max_length.
    The actual rotation is applied via :func:`apply_rotary_emb`.
    """

    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, d_head: int, max_length: int = 2048, base: float = 10000.0) -> None:
        super().__init__()

        # Rotation rate per dimension pair (radians per position step): (d_head // 2,)
        inv_freq = torch.tensor(1.0) / (base ** (torch.arange(0, d_head, 2).float() / d_head))

        # Token positions [0, 1, ..., max_length-1] in the sequence: (max_length,)
        pos = torch.arange(max_length, dtype=torch.float32)

        # Full rotation angles for every (position, dimension) pair: (max_length, d_head // 2)
        angles = torch.outer(pos, inv_freq)

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin caches sliced to seq_len.

        Returns:
            cos: (1, 1, seq_len, d_head // 2)
            sin: (1, 1, seq_len, d_head // 2)
        """
        # Add singleton batch and head dims so cos/sin broadcast over (B, n_head, T, d_head//2)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return cos, sin
