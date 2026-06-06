"""Transformer layers: attention, feed-forward, and residual block."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.embeddings import RoPE, apply_rotary_emb
from nn.norm import RMSNorm


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and PyTorch SDPA."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float,
        bias: bool = False,
        rope_max_length: int = 2048,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout_p = dropout
        self.output_dropout = nn.Dropout(dropout)

        self.rope = RoPE(
            d_head=self.d_head,
            max_length=rope_max_length,
            base=rope_base,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()

        # (B, T, 3 * d_model) -> 3 x (B, T, d_model)
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=-1)

        # (B, T, d_model) -> (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(T)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # SDPA with causal mask (optimized FlashAttention kernel when available)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )

        # (B, n_head, T, d_head) -> (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)

        y = self.output_dropout(self.out_proj(y))
        return y


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        return self.dropout(self.down_proj(hidden))


class TransformerBlock(nn.Module):
    """Pre-LN transformer block with residual connections."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float,
        bias: bool = False,
        rope_max_length: int = 2048,
        rope_base: float = 10000.0,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            bias=bias,
            rope_max_length=rope_max_length,
            rope_base=rope_base,
        )
        self.norm2 = RMSNorm(d_model)
        self.sff = SwiGLUFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.sff(self.norm2(x))
        return x
