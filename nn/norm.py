"""Root Mean Square Layer Normalization (RMSNorm)."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm normalizes by the root mean square of the input."""

    def __init__(self, ndim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, ndim)
        norm = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return x / (norm + self.eps) * self.weight
