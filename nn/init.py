"""Weight initialization utilities."""

import torch.nn as nn

from nn.norm import RMSNorm


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """Apply nanoGPT-style weight initialization.

    - Linear & Embedding weights: normal(mean=0, std=std)
    - RMSNorm weight: ones
    - Linear bias (defensive): zeros

    Residual output projections get additional scaling *after* this is applied,
    at the model level via ``named_parameters()``.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)
