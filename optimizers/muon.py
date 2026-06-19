"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

Muon is applied only to 2D hidden weight matrices. Embeddings, the LM head,
norms, and any 1D/scalar params should stay on AdamW (see `split_params`).
"""
import math
from typing import Iterable, Any

import torch
from torch import Tensor


def zeropower_via_newtonschulz(grad: Tensor, steps: int = 5, epsilon: float = 1e-8) -> Tensor:
    """Return an approximate orthogonalization of matrix G via Newton-Schulz.

    The Newton-Schulz iteration drives the singular values of G toward 1 while
    keeping its singular vectors, i.e. it approximates U @ V.T from G = U S V.T,
    using only matmuls (no SVD) so it runs fast on GPU in bf16.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    normalized_grad = grad / (grad.norm() + epsilon)

    height, width = normalized_grad.shape

    if width < height:
        normalized_grad = normalized_grad.T

    for _ in range(steps):
        normalized_grad_squared = normalized_grad @ normalized_grad.T
        normalized_grad = a * normalized_grad + \
            (b * normalized_grad_squared + c * (normalized_grad_squared @ normalized_grad_squared)) @ normalized_grad

    if width < height:
        normalized_grad = normalized_grad.T

    return normalized_grad


def split_params(model: torch.nn.Module) -> tuple[list[Tensor], list[Tensor]]:
    """Split model parameters into (muon_params, adamw_params)."""

    excluded = ('token_embedding.weight', 'lm_head.weight')
    muon_parameters = []
    adamw_parameters = []

    for name, params in model.named_parameters():
        if params.ndim == 2 and not name.endswith(excluded):
            muon_parameters.append(params)
        else:
            adamw_parameters.append(params)
    return muon_parameters, adamw_parameters


class Muon(torch.optim.Optimizer):
    """Muon for 2D weight matrices.

    Args:
        params: iterable of 2D parameters (use `split_params` to select them).
        lr: learning rate.
        momentum: SGD-style momentum coefficient (e.g. 0.95).
        nesterov: whether to use Nesterov-style momentum.
        ns_steps: Newton-Schulz iteration count (e.g. 5).
    """

    def __init__(
            self,
            params: Iterable[Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, Tensor]],
            lr: float | int = 0.02,
            momentum: float= 0.95,
            nesterov: bool = True,
            ns_steps: int = 5
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """One optimization step."""

        for pg in self.param_groups:

            params = pg['params']
            lr = pg['lr']
            momentum = pg['momentum']
            nesterov = pg['nesterov']
            ns_steps = pg['ns_steps']

            for p in params:

                if p.grad is None:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(p.grad)

                if nesterov:
                    update = buffer * momentum + p.grad
                else:
                    update = buffer

                update = zeropower_via_newtonschulz(update, steps=ns_steps)
                height, width = update.shape
                update *= math.sqrt(max(height, width))

                p -= lr * update

