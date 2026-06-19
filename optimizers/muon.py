"""
Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

Use `split_params` to split a model into Muon (2D hidden weights) and AdamW
(embeddings, head, norms, biases) groups.
"""
import math
from typing import Iterable, Any, Callable, overload

import torch
from torch import Tensor


def newton_schulz_orthogonalization(matrix: Tensor, steps: int = 5, eps: float = 1e-8) -> Tensor:
    """Newton-Schulz orthogonalization: drives singular values toward 1 via matmuls (no SVD)."""

    a, b, c = 3.4445, -4.7750, 2.0315
    X = matrix / (matrix.norm() + eps)

    transposed = X.shape[1] < X.shape[0]
    if transposed:
        X = X.T

    for _ in range(steps):
        gram = X @ X.T
        X = a * X + (b * gram + c * (gram @ gram)) @ X

    return X.T if transposed else X


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
    """Muon for 2D hidden weights, AdamW for everything else, in one optimizer."""

    def __init__(
            self,
            muon_params: Iterable[Tensor],
            adamw_params: Iterable[Tensor],
            muon_lr: float = 0.02,
            adamw_lr: float = 3e-4,
            momentum: float = 0.95,
            nesterov: bool = True,
            ns_steps: int = 5,
            betas: tuple[float, float] = (0.9, 0.95),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            betas=betas, eps=eps, weight_decay=weight_decay,
        )
        groups = [
            dict(params=list(muon_params), use_muon=True, lr=muon_lr),
            dict(params=list(adamw_params), use_muon=False, lr=adamw_lr),
        ]
        super().__init__(groups, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """One optimization step. Calls `closure` if given to populate gradients."""

        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)

            buffer = state["momentum_buffer"]
            buffer.mul_(momentum).add_(p.grad)
            update = p.grad.add(buffer, alpha=momentum) if nesterov else buffer

            update = newton_schulz_orthogonalization(update, steps=ns_steps)
            # aspect-ratio correction: keeps update norm consistent across non-square shapes
            scale = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
            p.add_(update, alpha=-lr * scale)

    def _adamw_step(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if weight_decay != 0 and p.ndim >= 2:
                p.mul_(1 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

