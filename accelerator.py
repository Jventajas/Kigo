"""Platform detection and Lightning-agnostic performance tuning.

Priority: CUDA → MPS → CPU
"""

import os
from dataclasses import dataclass
from typing import Literal

import torch

from config import Precision


@dataclass(frozen=True, slots=True)
class Platform:
    """Immutable descriptor for the compute platform we are running on."""

    device: torch.device
    name: Literal["cuda", "mps", "cpu"]

    @property
    def pin_memory(self) -> bool:
        """Whether ``DataLoader(pin_memory=True)`` helps."""
        return self.name == "cuda"

    @property
    def non_blocking(self) -> bool:
        """Whether ``tensor.to(device, non_blocking=True)`` is safe and helps."""
        return self.name in ("cuda", "mps")

    @property
    def num_workers(self) -> int:
        """Recommended ``DataLoader`` workers for this platform."""
        if self.name == "mps":
            return 0  # macOS multiprocessing is unreliable with MPS
        cores = os.cpu_count() or 1
        if self.name == "cpu":
            return min(4, cores // 2)
        # CUDA: scale with available host cores, but cap to avoid thrash.
        return min(8, max(2, cores // 4))

    @property
    def prefetch_factor(self) -> int | None:
        """Recommended prefetch factor (``None`` when workers == 0)."""
        return 2 if self.num_workers > 0 else None

    @property
    def persistent_workers(self) -> bool:
        """Whether to keep DataLoader worker processes alive."""
        return self.num_workers > 0

    @property
    def default_precision(self) -> Precision:
        """Best Lightning precision for this platform."""
        if self.name == "cuda":
            major, _ = torch.cuda.get_device_capability(self.device)
            return "bf16-mixed" if major >= 8 else "16-mixed"
        return "32-true"

    @property
    def should_compile(self) -> bool:
        """Whether ``torch.compile`` is worth enabling by default here."""
        return self.name == "cuda"

    def optimize(self) -> None:
        """Apply once-per-process PyTorch settings for this platform."""
        if self.name == "cuda":
            # TF32 speeds up matmuls on Ampere+ without hurting convergence.
            torch.backends.cuda.matmul.allow_tf32 = True
            # cuDNN benchmark searches for fastest conv algorithms.
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Explicit float32 matmul precision (redundant with allow_tf32 but
            # ensures consistency across PyTorch versions).
            major, _minor = torch.cuda.get_device_capability(self.device)
            if major >= 8:
                torch.set_float32_matmul_precision("high")
            else:
                # T4 (sm_75) has no FlashAttention; forbid the MATH SDPA backend so
                # attention uses the memory-efficient kernel instead of materializing
                # the full B·H·T·T score matrix.
                torch.backends.cuda.enable_math_sdp(False)

        elif self.name == "cpu":
            # Reserve cores for DataLoader workers so we don't oversubscribe.
            cores = os.cpu_count() or 1
            torch.set_num_threads(max(1, cores // (self.num_workers + 1)))

        elif self.name == "mps":
            # MPS does its own memory/queue management; no global knobs help yet.
            pass

    def reset_peak_memory(self) -> None:
        """Reset peak memory stats so the next query is fresh.

        No-op on platforms without a peak-memory API.
        """
        if self.name == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def memory_allocated(self) -> int:
        """Device memory in **bytes** since last reset (or process start).

        For CUDA this is the peak allocation. MPS has no peak API, so it
        returns the current driver allocation.
        """
        if self.name == "cuda":
            return torch.cuda.max_memory_allocated(self.device)
        if self.name == "mps":
            return torch.mps.driver_allocated_memory()
        return 0

    def total_memory(self) -> int:
        """Total device memory in **bytes** (0 where unknown)."""
        if self.name == "cuda":
            return torch.cuda.get_device_properties(self.device).total_memory
        return 0

    def device_count(self) -> int:
        """Number of devices this accelerator can train across (>= 1)."""
        if self.name == "cuda":
            return max(1, torch.cuda.device_count())
        return 1

    def synchronize(self) -> None:
        """Block until all queued device work has finished."""
        if self.name == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.name == "mps":
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        """Release cached device memory between allocations."""
        if self.name == "cuda":
            torch.cuda.empty_cache()
        elif self.name == "mps":
            torch.mps.empty_cache()

    @property
    def device_name(self) -> str:
        """Human-readable accelerator name for logging."""
        if self.name == "cuda":
            return torch.cuda.get_device_properties(self.device).name
        return self.name

    @staticmethod
    def is_oom_error(exc: RuntimeError) -> bool:
        """Whether a ``RuntimeError`` is an out-of-memory signal."""
        return "out of memory" in str(exc).lower()

    def __str__(self) -> str:
        return (
            f"Platform({self.name}, device={self.device!s}, "
            f"workers={self.num_workers})"
        )


def get_platform(prefer: str | None = None) -> Platform:
    """Detect and return the best available platform.

    Args:
        prefer: If given (``'cuda'``, ``'mps'``, ``'cpu'``), check that platform
            first.  If unavailable, fall back to the default priority order.

    Priority:
        1. NVIDIA CUDA
        2. Apple MPS
        3. CPU
    """
    prefer = (prefer or "").lower().strip()
    order = ["cuda", "mps", "cpu"]
    if prefer in order:
        # Bump up the preferred platform in the priority list.
        order = [prefer] + [name for name in order if name != prefer]

    for name in order:
        if name == "cuda" and torch.cuda.is_available():
            # Reduce allocator fragmentation to fit larger batches (read lazily on first alloc).
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            return Platform(device=torch.device("cuda", 0), name="cuda")

        if name == "mps" and torch.backends.mps.is_available():
            return Platform(device=torch.device("mps"), name="mps")

        if name == "cpu":
            return Platform(device=torch.device("cpu"), name="cpu")

    return Platform(device=torch.device("cpu"), name="cpu")
