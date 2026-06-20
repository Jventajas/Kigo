"""Platform detection and Lightning-agnostic performance tuning.

Priority: CUDA → TPU → MPS → CPU
"""

import os
from dataclasses import dataclass
from typing import Literal

import torch

from config import Precision

try:
    import torch_xla.core.xla_model as xm  # type: ignore[reportMissingImports]
except ImportError:
    xm = None


@dataclass(frozen=True, slots=True)
class Platform:
    """Immutable descriptor for the compute platform we are running on."""

    device: torch.device
    name: Literal["cuda", "tpu", "mps", "cpu"]

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
        # CUDA / TPU: scale with available host cores, but cap to avoid thrash.
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
        if self.name == "tpu":
            return "bf16-mixed"
        return "32-true"

    @property
    def should_compile(self) -> bool:
        """Whether ``torch.compile`` is worth enabling by default here."""
        return self.name in ("cuda", "tpu")

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

        elif self.name == "cpu":
            # Reserve cores for DataLoader workers so we don't oversubscribe.
            cores = os.cpu_count() or 1
            torch.set_num_threads(max(1, cores // (self.num_workers + 1)))

        elif self.name == "mps":
            # MPS does its own memory/queue management; no global knobs help yet.
            pass

        elif self.name == "tpu" and xm is not None:
            # XLA handles its own thread pool; leave PyTorch threads alone.
            pass

    def reset_peak_memory(self) -> None:
        """Reset peak memory stats so the next query is fresh.

        No-op on platforms without a peak-memory API.
        """
        if self.name == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def memory_allocated(self) -> int:
        """Device memory in **bytes** since last reset (or process start).

        For CUDA this is the peak allocation. For MPS and TPU no peak API
        exists, so this returns the current driver allocation.
        """
        if self.name == "cuda":
            return torch.cuda.max_memory_allocated(self.device)
        if self.name == "mps":
            return torch.mps.driver_allocated_memory()
        if self.name == "tpu" and xm is not None:
            try:
                info = xm.get_memory_info(self.device)
                return info.get("bytes_used", 0)
            except Exception:
                return 0
        return 0

    def __str__(self) -> str:
        return (
            f"Platform({self.name}, device={self.device!s}, "
            f"workers={self.num_workers})"
        )


def get_platform(prefer: str | None = None) -> Platform:
    """Detect and return the best available platform.

    Args:
        prefer: If given (``'cuda'``, ``'tpu'``, ``'mps'``, ``'cpu'``), check
            that platform first.  If unavailable, fall back to the default
            priority order.

    Priority:
        1. NVIDIA CUDA
        2. Google TPU (via torch_xla)
        3. Apple MPS
        4. CPU
    """
    prefer = (prefer or "").lower().strip()
    order = ["cuda", "tpu", "mps", "cpu"]
    if prefer in order:
        # Bump up the preferred platform in the priority list.
        order = [prefer] + [name for name in order if name != prefer]

    for name in order:
        if name == "cuda" and torch.cuda.is_available():
            # Reduce allocator fragmentation to fit larger batches (read lazily on first alloc).
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            return Platform(device=torch.device("cuda", 0), name="cuda")

        if name == "tpu" and xm is not None:
            try:
                tpu_dev = xm.xla_device()
                if xm.xla_device_hw(tpu_dev) == "TPU":
                    return Platform(device=tpu_dev, name="tpu")
            except Exception:
                pass

        if name == "mps" and torch.backends.mps.is_available():
            return Platform(device=torch.device("mps"), name="mps")

        if name == "cpu":
            return Platform(device=torch.device("cpu"), name="cpu")

    return Platform(device=torch.device("cpu"), name="cpu")
