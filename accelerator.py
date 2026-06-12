"""Hardware platform detection and throughput optimization.

Automatically selects the best available accelerator and configures PyTorch
for maximum performance on that platform.  Callers never touch raw device
strings or AMP configuration — they ask the :class:`Platform` object.

Priority: CUDA → TPU → MPS → CPU
"""

import os
from dataclasses import dataclass
from typing import Literal

import torch

try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    HAS_XLA = True
except ImportError:
    HAS_XLA = False


@dataclass(frozen=True, slots=True)
class Platform:
    """Immutable descriptor for the compute platform we are running on.

    Usage::

        plat = get_platform()
        model = GPT(cfg).to(plat.device)
        if plat.supports_grad_scaler:
            scaler = torch.cuda.amp.GradScaler()
        loader = get_dataloader(..., pin_memory=plat.pin_memory,
                                num_workers=plat.num_workers)
    """

    device: torch.device
    name: Literal["cuda", "tpu", "mps", "cpu"]


    @property
    def supports_amp(self) -> bool:
        """Whether ``torch.autocast`` is beneficial and safe on this platform."""
        return self.name in ("cuda", "tpu")

    @property
    def supports_grad_scaler(self) -> bool:
        """Whether to apply gradient scaling.

        Only pre-Ampere CUDA needs it — float16 underflows, bfloat16 doesn't.
        """
        if self.name == "cuda":
            major, _minor = torch.cuda.get_device_capability(self.device)
            return major < 8
        return False

    @property
    def amp_dtype(self) -> torch.dtype:
        """Recommended dtype to pass to ``torch.autocast``.

        * CUDA Ampere+ (sm80+) → ``bfloat16`` (wider dynamic range, same speed).
        * CUDA older          → ``float16``.
        * MPS / CPU           → ``float32`` (autocast disabled).
        """
        if self.name == "cuda":
            major, _minor = torch.cuda.get_device_capability(self.device)
            return torch.bfloat16 if major >= 8 else torch.float16
        if self.name == "tpu":
            return torch.bfloat16
        return torch.float32

    @property
    def supports_compile(self) -> bool:
        """Whether ``torch.compile`` is expected to improve throughput.

        CUDA and CPU use the default Inductor backend. TPU uses the XLA/OpenXLA
        backend. MPS is not supported by ``torch.compile`` as of PyTorch 2.x.
        """
        return self.name in ("cuda", "cpu", "tpu")

    def compile_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Compile *model* for this platform's backend."""
        if self.name == "tpu":
            # torch.compile's return type is loosely annotated by PyTorch.
            return torch.compile(model, backend="openxla")  # type: ignore[return-value]
        return torch.compile(model)  # type: ignore[return-value]

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

    # Global optimizations
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

        elif self.name == "tpu" and HAS_XLA:
            # XLA handles its own thread pool; leave PyTorch threads alone.
            pass

    # Helpers
    def sync(self) -> None:
        """Block until all queued ops on *device* have finished.

        No-op on CPU.  Useful for accurate timing.
        """
        if self.name == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.name == "mps":
            torch.mps.synchronize()
        elif self.name == "tpu" and HAS_XLA:
            xm.mark_step()
            xm.wait_device_ops()

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
        if self.name == "tpu" and HAS_XLA:
            try:
                info = xm.get_memory_info(self.device)
                return info.get("bytes_used", 0)
            except Exception:
                return 0
        return 0

    def __str__(self) -> str:
        return (
            f"Platform({self.name}, device={self.device}, "
            f"amp={self.supports_amp}({self.amp_dtype}), "
            f"scaler={self.supports_grad_scaler}, "
            f"compile={self.supports_compile}, workers={self.num_workers})"
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
            return Platform(device=torch.device("cuda", 0), name="cuda")

        if name == "tpu" and HAS_XLA:
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
