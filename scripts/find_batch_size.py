#!/usr/bin/env python3
"""Find the largest micro-batch that fits on the current device.

Probes peak memory with synthetic token batches (no dataset required) by running
a realistic training step -- forward, backward, and ``optimizer.step()`` so Muon
and AdamW state is allocated -- under the configured precision. Doubles the batch
until OOM, then binary-searches the boundary, and prints a config suggestion.

Only the CUDA path has a hard, catchable memory ceiling, so the probe is limited
to CUDA. MPS allocates from unified system RAM and thrashes swap instead of
raising OOM; CPU/TPU likewise have no comparable OOM to bisect.

Usage:
    python scripts/find_batch_size.py --config config/kigo-162m.yaml
"""

import argparse
import contextlib
import gc
import time

import torch

from accelerator import get_platform
from config import Config, Precision, load_config
from nn.model import GPT
from optimizers import Muon, split_params

# Untimed warmup steps (let cuDNN autotune and the optimizer allocate state),
# then timed steps used to measure throughput.
WARMUP_STEPS = 2
TIMED_STEPS = 3
# Stop doubling here so a huge-memory device can't loop forever.
MAX_BATCH = 8192
# Recommend the largest batch whose peak stays under this fraction of total
# VRAM, leaving room for fragmentation, the validation pass, sample generation,
# and torch.compile (which real training applies but this probe does not).
MEM_HEADROOM = 0.9

_AMP_DTYPE: dict[Precision, torch.dtype | None] = {
    "bf16-mixed": torch.bfloat16,
    "16-mixed": torch.float16,
    "32-true": None,
}


def build(config: Config, device: torch.device) -> tuple[GPT, Muon]:
    """Build a fresh model + optimizer, matching the training configuration."""
    model = GPT(config).to(device)
    muon_params, adamw_params = split_params(model)
    optimizer = Muon(
        muon_params,
        adamw_params,
        muon_lr=config.muon_lr,
        adamw_lr=config.adamw_lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    return model, optimizer


def trial(config: Config, device: torch.device, batch_size: int, amp_dtype: torch.dtype | None) -> tuple[int, float] | None:
    """Run warmup + timed steps at ``batch_size``.

    Returns ``(peak_bytes, tokens_per_sec)``, or ``None`` on OOM.
    """
    autocast = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else contextlib.nullcontext()
    )
    torch.cuda.reset_peak_memory_stats(device)
    model, optimizer = build(config, device)
    model.train()

    def step() -> None:
        # Each sample is block_size + 1 tokens (inputs + shifted targets).
        batch = torch.randint(
            0, config.vocab_size, (batch_size, config.block_size + 1), device=device
        )
        inputs, targets = batch[:, :-1], batch[:, 1:]
        with autocast:
            _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    try:
        for _ in range(WARMUP_STEPS):
            step()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(TIMED_STEPS):
            step()
        torch.cuda.synchronize(device)
        tokens_per_sec = batch_size * config.block_size * TIMED_STEPS / (time.perf_counter() - start)
        return torch.cuda.max_memory_allocated(device), tokens_per_sec
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        return None
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def search(
    config: Config, device: torch.device, amp_dtype: torch.dtype | None, budget: int
) -> tuple[int, list[tuple[int, int, float]]]:
    """Find the largest batch whose peak stays within ``budget`` bytes.

    Returns ``(recommended_batch, measured)`` where ``measured`` is the list of
    ``(batch, peak_bytes, tokens_per_sec)`` for every batch that fit the budget.
    """
    measured: list[tuple[int, int, float]] = []

    def within_budget(batch_size: int) -> bool:
        result = trial(config, device, batch_size, amp_dtype)
        if result is None:
            print(f"  bs={batch_size:<4d} OOM")
            return False
        peak, tput = result
        ok = peak <= budget
        print(f"  bs={batch_size:<4d} {peak / 1024**3:5.2f} GiB  {tput:>8,.0f} tok/s"
              f"{'' if ok else '  over budget'}")
        if ok:
            measured.append((batch_size, peak, tput))
        return ok

    # Phase 1: double until OOM / over budget (or the cap).
    bad = None
    batch = 1
    while batch <= MAX_BATCH:
        if not within_budget(batch):
            bad = batch
            break
        batch *= 2

    if not measured:
        raise RuntimeError("batch_size=1 already exceeds the memory budget; the model does not fit.")

    # Phase 2: binary-search the boundary between the largest good batch and the first bad one.
    if bad is not None:
        lo, hi = max(b for b, _, _ in measured), bad
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if within_budget(mid):
                lo = mid
            else:
                hi = mid

    return max(b for b, _, _ in measured), measured


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the largest micro-batch that fits.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    platform = get_platform(prefer=config.device)
    if platform.name != "cuda":
        print(
            f"Batch-size search is only supported on CUDA (got {platform.name!r})."
        )
        return

    device = platform.device
    platform.optimize()
    precision = config.dtype or platform.default_precision
    amp_dtype = _AMP_DTYPE[precision]

    props = torch.cuda.get_device_properties(device)
    n_params = sum(p.numel() for p in GPT(config).parameters())
    total = props.total_memory
    budget = int(MEM_HEADROOM * total)

    print("=== Kigo batch-size finder ===")
    print(f"Device : {props.name}  ({total / 1024**3:.1f} GiB total, "
          f"{budget / 1024**3:.1f} GiB budget @ {MEM_HEADROOM:.0%})")
    print(f"Model  : {n_params / 1e6:.0f}M params, block_size={config.block_size}, precision={precision}")
    print(f"\nSearching for the largest batch within budget "
          f"({WARMUP_STEPS} warmup + {TIMED_STEPS} timed steps each):")

    recommended, measured = search(config, device, amp_dtype, budget)
    peak = next(p for b, p, _ in measured if b == recommended)
    tput = next(t for b, _, t in measured if b == recommended)

    accumulation = max(1, round(config.global_batch_size / recommended))
    effective = recommended * accumulation

    print("\nResult:")
    print(f"  Largest batch within budget : bs={recommended}  ({peak / 1024**3:.2f} GiB, {tput:,.0f} tok/s)")
    # Show what the last doublings actually bought, so plateaus are obvious.
    smaller = [m for m in measured if m[0] <= recommended // 2]
    if smaller:
        base_b, _, base_t = max(smaller, key=lambda m: m[0])
        gain = (tput / base_t - 1) * 100
        note = "  <- diminishing returns" if gain < 10 else ""
        print(f"  Throughput vs bs={base_b:<11d}: {gain:+.0f}% for {recommended / base_b:.1f}x the batch{note}")

    print("\nApply to your config:")
    print(f"  batch_size: {recommended}")
    print(f"  global_batch_size: {effective}    # was {config.global_batch_size}  (= {recommended} x {accumulation})")
    print(f"  -> accumulate_grad_batches = {accumulation} (computed by train.py), "
          f"tokens/step = {effective * config.block_size:,}")


if __name__ == "__main__":
    main()
