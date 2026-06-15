"""Memory-mapped dataloader for uint16 token shards.

Each shard is a flat array of uint16 token IDs. Shards are concatenated logically
into a single token stream, then sliced into fixed-length sequences for training.
"""

import torch
import numpy as np
from pathlib import Path
from typing import cast
from numpy.typing import NDArray
from torch.utils.data import Dataset, DataLoader


class TokenBuffer:
    """A flat, concatenated view over multiple memory-mapped uint16 shard files."""

    def __init__(self, shard_dir: Path | str) -> None:
        self.shard_dir = Path(shard_dir)
        if not self.shard_dir.is_dir():
            raise FileNotFoundError(f"Shard directory not found: {self.shard_dir}")

        self.shard_paths: list[Path] = sorted(self.shard_dir.glob("*.bin"))
        if not self.shard_paths:
            raise ValueError(f"No *.bin shards found in {self.shard_dir}")

        # Size shards from the filesystem without memmapping: opening every
        # shard in the parent would bloat each forked DataLoader worker's
        # inherited address space. Shards are memmapped lazily, per worker.
        self.shard_sizes: list[int] = []
        self.cumulative_sizes: list[int] = []
        total = 0
        for path in self.shard_paths:
            n = path.stat().st_size // 2  # uint16 = 2 bytes per token
            self.shard_sizes.append(n)
            total += n
            self.cumulative_sizes.append(total)

        self.total_tokens = total
        self._shard_cache: dict[int, NDArray[np.uint16]] = {}

    def __len__(self) -> int:
        return self.total_tokens

    def get_slice(self, start: int, length: int) -> NDArray[np.uint16]:
        """Read a contiguous slice of tokens, potentially crossing shard boundaries.

        This is more efficient than repeated __getitem__ calls because it
        delegates to numpy's slicing where possible.
        """
        if start < 0 or start + length > self.total_tokens:
            raise IndexError(
                f"Slice [{start}:{start + length}) out of range [0, {self.total_tokens})"
            )

        tokens = cast(NDArray[np.uint16], np.empty(length, dtype=np.uint16))
        written = 0

        while written < length:
            global_pos = start + written
            shard_idx = int(np.searchsorted(self.cumulative_sizes, global_pos, side="right"))
            shard_start = 0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
            local_pos = global_pos - shard_start
            shard = self._get_shard(shard_idx)
            available_in_shard = self.shard_sizes[shard_idx] - local_pos
            to_copy = min(length - written, available_in_shard)

            tokens[written : written + to_copy] = shard[local_pos : local_pos + to_copy]
            written += to_copy

        return tokens

    def _get_shard(self, shard_idx: int) -> NDArray[np.uint16]:
        """Memmap a shard on first access and cache the handle for this process."""
        cached = self._shard_cache.get(shard_idx)
        if cached is None:
            cached = np.memmap(self.shard_paths[shard_idx], dtype=np.uint16, mode="r")
            self._shard_cache[shard_idx] = cached
        return cached


class MemmapDataset(Dataset):
    """PyTorch Dataset yielding fixed-length token sequences from a TokenBuffer.

    Sequences are non-overlapping so DataLoader(shuffle=True) gives a random
    order each epoch without extra randomness inside __getitem__.
    """

    def __init__(self, split_dir: Path | str, sequence_length: int) -> None:
        self.buffer = TokenBuffer(split_dir)
        self.sequence_length = sequence_length

        # Each sample needs (sequence_length + 1) tokens:
        # sequence_length inputs and 1 target.
        tokens_per_sample = sequence_length + 1
        self.num_samples = self.buffer.total_tokens // tokens_per_sample

        if self.num_samples == 0:
            raise ValueError(
                f"Not enough tokens ({self.buffer.total_tokens}) for one sequence "
                f"of length {sequence_length}"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a uint16 Tensor of shape (sequence_length + 1,)."""
        tokens_per_sample = self.sequence_length + 1
        start = idx * tokens_per_sample
        token_array = self.buffer.get_slice(start, tokens_per_sample)
        # Same width as on disk; the training loop casts to int64 before the GPU.
        return torch.from_numpy(token_array)


def get_dataloader(
    split_dir: Path | str,
    sequence_length: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
) -> DataLoader:
    """Create a DataLoader for a single split directory containing .bin shards.

    Args:
        split_dir: Path to a directory containing *.bin uint16 shards.
        sequence_length: Number of input tokens per sequence. Each sample will
            be (sequence_length + 1) tokens to provide targets.
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle sample order each epoch.
        num_workers: Number of background worker processes for data loading.
        pin_memory: Copy tensors to CUDA pinned memory before returning.
        drop_last: Drop the last incomplete batch.
        prefetch_factor: Number of batches loaded in advance by each worker.
            Only used when ``num_workers > 0``.
        persistent_workers: Whether to keep worker processes alive across
            epochs. Only used when ``num_workers > 0``.

    Returns:
        A DataLoader yielding batches of shape (batch_size, sequence_length + 1).
    """
    dataset = MemmapDataset(split_dir, sequence_length)

    kwargs: dict = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)
