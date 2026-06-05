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
    """A flat, concatenated view over multiple memory-mapped uint16 shard files.

    Usage:
        buffer = TokenBuffer(Path("data/fineweb-edu/train"))
        token_id = buffer[1_000_000]  # global index across all shards
    """

    def __init__(self, shard_dir: Path | str) -> None:
        self.shard_dir = Path(shard_dir)
        if not self.shard_dir.is_dir():
            raise FileNotFoundError(f"Shard directory not found: {self.shard_dir}")

        self.shard_paths: list[Path] = sorted(self.shard_dir.glob("*.bin"))
        if not self.shard_paths:
            raise ValueError(f"No *.bin shards found in {self.shard_dir}")

        self.shards: list[NDArray[np.uint16]] = []
        self.cumulative_sizes: list[int] = []
        total_tokens = 0

        for path in self.shard_paths:
            shard = np.memmap(path, dtype=np.uint16, mode="r")
            self.shards.append(shard)
            total_tokens += len(shard)
            self.cumulative_sizes.append(total_tokens)

        self.total_tokens = total_tokens

    def __len__(self) -> int:
        return self.total_tokens

    def __getitem__(self, global_idx: int) -> int:
        """Map a global token index to the correct shard and local offset."""
        if global_idx < 0 or global_idx >= self.total_tokens:
            raise IndexError(f"Token index {global_idx} out of range [0, {self.total_tokens})")

        shard_idx = int(np.searchsorted(self.cumulative_sizes, global_idx, side="right"))
        local_idx = global_idx if shard_idx == 0 else global_idx - self.cumulative_sizes[shard_idx - 1]
        return int(self.shards[shard_idx][local_idx])

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
            shard = self.shards[shard_idx]
            available_in_shard = len(shard) - local_pos
            to_copy = min(length - written, available_in_shard)

            tokens[written : written + to_copy] = shard[local_pos : local_pos + to_copy]
            written += to_copy

        return tokens


class MemmapDataset(Dataset):
    """PyTorch Dataset yielding fixed-length token sequences from a TokenBuffer.

    Sequences are non-overlapping so DataLoader(shuffle=True) gives a random
    order each epoch without extra randomness inside __getitem__.
    """

    def __init__(self, split_dir: Path | str, sequence_length: int) -> None:
        self.buffer = TokenBuffer(split_dir)
        self.sequence_length = sequence_length
        self.samples_per_shard: list[int] = []

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
        """Return a LongTensor of shape (sequence_length + 1,)."""
        tokens_per_sample = self.sequence_length + 1
        start = idx * tokens_per_sample
        token_array = self.buffer.get_slice(start, tokens_per_sample)
        # Return as uint16 — same width as on disk. The training loop will cast
        # to int32 (for nn.Embedding) or int64 (for CrossEntropyLoss) on the
        # GPU, keeping CPU→GPU transfer at 2 bytes per token.
        return torch.from_numpy(token_array)


def get_dataloader(
    split_dir: Path | str,
    sequence_length: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
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

    Returns:
        A DataLoader yielding batches of shape (batch_size, sequence_length + 1).
    """
    dataset = MemmapDataset(split_dir, sequence_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
