#!/usr/bin/env python3
"""Tokenize a HuggingFace dataset and write uint16 binary shards."""

import os
import sys
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import TypedDict, Any
from dotenv import load_dotenv
from datasets import load_dataset

from config import Config
from tokenizer import build_tokenizer

load_dotenv()  # loads HF_TOKEN from .env for gated datasets


class SplitState(TypedDict):
    dir: Path
    buf: np.ndarray
    buf_pos: int
    count: int
    shard_index: int
    target: int


class SplitFiller:
    """Routes token streams into train/val/test splits, flushing shards as they fill."""

    def __init__(self, splits: dict[str, SplitState], shard_size: int, pbar: tqdm[Any]) -> None:
        self.splits = splits
        self.shard_size = shard_size
        self.pbar = pbar
        self.active = [name for name in ("train", "val", "test") if name in splits]
        self.idx = 0

    def _copy(self, split: SplitState, tokens: np.ndarray) -> None:
        """Copy tokens into a split buffer, flushing full shards to disk."""
        offset = 0
        while offset < len(tokens):
            space = self.shard_size - split["buf_pos"]
            if space == 0:
                flush_shard(split)
                space = self.shard_size

            chunk_len = min(len(tokens) - offset, space)
            end = split["buf_pos"] + chunk_len
            split["buf"][split["buf_pos"]:end] = tokens[offset:offset + chunk_len]
            split["buf_pos"] = end
            offset += chunk_len
            split["count"] += chunk_len
            self.pbar.update(chunk_len)

    def all_full(self) -> bool:
        return all(s["count"] >= s["target"] for s in self.splits.values())

    def feed(self, doc_arr: np.ndarray) -> None:
        """Route a single document into splits, crossing split boundaries if necessary."""
        while len(doc_arr) > 0:
            if self.idx >= len(self.active):
                break  # all splits full — drop remaining tokens

            name = self.active[self.idx]
            split = self.splits[name]
            needed = split["target"] - split["count"]
            if needed <= 0:
                self.idx += 1
                continue

            take = min(needed, len(doc_arr))
            self._copy(split, doc_arr[:take])
            doc_arr = doc_arr[take:]

            if split["count"] >= split["target"]:
                flush_shard(split)
                self.idx += 1

    def flush(self) -> None:
        """Persist any trailing partial buffers."""
        for name in self.active:
            flush_shard(self.splits[name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a dataset into uint16 shards.")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name.")
    parser.add_argument("--dataset-config", type=str, default="sample-10BT", help="Dataset config name (default: sample-10BT).")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split to stream (default: train).")
    parser.add_argument("--text-key", type=str, default="text", help="Column name containing text.")
    parser.add_argument("--train-tokens", type=int, required=True, help="Number of training tokens.")
    parser.add_argument("--val-tokens", type=int, default=None, help="Number of validation tokens.")
    parser.add_argument("--test-tokens", type=int, default=None, help="Number of test tokens.")
    parser.add_argument("--shard-size", type=int, default=100_000_000, help="Tokens per shard.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Directory to write splits (default: ./data).")
    parser.add_argument("--tokenizer-backend", type=str, default="huggingface", choices=["tiktoken", "huggingface"], help="Tokenizer backend.")
    parser.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Encoding name (tiktoken) or model id (huggingface).")
    return parser.parse_args()


def make_split(dir: Path, target: int, shard_size: int) -> SplitState:
    """Create a fresh split state with an empty shard buffer."""
    return {
        "dir": dir,
        "buf": np.empty(shard_size, dtype=np.uint16),
        "buf_pos": 0,
        "count": 0,
        "shard_index": 0,
        "target": target,
    }


def build_splits(base_dir: Path, args: argparse.Namespace) -> dict[str, SplitState]:
    """Initialize split states based on CLI args."""
    splits: dict[str, SplitState] = {
        "train": make_split(base_dir / "train", args.train_tokens, args.shard_size),
    }
    if args.val_tokens:
        splits["val"] = make_split(base_dir / "val", args.val_tokens, args.shard_size)
    if args.test_tokens:
        splits["test"] = make_split(base_dir / "test", args.test_tokens, args.shard_size)
    return splits


def flush_shard(split: SplitState) -> None:
    """Write the active buffer to disk if it contains tokens, then reset it."""
    if split["buf_pos"] == 0:
        return
    shard_path = split["dir"] / f"shard_{split['shard_index']:05d}.bin"
    split["buf"][:split["buf_pos"]].tofile(shard_path)
    split["shard_index"] += 1
    split["buf_pos"] = 0


def flatten_batch(batch_tokens: list[list[int]], eot: int) -> np.ndarray:
    """Concatenate tokenized documents with an EOT token after each."""
    batch_len = sum(len(d) + 1 for d in batch_tokens)
    batch_arr = np.empty(batch_len, dtype=np.uint16)
    pos = 0
    for doc_tokens in batch_tokens:
        n = len(doc_tokens)
        batch_arr[pos:pos + n] = doc_tokens
        batch_arr[pos + n] = eot
        pos += n + 1
    return batch_arr


def print_summary(splits: dict[str, SplitState]) -> None:
    for name in ("train", "val", "test"):
        if name not in splits:
            continue
        split = splits[name]
        print(f"  {name}: {split['count']:,} tokens, {split['shard_index']} shards in {split['dir']}")


def hard_exit() -> None:
    """Flush streams and exit without waiting for datasets/fsspec background threads."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    args = parse_args()

    # Derive a filesystem-safe dataset name, e.g. "HuggingFaceFW/fineweb" → "fineweb"
    dataset_name = args.dataset.split("/")[-1]
    base_dir = args.output_dir / dataset_name

    splits = build_splits(base_dir, args)
    for split in splits.values():
        split["dir"].mkdir(parents=True, exist_ok=True)

    config = Config(tokenizer_backend=args.tokenizer_backend, tokenizer_name=args.tokenizer)
    enc = build_tokenizer(config)
    eot = enc.eos_id

    print(f"Streaming {args.dataset} ...")
    stream = load_dataset(args.dataset, args.dataset_config, streaming=True, split=args.dataset_split)
    # Group the stream into batches of docs to amortize Python loop overhead.
    stream = stream.batch(1_000)

    tokens_to_process = sum(s["target"] for s in splits.values())
    filler = SplitFiller(
        splits,
        args.shard_size,
        tqdm(total=tokens_to_process, desc=dataset_name, unit="tokens", unit_scale=True),
    )

    try:
        for batch in stream:
            if filler.all_full():
                break

            batch_tokens = enc.encode_batch(batch[args.text_key])
            batch_arr = flatten_batch(batch_tokens, eot)

            pos = 0
            for doc_tokens in batch_tokens:
                doc_len = len(doc_tokens) + 1
                filler.feed(batch_arr[pos:pos + doc_len])
                pos += doc_len
    finally:
        filler.flush()

    print_summary(splits)
    hard_exit()


if __name__ == "__main__":
    main()
