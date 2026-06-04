#!/usr/bin/env python3
"""Tokenize a HuggingFace dataset and write uint16 binary shards."""

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import tiktoken
from datasets import load_dataset

from dotenv import load_dotenv

load_dotenv()


class SplitState(TypedDict):
    dir: Path
    tokens: list[int]
    count: int
    shard_index: int
    target: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a dataset into uint16 shards.")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name.")
    parser.add_argument("--text-key", type=str, default="text", help="Column name containing text.")
    parser.add_argument("--train-tokens", type=int, required=True, help="Number of training tokens.")
    parser.add_argument("--val-tokens", type=int, default=None, help="Number of validation tokens.")
    parser.add_argument("--test-tokens", type=int, default=None, help="Number of test tokens.")
    parser.add_argument("--shard-size", type=int, default=100_000_000, help="Tokens per shard.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Directory to write splits (default: ./data).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling the dataset stream.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_name = args.dataset.split("/")[-1]
    base_dir = args.output_dir / dataset_name


    splits: dict[str, SplitState] = {
        "train": {"dir": base_dir / "train", "tokens": [], "count": 0, "shard_index": 0, "target": args.train_tokens},
    }
    if args.val_tokens:
        splits["val"] = {"dir": base_dir / "val", "tokens": [], "count": 0, "shard_index": 0, "target": args.val_tokens}
    if args.test_tokens:
        splits["test"] = {"dir": base_dir / "test", "tokens": [], "count": 0, "shard_index": 0, "target": args.test_tokens}

    for split in splits.values():
        split["dir"].mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]  # 50256

    print(f"Streaming {args.dataset} ...")
    stream = load_dataset(args.dataset, split="train", streaming=True)
    if args.seed is not None:
        stream = stream.shuffle(seed=args.seed)

    for doc in stream:
        text = doc.get(args.text_key, "")
        if not text:
            continue

        tokens = enc.encode_ordinary(text)
        tokens.append(eot)

        for token in tokens:
            split_name = _pick_split(splits)
            if split_name is None:
                break

            split = splits[split_name]
            split["tokens"].append(token)
            split["count"] += 1

            if len(split["tokens"]) >= args.shard_size:
                _write_shard(split["dir"], split["shard_index"], split["tokens"])
                split["shard_index"] += 1
                split["tokens"] = []

        if all(s["count"] >= s["target"] for s in splits.values()):
            break

    for split_name, split in splits.items():
        if split["tokens"]:
            _write_shard(split["dir"], split["shard_index"], split["tokens"])
            split["shard_index"] += 1
        _write_meta(split["dir"], args, split_name, enc.n_vocab, split["count"], split["shard_index"])

    counts = ", ".join(f"{name}: {s['count']:,}" for name, s in splits.items())
    print(f"Done. {counts}")
    print(f"Output: {args.output_dir}")


def _pick_split(splits: dict[str, SplitState]) -> str | None:
    """Assign to the split with the most remaining quota."""
    candidates: dict[str, int] = {name: s["target"] - s["count"] for name, s in splits.items() if s["count"] < s["target"]}
    if not candidates:
        return None
    return max(candidates, key=lambda k: candidates[k])


def _write_shard(output_dir: Path, index: int, tokens: list[int]) -> None:
    path = output_dir / f"data_{index:03d}.bin"
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(path)
    print(f"  -> {path} ({len(tokens):,} tokens)")


def _write_meta(
    output_dir: Path,
    args: argparse.Namespace,
    split: str,
    vocab_size: int,
    total_tokens: int,
    num_shards: int,
) -> None:
    meta = {
        "dataset": args.dataset,
        "split": split,
        "text_key": args.text_key,
        "tokenizer": "gpt2",
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "shard_size": args.shard_size,
        "num_shards": num_shards,
    }
    path = output_dir / "meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
