#!/usr/bin/env python3
"""Download only the latest checkpoint (plus wandb_run_id) from the Hub.

Resuming needs just the highest-step checkpoint, so we fetch that one file
instead of the whole repo -- the upstream repo still keeps the top-k. Run this
from a launcher before train.py so the training process only ever uploads.

Usage:
    python scripts/pull_checkpoint.py --hf-repo user/kigo --checkpoint-dir DIR
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull the latest checkpoint from the Hub.")
    parser.add_argument("--hf-repo", type=str, required=True, help="HF model repo, e.g. user/kigo.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory to download the checkpoint and wandb_run_id into.",
    )
    args = parser.parse_args()

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=args.hf_repo, repo_type="model")
    except RepositoryNotFoundError:
        print(f"No Hub repo {args.hf_repo} yet; starting fresh.")
        return

    # Zero-padded step names sort lexicographically, so the last is the highest.
    ckpts = sorted(f for f in files if f.startswith("checkpoint_step_") and f.endswith(".ckpt"))
    if not ckpts:
        print(f"No checkpoints in {args.hf_repo} yet; starting fresh.")
        return

    targets = [ckpts[-1]]
    if "wandb_run_id" in files:
        targets.append("wandb_run_id")

    for filename in targets:
        hf_hub_download(
            repo_id=args.hf_repo,
            repo_type="model",
            filename=filename,
            local_dir=args.checkpoint_dir,
        )
    print(f"Pulled {ckpts[-1]} into {Path(args.checkpoint_dir)}")


if __name__ == "__main__":
    main()
