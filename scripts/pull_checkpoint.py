#!/usr/bin/env python3
"""Download checkpoints from the Hub before training.

Pulls the full set so the local dir mirrors upstream: HubModelCheckpoint uploads
with delete_patterns, which prunes any remote checkpoint not present locally, so
a partial pull would delete the rest. Run this from a launcher before train.py so
the training process only ever uploads.

Usage:
    python scripts/pull_checkpoint.py --hf-repo user/kigo --checkpoint-dir DIR
"""

import argparse

from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull checkpoints from the Hub.")
    parser.add_argument("--hf-repo", type=str, required=True, help="HF model repo, e.g. user/kigo.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory to download the checkpoints and wandb_run_id into.",
    )
    args = parser.parse_args()

    try:
        snapshot_download(repo_id=args.hf_repo, repo_type="model", local_dir=args.checkpoint_dir)
    except RepositoryNotFoundError:
        print(f"No Hub repo {args.hf_repo} yet; starting fresh.")


if __name__ == "__main__":
    main()
