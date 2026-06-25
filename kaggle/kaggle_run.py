#!/usr/bin/env python3
"""Kaggle script-kernel launcher.

Clones the repo, points training at the attached tokenized dataset, and syncs
checkpoints to the Hub. All user-specific values come from Kaggle Secrets, so
this file holds no personal ids and runs unchanged for anyone. Push with:

    kaggle kernels push -p kaggle

Required Kaggle Secrets: HF_TOKEN, WANDB_API_KEY, REPO_URL (e.g.
github.com/you/Kigo.git), HF_CKPT_REPO (e.g. you/kigo). Add GITHUB_TOKEN too if
the repo is private.
"""

import os
import subprocess
import tarfile
from pathlib import Path

import torch
from kaggle_secrets import UserSecretsClient  # type: ignore[reportMissingImports]

secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
repo_url = secrets.get_secret("REPO_URL")
hf_repo = secrets.get_secret("HF_CKPT_REPO")

# Private repo needs a GITHUB_TOKEN secret; a public repo clones without one.
try:
    repo_url = f"{secrets.get_secret('GITHUB_TOKEN')}@{repo_url}"
except Exception:
    pass

# Optional per-target micro-batch; falls back to the config default if unset.
try:
    batch_size = secrets.get_secret("BATCH_SIZE")
except Exception:
    batch_size = None

subprocess.run(["git", "clone", "--depth", "1", f"https://{repo_url}", "repo"], check=True)
# Pin Kaggle's CUDA-matched torch so pip installs the pyproject deps without replacing it.
Path("repo/torch-pin.txt").write_text(f"torch=={torch.__version__}\n")
subprocess.run(["pip", "install", "-e", ".", "--constraint", "torch-pin.txt"], cwd="repo", check=True)

# Auto-discover the attached dataset: find the train/ split at any depth (the
# mount may nest it), or unpack a .tar/.tar.gz archive as a fallback.
inputs = Path("/kaggle/input")
train = next((p for p in inputs.rglob("train") if p.is_dir()), None)
if train is not None:
    data_dir = train.parent
else:
    archives = sorted(inputs.glob("*/*.tar*"))
    if not archives:
        raise FileNotFoundError(f"No train/ split or .tar archive under {inputs}")
    data_dir = Path("/kaggle/working/data")
    with tarfile.open(archives[0]) as tf:
        tf.extractall(data_dir)
    train = next(p for p in data_dir.rglob("train") if p.is_dir())
    data_dir = train.parent

# Keep checkpoints outside the repo clone (like the dataset) so a re-clone is clean.
checkpoint_dir = "/kaggle/working/checkpoints"

# Pull only the latest checkpoint so a disconnected session resumes without
# re-downloading the whole top-k repo.
subprocess.run(
    ["python", "scripts/pull_checkpoint.py", "--hf-repo", hf_repo, "--checkpoint-dir", checkpoint_dir],
    cwd="repo",
    check=True,
)

train_cmd = [
    "python", "train.py",
    "--config", "config/kigo-162m.yaml",
    "--data-dir", str(data_dir),
    "--checkpoint-dir", checkpoint_dir,
    "--hf-repo", hf_repo,
]
if batch_size:
    train_cmd += ["--batch-size", batch_size]
subprocess.run(train_cmd, cwd="repo", check=True)
