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

subprocess.run(["git", "clone", "--depth", "1", f"https://{repo_url}", "repo"], check=True)
subprocess.run(["pip", "install", "-e", "."], cwd="repo", check=True)

# Auto-discover the attached dataset: the split dir, or a tar to unpack.
inputs = Path("/kaggle/input")
data_dir = next((d for d in inputs.glob("*") if (d / "train").is_dir()), None)
if data_dir is None:
    tars = list(inputs.glob("*/*.tar"))
    if not tars:
        raise FileNotFoundError(f"No train/ split or .tar archive under {inputs}")
    data_dir = Path("/kaggle/working/data")
    with tarfile.open(tars[0]) as tf:
        tf.extractall(data_dir)

subprocess.run(
    ["python", "train.py",
     "--config", "config/kigo-162m.yaml",
     "--data-dir", str(data_dir),
     "--hf-repo", hf_repo],
    cwd="repo", check=True,
)
