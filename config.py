"""Configuration data class and YAML loading for Kigo training.

Usage:
    from config import Config, load_config
    cfg = load_config("config/kigo-162m.yaml")
"""

import yaml
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal


_Precision = Literal["16-mixed", "bf16-mixed", "32-true"]


@dataclass
class Config:
    """Single source of truth for model architecture and training hyperparameters."""

    # --- Tokenizer ---
    # Backend selects the implementation: "tiktoken" or "huggingface".
    # tokenizer_name is the encoding name (tiktoken) or model id (huggingface).
    tokenizer_backend: Literal["tiktoken", "huggingface"] = "huggingface"
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M"

    # --- Model architecture ---
    vocab_size: int = 49152
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 2048
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False

    # --- RoPE ---
    rope_base: float = 10000.0
    rope_max_length: int = 2048

    # --- Training ---
    batch_size: int = 8
    global_batch_size: int = 512
    max_epochs: int = 4
    adamw_lr: float = 6e-4  # AdamW group (embeddings, head, norms, biases)
    muon_lr: float = 0.02   # Muon group (2D hidden weight matrices)
    min_lr: float = 6e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # --- Intervals ---
    checkpoint_interval: int = 500
    eval_interval: float = 0.1
    log_interval: int = 10
    sample_interval: int = 2000

    # --- Device & precision ---
    # None = auto-detect best platform / dtype / compile support
    device: str | None = None
    dtype: _Precision | None = None
    compile: bool | None = None

    # --- W&B ---
    wandb_project: str = "kigo"
    wandb_run_name: str | None = None

    # --- Inference defaults ---
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        if self.global_batch_size % self.batch_size != 0:
            raise ValueError(
                f"global_batch_size ({self.global_batch_size}) must be divisible by "
                f"batch_size ({self.batch_size})"
            )
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout ({self.dropout}) must be in [0, 1]")
        if not (0.0 <= self.weight_decay <= 1.0):
            raise ValueError(f"weight_decay ({self.weight_decay}) must be in [0, 1]")


def _coerce_value(field_type: Any, raw: Any) -> Any:
    """Convert a raw YAML value to the type expected by the dataclass field."""
    if field_type is Path or getattr(field_type, "__origin__", None) is not None and Path in field_type.__args__:
        return Path(raw)
    return raw


def load_config(path: str | Path, base: Config | None = None) -> Config:
    """Load a YAML file into a Config dataclass.

    Unspecified fields keep their defaults (or the base config's values).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    cfg = base if base is not None else Config()
    cfg_fields = {f.name: f.type for f in fields(Config)}

    for key, value in raw.items():
        if key not in cfg_fields:
            raise ValueError(f"Unknown config key: {key}")
        setattr(cfg, key, _coerce_value(cfg_fields[key], value))

    return cfg
