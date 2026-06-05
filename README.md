# Kigo

A 124M-parameter language model pretrained from scratch with modern architectural improvements. Small enough to run on-device, capable enough to specialize for precision tasks through fine-tuning.

## What this is

Most foundation models are too large to run locally or too outdated to perform well. Kigo occupies the gap: a modern, efficient architecture at a parameter count that fits consumer hardware, pretrained on high-quality data so it can be fine-tuned effectively for specific domains.

The architecture swaps four components of the classic GPT-2 small for their modern equivalents — RoPE, SwiGLU, RMSNorm, and SDPA attention — without increasing parameter count. The result trains more stably, converges faster, and produces better representations at the same 124M scale.

## Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | ~124M |
| Layers | 12 |
| Heads | 12 |
| d_model | 768 |
| Context length | 1024 |
| Vocabulary | 50257 (GPT-2 BPE) |
| Positional encoding | RoPE (Rotary Position Embedding) |
| FFN activation | SwiGLU |
| Normalization | RMSNorm + Pre-LN |
| Attention kernel | PyTorch SDPA |
| FFN expansion | ~2.67× (d_ff = 2048) |

## Setup

```bash
uv sync
```

## Data

Training uses a mix of **FineWeb-Edu** (high-quality educational text) and **The Stack v2** (code) for structured reasoning. All data is pre-tokenized offline to uint16 binary shards and loaded via memory-mapped arrays during training.

### Download training data

One command per corpus. `--train-tokens` is required; `--val-tokens` and `--test-tokens` are optional. Output defaults to `./data/`, with each dataset auto-organized into its own subdirectory.

```bash
# FineWeb-Edu
uv run scripts/prepare_dataset.py \
    --dataset HuggingFaceFW/fineweb-edu \
    --train-tokens 2_000_000_000 \
    --val-tokens 20_000_000 \
    --test-tokens 20_000_000
```

This creates:
```
data/
├── fineweb-edu/
│   ├── train/
│   │   ├── data_000.bin
│   │   └── meta.json
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── the-stack-v2/
    ├── train/
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

The training script discovers all `train/` directories recursively and mixes them. `val/` directories are evaluated separately during training to track per-domain perplexity (text vs. code). `test/` directories are evaluated once at the end for a final per-domain perplexity number.

## Training

### Local development

```bash
uv run python train.py --config config/dev.yaml
```

Use a tiny configuration (small model, small batch, few tokens) to iterate quickly on architecture and training logic.

### Full training run

```bash
uv run python train.py --config config/train.yaml
```

The training script auto-discovers the latest checkpoint and resumes from `step + 1`. W&B logging continues under the same run ID across restarts.

### Platform notes

Kigo is trained on Lightning AI's free tier, which imposes a 4-hour limit per studio session. The training script handles this transparently:

- **Automatic checkpointing** every 500 steps to persistent storage (`/teamspace/studios/this_studio/`).
- **SIGTERM handler** catches the platform's pre-kill signal and writes an emergency checkpoint before exiting cleanly.
- **Resume on restart** — when you restart the studio and re-run the script, it picks up from the latest checkpoint without manual intervention.
- **W&B continuity** — logging resumes under the same run ID across sessions.

To run remotely, pass the persistent storage directory:

```bash
uv run python train.py \
    --config config/train.yaml \
    --checkpoint-dir /teamspace/studios/this_studio/checkpoints \
    --data-dir /teamspace/studios/this_studio/data
```

## Project Structure

```
├── train.py              # Main training loop, resume logic, W&B logging
├── nn/                   # Model architecture
│   ├── model.py          # GPT assembly
│   ├── layers.py         # Attention + MLP blocks
│   ├── embeddings.py     # Token + RoPE
│   ├── activations.py    # SwiGLU
│   ├── norm.py           # RMSNorm
│   └── init.py           # Weight initialization
├── data.py               # Memmap DataLoader, mixing, streaming
├── checkpoint.py         # Save/resume/cleanup, SIGTERM handler
├── eval.py               # Validation, benchmarks, generation
├── config.py             # Hyperparameters and model config
├── scripts/
│   ├── tokenize_data.py  # Dataset preprocessing
│   └── run_eval.py       # Standalone evaluation
└── checkpoints/          # Runtime: model checkpoints (not committed)
```

## License

MIT
