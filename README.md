# Kigo

A ~162M-parameter language model pretrained from scratch with modern architectural improvements. Small enough to run on-device, capable enough to specialize for precision tasks through fine-tuning.

## What this is

Most foundation models are too large to run locally or too outdated to perform well. Kigo occupies the gap: a modern, efficient architecture at a parameter count that fits consumer hardware, pretrained on high-quality data so it can be fine-tuned effectively for specific domains.

The architecture combines four modern components — RoPE, SwiGLU, RMSNorm, and SDPA attention — to train more stably, converge faster, and produce better representations than an equivalent-parameter vanilla transformer.

## Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | ~162M |
| Layers | 12 |
| Heads | 12 |
| d_model | 768 |
| Context length | 1024 |
| Vocabulary | 100277 (cl100k_base) |
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

The training script loads `train/` and `val/` splits from the path passed via `--data-dir` (default: `data/`). Both splits are required.

## Training

Kigo uses **PyTorch Lightning** for training. The config file controls the model architecture, training hyperparameters, device/precision, and inference defaults.

### Local development

```bash
uv run python train.py --config config/dev.yaml --data-dir data
```

Use a tiny configuration (small model, small batch, few tokens) to iterate quickly on architecture and training logic.

### Full training run

```bash
uv run python train.py --config config/kigo-162m.yaml
```

The trainer auto-discovers `checkpoints/last.ckpt` and resumes from the next step. W&B logging continues under the same run ID across restarts.

### Tuning `num_workers`

The `Platform` helper picks a default `DataLoader` worker count based on your accelerator and host CPU cores, but the real optimum depends on your preprocessing and storage speed. To maximize throughput:

1. Run a short training window (a few hundred steps).
2. Check accelerator utilization:
   - **CUDA**: `nvidia-smi dmon` should show the GPU near 100%.
   - **TPU**: `torch_xla` debug metrics or `xla_device_metrics`.
3. If utilization is low and the host CPU is busy, increase workers; if the CPU is oversubscribed or step latency spikes, decrease workers.
4. Sweep a few values (2, 4, 8, 16) and pick the one with the highest stable `tok/s` in the W&B dashboard (`train/tokens_per_sec`).

The default is a safe starting point; the best value is found empirically on your hardware.

### Platform notes

Kigo is trained on Lightning AI's free tier, which imposes a 4-hour limit per studio session. The training script handles this transparently:

- **Automatic checkpointing** every `checkpoint_interval` steps via PyTorch Lightning `ModelCheckpoint`.
- **SIGTERM handler** catches the platform's pre-kill signal and writes an emergency checkpoint before exiting cleanly.
- **Resume on restart** — when you restart the studio and re-run the script, it picks up from `last.ckpt` without manual intervention.
- **W&B continuity** — logging resumes under the same run ID across sessions.

To run remotely, point `--data-dir` and `--checkpoint-dir` at the studio paths:

```bash
uv run python train.py --config config/kigo-162m.yaml \
    --data-dir /teamspace/studios/this_studio/data \
    --checkpoint-dir /teamspace/studios/this_studio/checkpoints
```

## Project Structure

```
├── train.py              # Training orchestrator (PyTorch Lightning Trainer)
├── nn/                   # Model architecture
│   ├── model.py          # GPT assembly
│   ├── lightning_module.py  # PyTorch Lightning module
│   ├── layers.py         # Attention + MLP blocks
│   ├── embeddings.py     # Token + RoPE
│   ├── activations.py    # SwiGLU
│   ├── norm.py           # RMSNorm
│   └── init.py           # Weight initialization
├── data.py               # Memmap Dataset
├── data_module.py        # PyTorch Lightning data module
├── callbacks.py          # Throughput, sampling, emergency checkpoint callbacks
├── eval.py               # Text generation utilities
├── config.py             # Hyperparameters and model config
├── accelerator.py        # Platform detection and DataLoader tuning
├── scripts/
│   ├── tokenize_data.py  # Dataset preprocessing
│   └── run_eval.py       # Standalone evaluation
└── checkpoints/          # Runtime: model checkpoints (not committed)
```

## License

MIT
