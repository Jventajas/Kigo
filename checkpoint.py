"""Checkpoint manager: atomic saves, resume, cleanup, and SIGTERM handling."""

import json
import shutil
import signal
from pathlib import Path

import torch


class CheckpointManager:
    """Manages model checkpointing with atomic writes and automatic cleanup.

    Usage:
        ckpt = CheckpointManager("checkpoints")
        ckpt.save(step, model, optimizer, scheduler, rng_state, metadata)
        metadata = ckpt.load_latest(model, optimizer, scheduler)
    """

    REQUIRED_FILES = ("model.pt", "optimizer.pt", "rng_state.pt", "metadata.json")

    def __init__(self, checkpoint_dir: str | Path, keep_last_n: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.latest_symlink = self.checkpoint_dir / "latest_checkpoint.txt"
        self._sigterm_received = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Register SIGTERM and SIGINT handlers for emergency checkpoints."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, _frame) -> None:
        """Set flag so the training loop can write an emergency checkpoint."""
        print(f"\nReceived signal {signum}, preparing emergency checkpoint...")
        self._sigterm_received = True

    def sigterm_received(self) -> bool:
        """Return True if a SIGTERM or SIGINT has been received."""
        return self._sigterm_received

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        rng_state: dict,
        metadata: dict,
        emergency: bool = False,
    ) -> Path:
        """Save a checkpoint atomically.

        Writes all components to a temporary directory, then renames it.
        Updates the ``latest_checkpoint.txt`` symlink on success.
        """
        name = f"checkpoint_step_{step}"
        if emergency:
            name += "_emergency"

        checkpoint_dir = self.checkpoint_dir / name
        tmp_dir = checkpoint_dir.with_suffix(".tmp")

        # Remove any stale temp directory from a previous interrupted save.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        # Save each component separately so partial corruption is isolated.
        torch.save(model.state_dict(), tmp_dir / "model.pt")
        torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")
        torch.save(rng_state, tmp_dir / "rng_state.pt")

        with open(tmp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Atomic rename: readers never see a half-written checkpoint.
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        tmp_dir.rename(checkpoint_dir)

        # Update the symlink that the resume logic reads first.
        self._update_symlink(checkpoint_dir)

        # Clean up old regular checkpoints (never touch emergency ones).
        if not emergency:
            self._cleanup_old_checkpoints()

        return checkpoint_dir

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> dict | None:
        """Load the most recent valid checkpoint and restore training state.

        Returns the checkpoint metadata dict, or ``None`` if no valid checkpoint
        was found.
        """
        checkpoint_dir = self._find_latest_checkpoint()
        if checkpoint_dir is None:
            return None

        # Load model weights.
        model.load_state_dict(
            torch.load(checkpoint_dir / "model.pt", weights_only=True, map_location="cpu")
        )

        # Load optimizer state.
        if optimizer is not None:
            optimizer.load_state_dict(
                torch.load(checkpoint_dir / "optimizer.pt", weights_only=True, map_location="cpu")
            )

        # Load scheduler state.
        if scheduler is not None and (checkpoint_dir / "scheduler.pt").exists():
            scheduler.load_state_dict(
                torch.load(checkpoint_dir / "scheduler.pt", weights_only=True, map_location="cpu")
            )

        # Load metadata.
        with open(checkpoint_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Restore RNG state.
        rng_state = torch.load(checkpoint_dir / "rng_state.pt", weights_only=False)
        torch.set_rng_state(rng_state["torch"])
        if rng_state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda"])

        print(f"Resumed from checkpoint: {checkpoint_dir}")
        return metadata

    def _find_latest_checkpoint(self) -> Path | None:
        """Resolve the latest checkpoint, falling back to directory scan."""
        # Fast path: read the symlink written by the last successful save.
        if self.latest_symlink.exists():
            raw = self.latest_symlink.read_text().strip()
            path = Path(raw)
            if not path.is_absolute():
                path = self.checkpoint_dir / path
            if self._is_valid_checkpoint(path):
                return path

        # Fallback: scan the checkpoint directory for valid checkpoints.
        checkpoints = []
        for path in self.checkpoint_dir.glob("checkpoint_step_*"):
            if path.is_dir() and self._is_valid_checkpoint(path):
                try:
                    step = int(path.name.split("_")[2])
                    checkpoints.append((step, path))
                except (IndexError, ValueError):
                    continue

        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]

    def _is_valid_checkpoint(self, checkpoint_dir: Path) -> bool:
        """A valid checkpoint must contain the required files."""
        return all((checkpoint_dir / f).exists() for f in self.REQUIRED_FILES)

    def _update_symlink(self, checkpoint_dir: Path) -> None:
        """Write a relative path into ``latest_checkpoint.txt``."""
        try:
            rel_path = checkpoint_dir.relative_to(self.checkpoint_dir)
        except ValueError:
            rel_path = checkpoint_dir
        self.latest_symlink.write_text(str(rel_path) + "\n")

    def _cleanup_old_checkpoints(self) -> None:
        """Keep the last ``keep_last_n`` regular checkpoints.

        Emergency checkpoints are never auto-deleted.
        """
        regular: list[tuple[int, Path]] = []
        for path in self.checkpoint_dir.glob("checkpoint_step_*"):
            if path.is_dir() and not path.name.endswith("_emergency"):
                try:
                    step = int(path.name.split("_")[2])
                    regular.append((step, path))
                except (IndexError, ValueError):
                    continue

        regular.sort(key=lambda x: x[0])

        for _, path in regular[:-self.keep_last_n]:
            shutil.rmtree(path)
