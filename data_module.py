"""PyTorch Lightning data module for Kigo."""

from collections.abc import Iterator
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

from accelerator import Platform
from config import Config
from data import MemmapDataset


class StepSampler(Sampler[int]):
    """
    Deterministic shuffle keyed to (seed, epoch) with a resumable offset,
    so the data position is a pure function of global_step, independent of worker count and micro-batch size.
    """

    def __init__(self, samples_per_epoch: int, seed: int, consumed: int = 0) -> None:
        """
        Args:
            samples_per_epoch: Number of samples in one full pass (len of the train dataset).
            seed: Base seed; the per-epoch shuffle is keyed by ``seed + epoch``.
            consumed: Cumulative samples seen across all epochs so far (``step * global_batch_size``);
                split into the resumed epoch and the within-epoch offset.
        """
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        # The resume offset only skips into the epoch we resumed within; later epochs start fresh.
        self.start_epoch, self.start_offset = divmod(consumed, samples_per_epoch)
        self.epoch = self.start_epoch

    def set_epoch(self, epoch: int) -> None:
        # Lightning calls this at the start of each epoch; pure function of epoch
        # means worker copies all compute the same order without shared state.
        self.epoch = epoch

    def _offset(self) -> int:
        # Offset only applied when training is resumed, where start_epoch == epoch.
        return self.start_offset if self.epoch == self.start_epoch else 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.samples_per_epoch, generator=generator).tolist()
        yield from order[self._offset():]

    def __len__(self) -> int:
        return self.samples_per_epoch - self._offset()


class KigoDataModule(LightningDataModule):
    """Builds train/val/test DataLoaders from memory-mapped uint32 token shards."""

    def __init__(
            self,
            config: Config,
            platform: Platform,
            data_dir: Path,
            consumed: int = 0,
            seed: int = 1337,
    ) -> None:
        super().__init__()
        self.config = config
        self.platform = platform
        self.data_dir = data_dir
        self.batch_size = config.batch_size
        self.seed = seed
        self._consumed = consumed
        self.train_dataset: MemmapDataset | None = None
        self.val_dataset: MemmapDataset | None = None
        self.test_dataset: MemmapDataset | None = None

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._require_splits(("train", "val"))
            self.train_dataset = MemmapDataset(
                self.data_dir / "train", self.config.block_size
            )
            self.val_dataset = MemmapDataset(
                self.data_dir / "val", self.config.block_size
            )
        elif stage == "validate":
            self._require_splits(("val",))
            self.val_dataset = MemmapDataset(
                self.data_dir / "val", self.config.block_size
            )
        elif stage == "test":
            self._require_splits(("test",))
            self.test_dataset = MemmapDataset(
                self.data_dir / "test", self.config.block_size
            )
        elif stage == "predict":
            raise NotImplementedError(
                "The predict stage is not implemented. "
                "Add a prompt dataset (e.g., tokenized on the fly from text) "
                "and implement predict_step in the LightningModule to generate text."
            )

    def _require_splits(self, names: tuple[str, ...]) -> None:
        missing = [
            name for name in names if not (self.data_dir / name).exists()
        ]
        if missing:
            flags = " ".join(f"--{name}-tokens N" for name in missing)
            raise FileNotFoundError(
                f"Missing data splits: {', '.join(missing)} under {self.data_dir}.\n"
                "Run the dataset preparation script with the required splits:\n"
                f"  python scripts/prepare_dataset.py --dataset DATASET {flags}"
            )

    def _dataloader(
            self,
            dataset: MemmapDataset,
            shuffle: bool,
            drop_last: bool,
            sampler: Sampler[int] | None = None,
    ) -> DataLoader[Tensor]:
        # Workers are per-process; split across devices so N ranks don't oversubscribe the host CPU.
        workers = self.platform.num_workers // self.platform.device_count()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=workers,
            pin_memory=self.platform.pin_memory,
            prefetch_factor=2 if workers > 0 else None,
            persistent_workers=workers > 0,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader[Tensor]:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not set; call setup('fit') first.")
        # Position the shuffle by samples already consumed so a resume continues from the same spot on any hardware.
        sampler = StepSampler(len(self.train_dataset), self.seed, self._consumed)
        return self._dataloader(self.train_dataset, shuffle=False, drop_last=True, sampler=sampler)

    def val_dataloader(self) -> DataLoader[Tensor]:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not set; call setup('fit') or setup('validate') first.")
        return self._dataloader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader[Tensor]:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not set; call setup('test') first.")
        return self._dataloader(self.test_dataset, shuffle=False, drop_last=False)

    def predict_dataloader(self) -> DataLoader[Tensor]:
        raise NotImplementedError(
            "predict_dataloader is not implemented; "
            "add a prompt dataset and predict_step first."
        )
