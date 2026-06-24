"""PyTorch Lightning data module for Kigo."""

from pathlib import Path
from typing import Any

from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from accelerator import Platform
from config import Config
from data import MemmapDataset


class KigoDataModule(LightningDataModule):
    """Builds train/val/test DataLoaders from memory-mapped uint32 token shards."""

    def __init__(self, config: Config, platform: Platform, data_dir: Path) -> None:
        super().__init__()
        self.config = config
        self.platform = platform
        self.data_dir = data_dir
        self.batch_size = config.batch_size
        self.train_dataset: MemmapDataset | None = None
        self.val_dataset: MemmapDataset | None = None
        self.test_dataset: MemmapDataset | None = None
        self._train_loader: StatefulDataLoader[Tensor] | None = None
        self._train_loader_state: dict[str, object] | None = None

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

    def _dataloader(self, dataset: MemmapDataset, shuffle: bool, drop_last: bool) -> StatefulDataLoader[Tensor]:
        # Workers are per-process; split across devices so N ranks don't oversubscribe the host CPU.
        workers = self.platform.num_workers // self.platform.device_count()
        return StatefulDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=self.platform.pin_memory,
            prefetch_factor=2 if workers > 0 else None,
            persistent_workers=workers > 0,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader[Tensor]:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not set; call setup('fit') first.")
        loader = self._dataloader(self.train_dataset, shuffle=True, drop_last=True)
        if self._train_loader_state is not None:
            loader.load_state_dict(self._train_loader_state)
            self._train_loader_state = None
        self._train_loader = loader
        return loader

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

    def state_dict(self) -> dict[str, Any]:
        # Lightning persists this into the checkpoint so a mid-epoch resume
        # continues from the same data position.
        if self._train_loader is None:
            return {}
        return {"train_loader": self._train_loader.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Stashed here; applied when train_dataloader() rebuilds the loader.
        self._train_loader_state = state_dict.get("train_loader")
