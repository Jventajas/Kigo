"""PyTorch Lightning data module for Kigo."""

from pathlib import Path

from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

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

    def _dataloader(self, dataset: MemmapDataset, shuffle: bool, drop_last: bool) -> DataLoader[Tensor]:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.platform.num_workers,
            pin_memory=self.platform.pin_memory,
            prefetch_factor=self.platform.prefetch_factor,
            persistent_workers=self.platform.persistent_workers,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader[Tensor]:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not set; call setup('fit') first.")
        return self._dataloader(self.train_dataset, shuffle=True, drop_last=True)

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
