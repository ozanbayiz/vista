from __future__ import annotations

from typing import Callable, Optional

import lightning as L
from torch.utils.data import DataLoader

from src.datasets.fairface import FairFaceDataset, dataset_worker_init_fn


class FairFaceDataModule(L.LightningDataModule):
    def __init__(
        self,
        hdf5_path: str,
        return_labels: Optional[list[str]] = None,
        batch_size: int = 128,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._train: Optional[FairFaceDataset] = None
        self._val: Optional[FairFaceDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self._train = FairFaceDataset(
                hdf5_path=self.hparams.hdf5_path,
                mode="training",
                return_labels=self.hparams.return_labels,
            )
            self._val = FairFaceDataset(
                hdf5_path=self.hparams.hdf5_path,
                mode="validation",
                return_labels=self.hparams.return_labels,
            )
        if stage == "validate":
            self._val = FairFaceDataset(
                hdf5_path=self.hparams.hdf5_path,
                mode="validation",
                return_labels=self.hparams.return_labels,
            )

    def _worker_init(self) -> Optional[Callable]:
        if self.hparams.num_workers > 0:
            return dataset_worker_init_fn
        return None

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        return DataLoader(
            self._train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            worker_init_fn=self._worker_init(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None
        return DataLoader(
            self._val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            worker_init_fn=self._worker_init(),
        )
