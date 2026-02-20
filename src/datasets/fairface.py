"""FairFace HDF5 dataset for loading vision-encoder features and demographic labels."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import h5py
import hdf5plugin  # noqa: F401 – registers Zstandard codec with h5py
import torch
from torch.utils.data import Dataset, get_worker_info

log = logging.getLogger(__name__)


def dataset_worker_init_fn(worker_id: int) -> None:
    """Open the HDF5 file once per DataLoader worker to avoid repeated I/O."""
    worker_info = get_worker_info()
    if worker_info is not None:
        dataset: FairFaceDataset = worker_info.dataset  # type: ignore[assignment]
        try:
            dataset.file_handle = h5py.File(dataset.hdf5_path, "r")
        except Exception:
            log.exception("Worker %d: failed to open %s", worker_id, dataset.hdf5_path)
            dataset.file_handle = None


class FairFaceDataset(Dataset):
    """PyTorch Dataset backed by an HDF5 file of FairFace features and labels.

    Args:
        hdf5_path: Path to the HDF5 file.
        mode: Group key inside the file (``'training'`` or ``'validation'``).
        return_labels: Which labels to return.
            * ``None`` or ``[]`` -- features only.
            * ``['all']`` -- every label key under the ``labels`` group.
            * A list of specific keys, e.g. ``['age', 'race']``.
    """

    def __init__(
        self,
        hdf5_path: str,
        mode: str = "training",
        return_labels: Optional[List[str]] = None,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.mode = mode
        self._requested_labels: List[str] = return_labels if return_labels is not None else []
        self.file_handle: Optional[h5py.File] = None
        self.label_keys_to_load: List[str] = []

        log.info("Initializing FairFaceDataset: path=%s, mode=%s, labels=%s", hdf5_path, mode, self._requested_labels)

        with h5py.File(self.hdf5_path, "r") as f:
            if self.mode not in f:
                raise ValueError(f"Mode '{self.mode}' not found in HDF5 groups: {list(f.keys())}")
            group = f[self.mode]
            if "encoded" not in group:
                raise ValueError(f"'encoded' dataset missing in group '{self.mode}'")
            self.length: int = group["encoded"].shape[0]

            if not self._requested_labels:
                self.label_keys_to_load = []
            elif "labels" not in group:
                log.warning("No 'labels' group in '%s'; returning features only.", self.mode)
                self.label_keys_to_load = []
            else:
                available = list(group["labels"].keys())
                if self._requested_labels == ["all"]:
                    self.label_keys_to_load = available
                else:
                    missing = [k for k in self._requested_labels if k not in available]
                    if missing:
                        raise ValueError(f"Requested labels {missing} not in {available} for mode '{self.mode}'")
                    self.label_keys_to_load = list(self._requested_labels)

        log.info("Dataset '%s': %d samples, labels=%s", self.mode, self.length, self.label_keys_to_load)

    def __len__(self) -> int:
        return self.length

    def _ensure_open(self) -> h5py.File:
        """Return an open file handle, opening lazily if needed.

        When ``num_workers > 0``, :func:`dataset_worker_init_fn` sets
        ``file_handle`` before any ``__getitem__`` call.  When
        ``num_workers == 0`` the init function is never invoked, so the
        first ``__getitem__`` call opens the file here and caches the
        handle for the lifetime of the dataset object.
        """
        if self.file_handle is None:
            self.file_handle = h5py.File(self.hdf5_path, "r")
        return self.file_handle

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        try:
            fh = self._ensure_open()
            group = fh[self.mode]
            features = torch.tensor(group["encoded"][idx], dtype=torch.float32)

            if not self.label_keys_to_load:
                return features

            labels_group = group["labels"]
            labels_dict: Dict[str, torch.Tensor] = {
                key: torch.tensor(labels_group[key][idx], dtype=torch.long) for key in self.label_keys_to_load
            }
            return features, labels_dict

        except Exception:
            log.exception("Error reading item %d from %s mode '%s'", idx, self.hdf5_path, self.mode)
            raise

    def __del__(self) -> None:
        if getattr(self, "file_handle", None) is not None:
            try:
                self.file_handle.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self.file_handle = None
