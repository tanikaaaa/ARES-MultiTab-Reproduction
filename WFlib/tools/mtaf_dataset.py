"""Read-only, row-wise loading of completed streaming MTAF datasets.

Use with DataLoader(..., num_workers=0) on macOS. Only the requested row is
copied into writable tensor storage; both source arrays remain memory-mapped.
"""

import json
import operator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MTAFDataset(Dataset):
    """Dataset for the X.npy, y.npy, COMPLETE.json streaming output format."""

    def __init__(self, directory):
        directory = Path(directory)
        marker = directory / "COMPLETE.json"
        if not marker.is_file():
            raise ValueError(f"Missing completion marker: {marker}")
        try:
            manifest = json.loads(marker.read_text())
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"Invalid completion manifest: {marker}") from error
        if not isinstance(manifest, dict) or manifest.get("complete") is not True:
            raise ValueError("Manifest must declare complete=true")
        rows = manifest.get("rows_written")
        if type(rows) is not int or rows <= 0:
            raise ValueError("Manifest rows_written must be a positive integer")
        expected = {
            "features": "X.npy",
            "labels": "y.npy",
            "shape": [rows, 8, 8000],
            "dtype": "float32",
            "label_shape": [rows, 100],
            "label_dtype": "int8",
            "interval_ms": 20,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Invalid manifest {key}: expected {value!r}")
        for key in ("shape", "label_shape"):
            if any(type(size) is not int for size in manifest[key]):
                raise ValueError(f"Manifest {key} must contain integer dimensions")
        for name in ("X.npy", "y.npy"):
            if not (directory / name).is_file():
                raise ValueError(f"Missing required array: {directory / name}")

        self._features = self._labels = None
        try:
            self._features = np.load(directory / "X.npy", mmap_mode="r", allow_pickle=False)
            self._labels = np.load(directory / "y.npy", mmap_mode="r", allow_pickle=False)
            for array, shape, dtype in (
                (self._features, expected["shape"], np.dtype("float32")),
                (self._labels, expected["label_shape"], np.dtype("int8")),
            ):
                if not isinstance(array, np.memmap):
                    raise ValueError("Expected memory-mappable .npy arrays")
                if array.shape != tuple(shape) or array.dtype != dtype:
                    raise ValueError("Array shape or dtype does not match the manifest")
        except Exception:
            self.close()
            raise
        self.feature_shape = self._features.shape
        self.label_shape = self._labels.shape

    def __len__(self):
        return self.feature_shape[0]

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._features is None:
            raise RuntimeError("Dataset has been closed")
        # Copies are confined to one row; tensors own writable NumPy storage.
        features = np.array(self._features[index], dtype=np.float32, copy=True, order="C")
        labels = np.array(self._labels[index], dtype=np.float32, copy=True, order="C")
        return torch.from_numpy(features), torch.from_numpy(labels)

    def close(self):
        """Release mappings; already-returned tensors remain valid."""
        for name in ("_features", "_labels"):
            array = getattr(self, name, None)
            if isinstance(array, np.memmap):
                array._mmap.close()
            elif array is not None and hasattr(array, "close"):
                array.close()
            setattr(self, name, None)
