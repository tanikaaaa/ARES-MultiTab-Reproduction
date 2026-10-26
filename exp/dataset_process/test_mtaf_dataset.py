"""Small lazy-loading tests; no extraction or training.

Run from the repository root:
    python -B -m unittest exp.dataset_process.test_mtaf_dataset -v
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from WFlib.tools.mtaf_dataset import MTAFDataset


DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets/closed_2tab"


class MTAFDatasetTests(unittest.TestCase):
    def test_real_splits_rows_and_batches(self):
        for split, rows in (("train", 46980), ("valid", 5220), ("test", 5800)):
            with self.subTest(split=split):
                directory = DATA_ROOT / f"mtaf_{split}_stream"
                dataset = MTAFDataset(directory)
                disk_X = np.load(directory / "X.npy", mmap_mode="r")
                disk_y = np.load(directory / "y.npy", mmap_mode="r")
                try:
                    self.assertEqual(len(dataset), rows)
                    self.assertEqual(dataset.feature_shape, (rows, 8, 8000))
                    self.assertEqual(dataset.label_shape, (rows, 100))
                    self.assertIsInstance(dataset._features, np.memmap)
                    self.assertEqual(dataset._features.mode, "r")
                    for index in (rows - 1, 3, 0, rows // 2):
                        features, labels = dataset[index]
                        self.assertEqual(features.dtype, torch.float32)
                        self.assertEqual(labels.dtype, torch.float32)
                        self.assertEqual(tuple(features.shape), (8, 8000))
                        self.assertEqual(tuple(labels.shape), (100,))
                        self.assertTrue(features.numpy().flags.writeable)
                        self.assertTrue(labels.numpy().flags.writeable)
                        np.testing.assert_array_equal(features.numpy(), disk_X[index])
                        np.testing.assert_array_equal(labels.numpy(), disk_y[index].astype(np.float32))
                        self.assertEqual(features.numpy().tobytes(), disk_X[index].tobytes())
                        # Mutating returned tensors must not touch mapped data.
                        features.fill_(-123)
                        labels.fill_(-123)
                        again_X, again_y = dataset[index]
                        np.testing.assert_array_equal(again_X.numpy(), disk_X[index])
                        np.testing.assert_array_equal(again_y.numpy(), disk_y[index])
                    loader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)
                    batch_X, batch_y = next(iter(loader))
                    self.assertEqual(tuple(batch_X.shape), (2, 8, 8000))
                    self.assertEqual(tuple(batch_y.shape), (2, 100))
                    self.assertEqual(batch_X.dtype, torch.float32)
                    self.assertEqual(batch_y.dtype, torch.float32)
                    np.testing.assert_array_equal(batch_X.numpy(), disk_X[:2])
                    np.testing.assert_array_equal(batch_y.numpy(), disk_y[:2])
                    for index in (-1, rows):
                        with self.assertRaises(IndexError):
                            dataset[index]
                finally:
                    dataset.close()
                    disk_X._mmap.close()
                    disk_y._mmap.close()

    def test_incomplete_and_invalid_manifests(self):
        valid = {
            "complete": True, "features": "X.npy", "labels": "y.npy",
            "shape": [2, 8, 8000], "dtype": "float32",
            "label_shape": [2, 100], "label_dtype": "int8",
            "rows_written": 2, "interval_ms": 20,
        }
        cases = {
            "missing_marker": None,
            "malformed_json": "{",
            "not_an_object": "[]",
            "incomplete": {**valid, "complete": False},
            "wrong_shape": {**valid, "shape": [2, 8, 7999]},
            "wrong_dtype": {**valid, "dtype": "float64"},
            "wrong_rows": {**valid, "rows_written": 3},
            "missing_rows": {key: value for key, value in valid.items() if key != "rows_written"},
            "wrong_labels": {**valid, "label_shape": [2, 99]},
            "wrong_label_dtype": {**valid, "label_dtype": "float32"},
            "unexpected_filename": {**valid, "features": "other.npy"},
            "noninteger_dimensions": {**valid, "shape": [2.0, 8, 8000]},
        }
        for name, manifest in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                np.save(directory / "X.npy", np.zeros((2, 8, 8000), dtype=np.float32))
                np.save(directory / "y.npy", np.zeros((2, 100), dtype=np.int8))
                if manifest is not None:
                    (directory / "COMPLETE.json").write_text(
                        manifest if isinstance(manifest, str) else json.dumps(manifest)
                    )
                with self.assertRaises(ValueError):
                    MTAFDataset(directory)

    def test_missing_files_and_array_metadata_mismatch(self):
        manifest = {
            "complete": True, "features": "X.npy", "labels": "y.npy",
            "shape": [2, 8, 8000], "dtype": "float32",
            "label_shape": [2, 100], "label_dtype": "int8",
            "rows_written": 2, "interval_ms": 20,
        }
        for case in ("missing_X", "missing_y", "X_dtype", "X_shape", "y_dtype", "y_shape", "truncated_X"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / "COMPLETE.json").write_text(json.dumps(manifest))
                if case != "missing_X":
                    np.save(directory / "X.npy", np.zeros(
                        (1 if case == "X_shape" else 2, 8, 8000),
                        dtype=np.float64 if case == "X_dtype" else np.float32,
                    ))
                if case != "missing_y":
                    np.save(directory / "y.npy", np.zeros(
                        (2, 99 if case == "y_shape" else 100),
                        dtype=np.float32 if case == "y_dtype" else np.int8,
                    ))
                if case == "truncated_X":
                    with (directory / "X.npy").open("r+b") as handle:
                        handle.truncate(200)
                with self.assertRaises(ValueError):
                    MTAFDataset(directory)


if __name__ == "__main__":
    unittest.main()
