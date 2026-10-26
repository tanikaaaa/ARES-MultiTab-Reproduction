"""Small real-row equivalence test; never extracts the complete dataset.

Run from the repository root:
    python -m exp.dataset_process.test_mtaf_stream \
        --input datasets/closed_2tab.npz
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np

from WFlib.tools.data_processor import process_MTAF
from . import gen_mtaf_stream


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_test(input_path):
    input_path = Path(input_path).resolve(strict=True)
    before = sha256(input_path)
    reference_path = Path(gen_mtaf_stream.__file__).resolve().parents[2] / "WFlib/tools/data_processor.py"
    reference_before = sha256(reference_path)
    with tempfile.TemporaryDirectory(prefix="mtaf-equivalence-") as temporary:
        root = Path(temporary)
        raw_path = root / "raw.npy"
        with zipfile.ZipFile(input_path) as archive:
            with archive.open("X.npy") as source, raw_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        try:
            reversal_index = None
            for index in range(raw.shape[0]):
                packets = np.trim_zeros(raw[index], "fb")
                if np.any(np.diff(np.abs(packets)) < 0):
                    reversal_index = index
                    break
            assert reversal_index is not None, "No reversal trace found"
            # Deliberately non-sorted source indices test output row ordering.
            indices = list(dict.fromkeys([raw.shape[0] - 1, reversal_index, 0, 1]))
            selected_X = np.array(raw[indices], copy=True)
        finally:
            raw._mmap.close()
        with np.load(input_path, allow_pickle=False) as archive:
            selected_y = archive["y"][indices].copy()
        unchanged_X = selected_X.copy()
        sample_path = root / "sample.npz"
        np.savez(sample_path, X=selected_X, y=selected_y)
        sample_before = sha256(sample_path)

        output = root / "output"
        manifest = gen_mtaf_stream.extract_stream(sample_path, output)
        assert manifest["rows_written"] == len(indices)
        assert json.loads((output / "COMPLETE.json").read_text()) == manifest
        actual = np.load(output / "X.npy", mmap_mode="r", allow_pickle=False)
        try:
            assert actual.shape == (len(indices), 8, 8000)
            assert actual.dtype == np.dtype("float32")
            for position in range(len(indices)):
                sequence = selected_X[position].copy()
                sequence *= 1000
                row_index, reference = process_MTAF(position, sequence, 20, 8000)
                assert row_index == position
                expected = reference.astype(np.float32)
                np.testing.assert_array_equal(actual[position], expected)
                assert actual[position].tobytes() == expected.tobytes()
                np.testing.assert_array_equal(actual[position].astype(np.float64), reference)
        finally:
            actual._mmap.close()
        actual_y = np.load(output / "y.npy", allow_pickle=False)
        assert actual_y.dtype == selected_y.dtype
        np.testing.assert_array_equal(actual_y, selected_y)
        np.testing.assert_array_equal(selected_X, unchanged_X)
        with zipfile.ZipFile(sample_path) as archive:
            assert archive.read("y.npy") == (output / "y.npy").read_bytes()
        assert sha256(sample_path) == sample_before
        assert not list(output.glob("*.partial"))
        assert not (output / ".raw-X.npy").exists()

        completed_hashes = {p.name: sha256(p) for p in output.iterdir()}
        try:
            gen_mtaf_stream.extract_stream(sample_path, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing completed output was accepted")
        assert completed_hashes == {p.name: sha256(p) for p in output.iterdir()}

        def interrupt(index, sequence, interval, max_len):
            if index == 1:
                raise RuntimeError("simulated interruption")
            return process_MTAF(index, sequence, interval, max_len)

        incomplete = root / "interrupted"
        with patch.object(gen_mtaf_stream, "process_MTAF", side_effect=interrupt):
            try:
                gen_mtaf_stream.extract_stream(sample_path, incomplete)
            except RuntimeError as error:
                assert str(error) == "simulated interruption"
            else:
                raise AssertionError("Expected simulated interruption")
        assert not (incomplete / "COMPLETE.json").exists()
        assert (incomplete / "X.npy.partial").exists()
        try:
            gen_mtaf_stream.extract_stream(sample_path, incomplete)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing partial output was accepted")

    after = sha256(input_path)
    assert before == after, "Raw dataset changed"
    assert reference_before == sha256(reference_path), "Reference implementation changed"
    print(json.dumps({
        "result": "PASS",
        "source_row_indices": indices,
        "timestamp_reversal_row": reversal_index,
        "shape": [len(indices), 8, 8000],
        "float32_bitwise_equivalence": True,
        "float64_roundtrip_equivalence": True,
        "labels_and_row_order": "PASS",
        "completion_and_overwrite_checks": "PASS",
        "interruption_check": "PASS",
        "raw_sha256_before": before,
        "raw_sha256_after": after,
        "reference_unchanged": True,
        "temporary_artifacts_removed": True,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    run_test(parser.parse_args().input)
