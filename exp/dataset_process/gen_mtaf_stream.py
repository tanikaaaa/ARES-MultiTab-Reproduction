"""Serial, bounded-memory MTAF extraction; only COMPLETE.json marks success.

Example (run from the repository root):
    python -m exp.dataset_process.gen_mtaf_stream \
        --input datasets/closed_2tab.npz --output-dir datasets/closed_2tab_mtaf

The output directory must not exist. Failed runs are left for inspection and
are never resumed or overwritten automatically. X.npy is float32, y.npy retains
the input dtype/values. Consumers must require COMPLETE.json before loading.
The ZIP's raw X.npy is streamed to temporary disk storage for memory mapping;
allow space for raw X plus the output. No input files are written.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np

from WFlib.tools.data_processor import process_MTAF


def _copy_member(archive, name, destination):
    with archive.open(name) as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())


def extract_stream(input_path, output_dir):
    """Extract all input rows in order, with one reference result in memory."""
    input_path = Path(input_path).resolve(strict=True)
    output_dir = Path(output_dir)
    # Exclusive creation also prevents concurrent writers and accidental reuse.
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / ".raw-X.npy"
    feature_path = output_dir / "X.npy.partial"
    label_path = output_dir / "y.npy.partial"
    raw = labels = features = None
    try:
        with zipfile.ZipFile(input_path) as archive:
            for name in ("X.npy", "y.npy"):
                if archive.namelist().count(name) != 1:
                    raise ValueError("Input must contain exactly one " + name)
            _copy_member(archive, "X.npy", raw_path)
            _copy_member(archive, "y.npy", label_path)
        raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        if raw.ndim != 2 or raw.dtype != np.dtype("float32"):
            raise ValueError("Expected a two-dimensional float32 X")
        if raw.shape[0] == 0 or labels.ndim == 0 or labels.shape[0] != raw.shape[0]:
            raise ValueError("Expected nonempty X and matching label rows")
        feature_shape = (raw.shape[0], 8, 8000)
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float32, shape=feature_shape
        )
        for index in range(raw.shape[0]):
            sequence = np.array(raw[index], copy=True)
            sequence *= 1000  # Preserve the reference float32 in-place scaling.
            returned_index, result = process_MTAF(index, sequence, 20, 8000)
            if returned_index != index:
                raise RuntimeError("Reference returned an unexpected row index")
            features[index] = result
            del result, sequence
            if (index + 1) % 128 == 0:
                features.flush()
        features.flush()
        manifest = {
            "complete": True,
            "source": str(input_path),
            "features": "X.npy",
            "labels": "y.npy",
            "shape": list(feature_shape),
            "dtype": "float32",
            "label_shape": list(labels.shape),
            "label_dtype": str(labels.dtype),
            "interval_ms": 20,
            "rows_written": raw.shape[0],
        }
    finally:
        # Close mappings even on failure; partial files have no completion marker.
        for array in (features, labels, raw):
            if array is not None:
                array._mmap.close()

    with feature_path.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(feature_path, output_dir / "X.npy")
    os.replace(label_path, output_dir / "y.npy")
    raw_path.unlink()
    marker = output_dir / "COMPLETE.json.partial"
    with marker.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(marker, output_dir / "COMPLETE.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = extract_stream(args.input, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
