"""Validation-only ARES diagnostics. Run from repository root with python -m.

Requires the recorded epoch-18 validation scores, not held-out test scores.
Website identities are column indices; pairs are unordered true label pairs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import rankdata

from WFlib.tools.evaluator import precision_at_k, average_precision_at_k


def correlation(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def analyze(y, scores, expected_p2, expected_ap2):
    if y.ndim != 2 or scores.shape != y.shape or len(y) == 0 or y.shape[1] < 2:
        raise ValueError("Expected matching nonempty sample-by-class matrices")
    if not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all() or not np.all(y.sum(1) == 2):
        raise ValueError("Expected finite scores and exactly two binary positive labels per row")
    # Same argsort and tie behavior as the released evaluator.
    ranked = np.argsort(scores, axis=1)[:, -2:][:, ::-1]
    correct = np.take_along_axis(y, ranked, axis=1)
    p1 = float(precision_at_k(y, scores, 1))
    p2 = float(precision_at_k(y, scores, 2))
    ap2 = float(average_precision_at_k(y, scores, 2))
    # Float32 target means round differently from the reference's scalar
    # divisions/float64 mean in NumPy 1.26; allow float32-scale roundoff only.
    np.testing.assert_allclose([correct[:, 0].mean(), correct.mean(), (p1 + p2) / 2],
                               [p1, p2, ap2], rtol=0, atol=1e-7)
    hits = np.zeros_like(y, dtype=bool)
    np.put_along_axis(hits, ranked, True, axis=1)
    frequency = y.sum(0).astype(int)
    recovered = (hits & (y == 1)).sum(0)
    supported = frequency > 0
    recall = recovered[supported] / frequency[supported]
    classes = [{"class_index": c, "support": int(frequency[c]),
                "top2_hits": int(recovered[c]), "misses": int(frequency[c] - recovered[c]),
                "recall_at_2": float(recovered[c] / frequency[c]) if frequency[c] else None}
               for c in range(y.shape[1])]
    pairs = defaultdict(lambda: {"support": 0, "samples_with_any_miss": 0,
                                 "samples_with_both_missed": 0, "misses_by_label": [0, 0]})
    for row in range(len(y)):
        pair = tuple(int(c) for c in np.flatnonzero(y[row]))
        missed = [not bool(hits[row, c]) for c in pair]
        entry = pairs[pair]
        entry["support"] += 1
        entry["samples_with_any_miss"] += int(any(missed))
        entry["samples_with_both_missed"] += int(all(missed))
        for j in range(2):
            entry["misses_by_label"][j] += int(missed[j])
    pair_rows = [{"true_label_pair": list(pair), **entry,
                  "any_miss_rate": entry["samples_with_any_miss"] / entry["support"]}
                 for pair, entry in pairs.items() if entry["samples_with_any_miss"]]
    pair_rows.sort(key=lambda e: (-e["samples_with_any_miss"], -sum(e["misses_by_label"]), e["true_label_pair"]))
    checks = {}
    for name, value, expected in (("P@2", p2, expected_p2), ("AP@2", ap2, expected_ap2)):
        if not np.isfinite(expected) or not 0 <= expected <= 1:
            raise ValueError("Expected validation metrics must lie in [0,1]")
        checks[name] = {"expected_reported": expected, "actual_unrounded": value,
                        "absolute_difference": abs(value - expected),
                        "rounding_tolerance": 0.00005,
                        "passed": bool(abs(value - expected) <= 0.00005 + 1e-12)}
    return {
        "samples": len(y), "classes": y.shape[1],
        "metrics": {"P@1": p1, "P@2": p2, "AP@2": ap2},
        "rank_correctness": {"first": float(correct[:, 0].mean()),
                             "second": float(correct[:, 1].mean()),
                             "first_correct_count": int(correct[:, 0].sum()),
                             "second_correct_count": int(correct[:, 1].sum())},
        "per_class": classes,
        "frequency_recall_relationship": {
            "supported_classes": int(supported.sum()),
            "pearson": correlation(frequency[supported], recall),
            "spearman": correlation(rankdata(frequency[supported]), rankdata(recall)),
            "note": "Unweighted class-level association, not causation; absent classes excluded; undefined correlations are null",
        },
        "frequently_missed_true_pairs": pair_rows[:20],
        "pair_definition": "Top 20 unordered true label pairs by number of samples with >=1 true label missing from top two; misses_by_label follows pair order",
        "epoch18_consistency": checks,
        "consistency_passed": all(check["passed"] for check in checks.values()),
    }


def write_report(path, report):
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publication without replacing an existing result, including races.
    with tempfile.TemporaryDirectory(prefix=".ares-diagnostic-", dir=path.parent) as staging:
        staged = Path(staging) / "report.json"
        staged.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.link(staged, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--valid-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-p2", required=True, type=float)
    parser.add_argument("--expected-ap2", required=True, type=float)
    args = parser.parse_args()
    if os.path.lexists(args.output):
        parser.error("Output already exists; refusing to overwrite")
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in (args.expected_p2, args.expected_ap2)):
        parser.error("Expected validation metrics must lie in [0,1]")
    # Fixed to the validated run's device and batch size; no optimizer exists.
    import torch
    from torch.utils.data import DataLoader
    from WFlib.models.ARES import ARES
    from WFlib.tools.mtaf_dataset import MTAFDataset
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    dataset = MTAFDataset(args.valid_dir)
    try:
        if dataset.feature_shape != (5220, 8, 8000) or dataset.label_shape != (5220, 100):
            raise ValueError("Expected the 5220-row validation split")
        checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        model = ARES(num_classes=100)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        model.to("mps")
        model.eval()
        loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=0)
        scores, labels = [], []
        with torch.no_grad():
            for features, target in loader:
                scores.append(torch.sigmoid(model(features.to("mps"))).cpu().numpy())
                labels.append(target.numpy())
        report = analyze(np.concatenate(labels), np.concatenate(scores), args.expected_p2, args.expected_ap2)
        report["provenance"] = {"split": "validation", "validation_directory": str(args.valid_dir.resolve()),
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
            "reference_epoch": 18, "device": "mps", "batch_size": 64,
            "torch_version": torch.__version__, "numpy_version": np.__version__}
    finally:
        dataset.close()
    write_report(args.output, report)
    print("Validation only: 5220 samples; batch size 64; MPS")
    print(" | ".join(f"{key}={value:.6f}" for key, value in report["metrics"].items()))
    print(f"First-rank correctness={report['rank_correctness']['first']:.6f}; "
          f"second-rank correctness={report['rank_correctness']['second']:.6f}")
    print("Frequency/recall:", report["frequency_recall_relationship"])
    print("Most frequently missed true pairs:", report["frequently_missed_true_pairs"][:5])
    print("Epoch-18 rounding checks:", "PASS" if report["consistency_passed"] else "FAIL")
    print(f"Report: {args.output}")
    if not report["consistency_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
