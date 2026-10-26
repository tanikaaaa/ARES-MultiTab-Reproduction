"""Synthetic-only diagnostics tests; never open real splits/checkpoints."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from exp.diagnose_ares_valid import analyze, correlation, write_report
from WFlib.tools.evaluator import precision_at_k, average_precision_at_k


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]])
        self.scores = np.array([[4, 3, 2, 1], [4, 1, 3, 2], [1, 0, 4, 3], [3, 1, 2, 4]])

    def test_hand_calculated_metrics_and_pairs(self):
        report = analyze(self.y, self.scores, .5, .625)
        self.assertEqual(report['metrics'], {'P@1': .75, 'P@2': .5, 'AP@2': .625})
        self.assertEqual(report['rank_correctness']['second'], .25)
        self.assertEqual([c['recall_at_2'] for c in report['per_class']], [2/3, 1/3, .5, None])
        pairs = {tuple(p['true_label_pair']): p for p in report['frequently_missed_true_pairs']}
        self.assertEqual(pairs[(0, 1)]['support'], 2)
        self.assertEqual(pairs[(0, 1)]['misses_by_label'], [0, 1])
        self.assertEqual(pairs[(1, 2)]['samples_with_both_missed'], 1)
        self.assertTrue(report['consistency_passed'])

    def test_rounding_and_mismatch(self):
        report = analyze(self.y[:3], self.scores[:3], .6667, .8333)
        self.assertTrue(report['consistency_passed'])
        self.assertFalse(analyze(self.y, self.scores, .6, .625)['consistency_passed'])

    def test_float32_reduction_roundoff_regression(self):
        # Synthetic counts reproduce the observed 2.43899466e-08 discrepancy;
        # no real validation data or model is loaded.
        n = 5220
        y = np.tile(np.array([1, 1, 0, 0], dtype=np.float32), (n, 1))
        scores = np.zeros((n, 4), dtype=np.float32)
        rows = np.arange(n)
        scores[rows, np.where(rows < 4974, 0, 2)] = 4
        scores[rows, np.where(rows < 3763, 1, 3)] = 3
        ranked = np.argsort(scores, axis=1)[:, -2:][:, ::-1]
        correct = np.take_along_axis(y, ranked, axis=1)
        p1 = float(precision_at_k(y, scores, 1))
        p2 = float(precision_at_k(y, scores, 2))
        ap2 = float(average_precision_at_k(y, scores, 2))
        self.assertEqual(correct[:, 0].mean().dtype, np.dtype('float32'))
        difference = abs(float(correct[:, 0].mean()) - p1)
        self.assertAlmostEqual(difference, 2.438994661790872e-08, delta=1e-15)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                [correct[:, 0].mean(), correct.mean(), (p1 + p2) / 2],
                [p1, p2, ap2], rtol=0, atol=1e-12)
        report = analyze(y, scores, .8369, .8949)
        self.assertEqual(report['metrics'], {'P@1': p1, 'P@2': p2, 'AP@2': ap2})
        self.assertTrue(report['consistency_passed'])
        for check in report['epoch18_consistency'].values():
            self.assertEqual(check['rounding_tolerance'], .00005)
        # The separate four-decimal checks must still reject larger mismatches.
        self.assertFalse(analyze(y, scores, .8370, .8950)['consistency_passed'])

    def test_ties_match_released_ranking(self):
        report = analyze(self.y, np.zeros_like(self.scores), .25, .125)
        self.assertTrue(report['consistency_passed'])
        self.assertEqual(report['rank_correctness']['first'], 0)
        self.assertEqual(report['rank_correctness']['second'], .5)

    def test_correlations_and_invalid_data(self):
        self.assertAlmostEqual(correlation([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(correlation([1, 1], [0, 1]))
        self.assertIsNone(correlation([], []))
        with self.assertRaises(ValueError):
            analyze(self.y, np.full(self.scores.shape, np.nan), .5, .625)
        with self.assertRaises(ValueError):
            analyze(np.zeros_like(self.y), self.scores, .5, .625)

    def test_json_and_no_overwrite(self):
        report = analyze(self.y, self.scores, .5, .625)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            write_report(path, report)
            before = path.read_bytes()
            self.assertEqual(json.loads(before), report)
            with self.assertRaises(FileExistsError):
                write_report(path, report)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
