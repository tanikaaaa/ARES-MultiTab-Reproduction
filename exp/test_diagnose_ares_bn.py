"""Synthetic CPU tests only; no ARES checkpoint or real data is opened."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from exp.diagnose_ares_bn import (select_indices, cpu_state, bn_snapshot, update_bn_once,
                                    predict, subset_metrics, prediction_changes, write_report)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2)
        self.dropout = torch.nn.Dropout(.1)
        self.head = torch.nn.Linear(2, 6)

    def forward(self, x):
        return self.head(self.dropout(self.bn(x)).mean(-1))


class BatchNormDiagnosticTests(unittest.TestCase):
    def test_indices_are_fixed_unique_and_nested(self):
        first = select_indices(100, 80, 64, 2024)
        self.assertEqual(first, select_indices(100, 80, 64, 2024))
        self.assertEqual(first[1], first[0][:4])
        self.assertEqual(len(set(first[0])), 64)
        self.assertEqual(len(set(first[2])), 64)
        with self.assertRaises(ValueError):
            select_indices(63, 80, 64, 2024)

    def test_bn_formula_and_unchanged_weights_for_both_sizes(self):
        torch.manual_seed(2024)
        baseline = ToyModel()
        original = cpu_state(baseline)
        original_bn = bn_snapshot(baseline)
        batch = torch.arange(64 * 2 * 8, dtype=torch.float32).reshape(64, 2, 8) / 100
        for size in (64, 4):
            model = copy.deepcopy(baseline)
            report = update_bn_once(model, batch[:size], 2026)
            torch.testing.assert_close(model.bn.running_mean, batch[:size].mean((0, 2)) * .1)
            torch.testing.assert_close(model.bn.running_var, .9 + batch[:size].var((0, 2), unbiased=True) * .1)
            self.assertEqual(report['bn']['batches_after'] - report['bn']['batches_before'], 1)
            self.assertGreater(report['bn']['running_mean']['max_absolute_change'], 0)
            for key, value in model.named_parameters():
                self.assertTrue(torch.equal(value, original[key]))
                self.assertIsNone(value.grad)
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(original[key], value))
        self.assertEqual(int(original_bn['bn']['num_batches_tracked']), 0)

    def test_eval_is_repeatable_and_does_not_change_state(self):
        torch.manual_seed(12)
        model = ToyModel()
        data = torch.randn(8, 2, 8)
        labels = torch.zeros(8, 6)
        labels[:, :4] = 1
        loader = DataLoader(TensorDataset(data, labels), batch_size=4)
        before = cpu_state(model)
        y, first = predict(model, loader, 'cpu')
        y2, second = predict(model, loader, 'cpu')
        np.testing.assert_array_equal(y, y2)
        np.testing.assert_array_equal(first, second)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertEqual(prediction_changes(first, second)['top4_set_changed_fraction'], 0)

    def test_metrics_auc_coverage_and_prediction_changes(self):
        y = np.zeros((6, 6), dtype=np.float32)
        for row in range(6):
            y[row, [(row + j) % 6 for j in range(4)]] = 1
        scores = y * .8 + .1
        report = subset_metrics(y, scores)
        self.assertEqual([report[key] for key in ('AUC', 'P@4', 'AP@4')], [1, 1, 1])
        self.assertEqual(report['auc_undefined_class_indices'], [])
        missing_class = subset_metrics(y[:1], scores[:1])
        self.assertIsNone(missing_class['AUC'])
        self.assertEqual(len(missing_class['auc_undefined_class_indices']), 6)
        delta = prediction_changes(scores, 1 - scores)
        self.assertEqual(delta['top4_set_changed_fraction'], 1)
        self.assertAlmostEqual(delta['mean_absolute_probability_change'], .8, places=6)
        with self.assertRaises(ValueError):
            subset_metrics(y, np.full_like(scores, np.nan))

    def test_atomic_report_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'diagnostic.json'
            report = {'AUC': None, 'conditions': {'batch4': {'parameters_unchanged': True}}}
            write_report(path, report)
            before = path.read_bytes()
            self.assertEqual(json.loads(before), report)
            with self.assertRaises(FileExistsError):
                write_report(path, {})
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
