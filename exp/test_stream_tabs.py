"""Synthetic wrapper tests; model execution and research loops are mocked."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from exp import train_stream, test_stream


def fixture(path, k):
    path.mkdir()
    np.save(path / 'X.npy', np.zeros((2, 8, 8000), dtype=np.float32))
    y = np.zeros((2, 100), dtype=np.int8)
    y[:, :k] = 1
    np.save(path / 'y.npy', y)
    (path / 'COMPLETE.json').write_text(json.dumps({
        'complete': True, 'features': 'X.npy', 'labels': 'y.npy',
        'shape': [2, 8, 8000], 'dtype': 'float32', 'rows_written': 2,
        'label_shape': [2, 100], 'label_dtype': 'int8', 'interval_ms': 20,
    }))


class TabWrapperTests(unittest.TestCase):
    def test_metrics_and_invalid_tab_counts(self):
        for k in (2, 3, 4, 5):
            self.assertEqual(train_stream.tab_metrics(k), ['AUC', f'P@{k}', f'AP@{k}'])
        for k in (1, 6, True, 2.0):
            with self.assertRaises(ValueError):
                train_stream.tab_metrics(k)

    def test_label_chunks_and_nonbinary_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            for k in (2, 3, 4, 5):
                y = np.zeros((1025, 100), dtype=np.int8)
                y[:, :k] = 1
                np.save(path / 'y.npy', y)
                # There is deliberately no X.npy: this check reads labels only.
                train_stream.validate_labels(path, k)
                with self.assertRaises(ValueError):
                    train_stream.validate_labels(path, 3 if k == 2 else 2)
                y[-1, 0], y[-1, 1] = 2, 0  # same sum, invalid binary encoding
                np.save(path / 'y.npy', y)
                with self.assertRaisesRegex(ValueError, 'row 1024'):
                    train_stream.validate_labels(path, k)

    def test_train_and_test_routing_all_tabs(self):
        for k in (2, 3, 4, 5):
            with self.subTest(k=k), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root = Path(temp)
                data, run, result = root / 'data', root / 'run', root / 'result.json'
                fixture(data, k)
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(redirect_stderr(io.StringIO()))
                stack.enter_context(patch.object(train_stream, 'setup_device', return_value='cpu'))
                stack.enter_context(patch.object(train_stream, 'ARES'))
                stack.enter_context(patch.object(train_stream.torch.optim, 'AdamW'))
                def fake_train(**kwargs):
                    Path(kwargs['out_file']).write_bytes(b'synthetic-checkpoint')
                training = stack.enter_context(patch.object(train_stream.model_utils, 'model_train', side_effect=fake_train))
                # Omit --num-tabs for k=2 to test backward-compatible default.
                tabs = [] if k == 2 else ['--num-tabs', str(k)]
                argv = ['train', *tabs, '--train-dir', str(data), '--valid-dir', str(data), '--run-dir', str(run)]
                with patch('sys.argv', argv):
                    train_stream.main()
                call = training.call_args.kwargs
                self.assertEqual(call['num_tabs'], k)
                self.assertEqual(call['eval_metrics'], train_stream.tab_metrics(k))
                self.assertEqual(call['save_metric'], f'AP@{k}')
                self.assertFalse(call['train_iter'].drop_last)
                config = json.loads((run / 'run_config.json').read_text())
                self.assertEqual(config['num_tabs'], k)
                self.assertEqual(config['learning_rate'], .002)
                self.assertEqual(config['seed'], 2024)
                self.assertEqual(config['train_dir'], str(data.resolve()))
                with patch('sys.argv', argv), self.assertRaises(SystemExit):
                    train_stream.main()  # existing run refused
                stack.enter_context(patch.object(test_stream, 'setup_device', return_value='cpu'))
                stack.enter_context(patch.object(test_stream, 'ARES'))
                stack.enter_context(patch.object(test_stream.torch, 'load', return_value={}))
                def fake_eval(**kwargs):
                    Path(kwargs['out_file']).write_text('{}')
                evaluation = stack.enter_context(patch.object(test_stream.model_utils, 'model_eval', side_effect=fake_eval))
                argv = ['test', *tabs, '--test-dir', str(data), '--checkpoint', str(run / 'best.pth'), '--result', str(result)]
                with patch('sys.argv', argv):
                    test_stream.main()
                self.assertEqual(evaluation.call_args.kwargs['num_tabs'], k)
                self.assertEqual(evaluation.call_args.kwargs['eval_metrics'], train_stream.tab_metrics(k))
                with patch('sys.argv', argv), self.assertRaises(SystemExit):
                    test_stream.main()  # existing result refused
                with self.assertRaises(ValueError):
                    test_stream.validate_run_config(run / 'best.pth', 3 if k == 2 else 2)

    def test_config_missing_and_inconsistent_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'best.pth'
            with self.assertRaises(FileNotFoundError):
                test_stream.validate_run_config(checkpoint, 2)
            (checkpoint.parent / 'run_config.json').write_text(json.dumps({
                'num_tabs': 3, 'metrics': ['AUC', 'P@2', 'AP@2'], 'save_metric': 'AP@2'}))
            with self.assertRaises(ValueError):
                test_stream.validate_run_config(checkpoint, 3)

    def test_bounded_modes_and_default_paths(self):
        for k in (2, 3, 4, 5):
            for mode, handler in ((['--smoke-test'], 'smoke_test'), (['--benchmark-steps', '5'], 'benchmark')):
                with self.subTest(k=k, mode=mode), ExitStack() as stack:
                    stack.enter_context(patch('sys.argv', ['train', '--num-tabs', str(k), *mode]))
                    stack.enter_context(patch.object(train_stream, 'setup_device', return_value='cpu'))
                    dataset = stack.enter_context(patch.object(train_stream, 'MTAFDataset'))
                    labels = stack.enter_context(patch.object(train_stream, 'validate_labels'))
                    stack.enter_context(patch.object(train_stream, 'DataLoader'))
                    stack.enter_context(patch.object(train_stream, 'ARES'))
                    stack.enter_context(patch.object(train_stream.torch.optim, 'AdamW'))
                    bounded = stack.enter_context(patch.object(train_stream, handler))
                    training = stack.enter_context(patch.object(train_stream.model_utils, 'model_train'))
                    create = stack.enter_context(patch.object(train_stream, 'create_run_directory'))
                    train_stream.main()
                    expected = Path(f'datasets/closed_{k}tab/mtaf_train_stream')
                    dataset.assert_called_once_with(expected)
                    labels.assert_called_once_with(expected, k)
                    bounded.assert_called_once()
                    training.assert_not_called()
                    create.assert_not_called()
                    dataset.return_value.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
