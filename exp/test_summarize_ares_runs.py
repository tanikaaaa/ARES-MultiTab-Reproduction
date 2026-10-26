"""Synthetic artifact tests; never read real runs/checkpoints/datasets."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from exp import summarize_ares_runs as summary


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ares-summary-tests-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        evaluator = self.root / 'WFlib/tools/evaluator.py'
        evaluator.parent.mkdir(parents=True)
        source = Path(__file__).resolve().parents[1] / 'WFlib/tools/evaluator.py'
        evaluator.write_text(source.read_text())
        for k in (2, 3, 4, 5):
            run = self.root / 'checkpoints' / f'ares_{k}tab_30epochs_bs64'
            run.mkdir(parents=True)
            (run / 'best.pth').write_bytes(f'synthetic checkpoint {k}'.encode())
            config = {'model': 'ARES', 'num_classes': 100, 'num_tabs': k,
                      'run_dir': str(run), 'checkpoint': str(run / 'best.pth'),
                      'metrics': ['AUC', f'P@{k}', f'AP@{k}'], 'save_metric': f'AP@{k}',
                      'seed': 2024, 'epochs': 30, 'batch_size': 64}
            for split, count in summary.COUNTS.items():
                directory = self.root / 'datasets' / f'closed_{k}tab' / f'mtaf_{split}_stream'
                directory.mkdir(parents=True)
                # Placeholders prove that tensor deserialization is not attempted.
                (directory / 'X.npy').write_bytes(b'not an array')
                (directory / 'y.npy').write_bytes(b'not an array')
                marker = {'complete': True, 'shape': [count, 8, 8000],
                          'label_shape': [count, 100], 'rows_written': count,
                          'dtype': 'float32', 'label_dtype': 'int8',
                          'features': 'X.npy', 'labels': 'y.npy'}
                (directory / 'COMPLETE.json').write_text(json.dumps(marker))
                if split != 'test':
                    config[f'{split}_dir'] = str(directory)
            (run / 'run_config.json').write_text(json.dumps(config))
            (run / 'test_results.json').write_text(json.dumps({'AUC': .9, f'P@{k}': .8, f'AP@{k}': .85}))
        self.run = self.root / 'checkpoints/ares_2tab_30epochs_bs64'

    def test_full_synthetic_summary_provenance_and_outputs(self):
        result = summary.build_summary(self.root)
        self.assertEqual([r['num_tabs'] for r in result['runs']], [2, 3, 4, 5])
        for run in result['runs']:
            self.assertIn('Transcribed from training logs', run['best_validation_epoch']['source'])
            self.assertEqual(run['best_validation_epoch']['value'], summary.EPOCHS_FROM_LOGS[run['num_tabs']])
            self.assertNotIn('optimizer', run['training_parameters'])
            self.assertIn('optimizer', run['missing_training_metadata'])
            self.assertEqual(run['splits']['test']['count'], 5800)
        paths = summary.write_outputs(self.root, result)
        self.assertEqual(json.loads(paths[0].read_text()), result)
        self.assertIn('Published MAP@k', paths[1].read_text())
        self.assertIn('BatchNorm has not been proven', paths[1].read_text())

    def test_metric_validation(self):
        path = self.run / 'test_results.json'
        for value in (float('nan'), float('inf'), -0.1, 1.1, True, '0.9', None):
            with self.subTest(value=value):
                path.write_text(json.dumps({'AUC': value, 'P@2': .8, 'AP@2': .85}))
                with self.assertRaises(ValueError):
                    summary.read_run(self.root, 2)
        path.write_text(json.dumps({'AUC': .9, 'P@2': .8, 'MAP@2': .85}))
        with self.assertRaises(ValueError):
            summary.read_run(self.root, 2)

    def test_tab_and_path_validation(self):
        path = self.run / 'run_config.json'
        initial = json.loads(path.read_text())
        for key, value in (('num_tabs', 3), ('num_tabs', True), ('train_dir', '/wrong/path'),
                           ('checkpoint', '/wrong/best.pth'), ('run_dir', '/wrong/run'),
                           ('save_metric', 'AP@3')):
            with self.subTest(key=key, value=value):
                path.write_text(json.dumps({**initial, key: value}))
                with self.assertRaises(ValueError):
                    summary.read_run(self.root, 2)

    def test_missing_files(self):
        for path in (self.run / 'best.pth', self.run / 'run_config.json', self.run / 'test_results.json',
                     self.root / 'datasets/closed_2tab/mtaf_test_stream/COMPLETE.json'):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    summary.read_run(self.root, 2)
                path.write_bytes(original)

    def test_overwrite_protection_for_either_output(self):
        result = summary.build_summary(self.root)
        paths = summary.output_paths(self.root)
        paths[0].parent.mkdir()
        for path in paths:
            path.write_text('existing content')
            with self.assertRaises(FileExistsError):
                summary.write_outputs(self.root, result)
            self.assertEqual(path.read_text(), 'existing content')
            self.assertFalse(next(p for p in paths if p != path).exists())
            path.unlink()

    def test_parsing_and_streaming_hash(self):
        path = self.root / 'input.json'
        for text in ('{', '[]', '{"a":1,"a":2}', '{"a":NaN}'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                summary.read_json(path)
        data = b'synthetic' * 300000  # More than two 1 MiB streaming blocks.
        path.write_bytes(data)
        self.assertEqual(summary.sha256(path), hashlib.sha256(data).hexdigest())

    def test_definition_drift_and_manifest_counts(self):
        path = self.root / 'WFlib/tools/evaluator.py'
        path.write_text(path.read_text().replace('res /= k', 'res /= (k + 1)'))
        with self.assertRaises(ValueError):
            summary.build_summary(self.root)
        marker = self.root / 'datasets/closed_2tab/mtaf_valid_stream/COMPLETE.json'
        obj = json.loads(marker.read_text())
        obj['rows_written'] = 1
        marker.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):
            summary.read_run(self.root, 2)


if __name__ == '__main__':
    unittest.main()
