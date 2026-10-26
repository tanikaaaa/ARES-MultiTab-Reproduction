"""Summarize existing ARES artifacts without importing torch or loading models.

Run from the repository root: python -m exp.summarize_ares_runs
Checkpoints are read only in 1 MiB blocks for SHA-256. Dataset tensors are not read.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

EPOCHS_FROM_LOGS = {2: 18, 3: 28, 4: 27, 5: 24}
PUBLISHED = {2: (.995, .904, .938), 3: (.990, .873, .916),
             4: (.989, .884, .922), 5: (.988, .869, .909)}
COUNTS = {'train': 46980, 'valid': 5220, 'test': 5800}
REVIEWED_FUNCTIONS = {
    'precision_at_k': 'a2f496e7358683ab9a5c84ba4b3754912e24cc2625d86e5faaa9707a04eff5a7',
    'average_precision_at_k': 'b69dbb7a3c440a790973286f1f1b1bcecb451b922159f3838b0f95ab5a286db9',
}
PARAMETERS = ('device', 'batch_size', 'epochs', 'seed', 'workers', 'loss', 'optimizer',
              'learning_rate', 'optimizer_defaults', 'scheduler', 'train_shuffle',
              'valid_shuffle', 'drop_last', 'torch_version', 'numpy_version', 'deviations')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f'{path}: nonfinite JSON constant {value}')
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'{path}: duplicate key {key}')
            result[key] = value
        return result
    with Path(path).open() as handle:
        result = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    if not isinstance(result, dict):
        raise ValueError(f'{path}: expected JSON object')
    return result


def check_metric_definitions(path):
    """Fail closed if the two manually reviewed ranking formulas have changed."""
    tree = ast.parse(Path(path).read_text())
    for name, expected in REVIEWED_FUNCTIONS.items():
        matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(matches) != 1 or hashlib.sha256(
                ast.dump(matches[0], include_attributes=False).encode()).hexdigest() != expected:
            raise ValueError(f'{name} differs from reviewed implementation; review compatibility before comparing')
    return {
        'implementation_file': str(Path(path).resolve()), 'implementation_sha256': sha256(path),
        'P@k': 'Mean over samples of true labels among top-k predictions divided by k',
        'AP@k': 'sum(P@i for i=1..k)/k; not conventional relevance-weighted average precision',
        'published_MAP@k': 'Equation (10): sum(P@i for i=1..k)/k',
        'ranking_definition_compatible': True,
        'verification_scope': 'Released functions checked against reviewed AST fingerprints; equations (9)-(10) checked in arXiv:2501.12622v1',
        'AUC': 'Released evaluator uses class-wise macro ROC AUC; published averaging convention remains unverified; no AUC delta asserted',
        'final_journal_equations_independently_verified': False,
    }


def check_path(root, configured, expected, label):
    if not isinstance(configured, str) or not configured:
        raise ValueError(f'Missing/invalid {label} path')
    actual = Path(configured)
    if not actual.is_absolute():
        actual = root / actual
    if actual.resolve() != expected.resolve():
        raise ValueError(f'{label} path does not match expected artifact: {expected}')


def read_run(root, k):
    run = root / 'checkpoints' / f'ares_{k}tab_30epochs_bs64'
    checkpoint, config_path, result_path = run / 'best.pth', run / 'run_config.json', run / 'test_results.json'
    for path in (checkpoint, config_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if checkpoint.stat().st_size == 0:
        raise ValueError(f'Empty checkpoint: {checkpoint}')
    config, metrics = read_json(config_path), read_json(result_path)
    names = ['AUC', f'P@{k}', f'AP@{k}']
    if type(config.get('num_tabs')) is not int or config['num_tabs'] != k:
        raise ValueError(f'{run}: incorrect num_tabs')
    if config.get('model') != 'ARES' or config.get('num_classes') != 100:
        raise ValueError(f'{run}: expected ARES with 100 classes')
    if config.get('metrics') != names or config.get('save_metric') != f'AP@{k}':
        raise ValueError(f'{run}: unexpected configured metrics or selection metric')
    if set(metrics) != set(names):
        raise ValueError(f'{result_path}: expected precisely {names}')
    for name, value in metrics.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{result_path}: invalid metric {name}')
    check_path(root, config.get('run_dir'), run, 'run_dir')
    check_path(root, config.get('checkpoint'), checkpoint, 'checkpoint')
    splits = {}
    for split, count in COUNTS.items():
        directory = root / 'datasets' / f'closed_{k}tab' / f'mtaf_{split}_stream'
        key = f'{split}_dir'
        if split != 'test' or key in config:
            check_path(root, config.get(key), directory, key)
        for name in ('COMPLETE.json', 'X.npy', 'y.npy'):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        marker = read_json(directory / 'COMPLETE.json')
        if (marker.get('complete') is not True or marker.get('shape') != [count, 8, 8000]
                or marker.get('label_shape') != [count, 100] or marker.get('rows_written') != count
                or marker.get('dtype') != 'float32' or marker.get('label_dtype') != 'int8'
                or marker.get('features') != 'X.npy' or marker.get('labels') != 'y.npy'):
            raise ValueError(f'{directory}: invalid completion metadata')
        splits[split] = {'count': count, 'path': str(directory.resolve()),
            'count_source': 'COMPLETE.json; tensor contents not revalidated',
            'path_source': 'run_config' if key in config else 'expected experiment layout; not saved test invocation',
            'manifest_sha256': sha256(directory / 'COMPLETE.json')}
    epoch = EPOCHS_FROM_LOGS[k]
    epoch_source = 'Transcribed from training logs, supplied by user; logs not independently parsed'
    if 'best_epoch' in config:
        if type(config['best_epoch']) is not int or config['best_epoch'] != epoch:
            raise ValueError('Recorded best_epoch conflicts with supplied zero-indexed log transcription')
        epoch_source += '; matching best_epoch also recorded in run_config.json'
    auc, precision, mapk = PUBLISHED[k]
    return {
        'num_tabs': k, 'run_directory': str(run.resolve()), 'splits': splits,
        'artifacts': {name: {'path': str(path.resolve()), 'sha256': sha256(path)}
                      for name, path in (('checkpoint', checkpoint), ('run_config', config_path), ('test_results', result_path))},
        'run_config': config, 'training_parameters': {key: config[key] for key in PARAMETERS if key in config},
        'missing_training_metadata': [key for key in PARAMETERS if key not in config],
        'best_validation_epoch': {'value': epoch, 'indexing': 'zero-based', 'source': epoch_source,
                                  'checkpoint_metadata_inspected': False},
        'test_metrics': metrics,
        'published_reference_metrics': {'AUC': auc, f'P@{k}': precision, f'MAP@{k}': mapk},
        'ranking_deltas_ours_minus_reference': {
            f'P@{k}': round(metrics[f'P@{k}'] - precision, 10),
            f'AP@{k}_minus_published_MAP@{k}': round(metrics[f'AP@{k}'] - mapk, 10)},
    }


def build_summary(root):
    root = Path(root).resolve()
    compatibility = check_metric_definitions(root / 'WFlib/tools/evaluator.py')
    return {
        'schema_version': 1,
        'reference': {
            'citation': 'Xinhao Deng et al., Toward Robust Multi-Tab Website Fingerprinting, IEEE Transactions on Networking, 2026. DOI: 10.1109/TON.2026.3666721',
            'journal_url': 'https://doi.org/10.1109/TON.2026.3666721',
            'accessible_manuscript': 'https://arxiv.org/html/2501.12622v1',
            'metrics_source': 'User-supplied published values, checked against accessible manuscript Table II',
        },
        'metric_compatibility': compatibility,
        'limitations': [
            'Recurring validation-score collapses remain unresolved; BatchNorm has not been proven responsible.',
            'Best epochs are user-supplied training-log transcriptions unless also recorded in configuration.',
            'Only saved rounded aggregate metrics are available; no uncertainty or significance is inferred.',
            'Checkpoint hashes identify current artifacts; result JSON does not independently bind scores to a checkpoint hash or test invocation.',
            'Metadata paths and manifests are checked; datasets and checkpoint tensors are not deserialized.',
        ],
        'runs': [read_run(root, k) for k in (2, 3, 4, 5)],
    }


def markdown(summary):
    lines = ['# ARES closed-world 2–5-tab results', '',
             '| Tabs | Train / valid / test | Best epoch (0-based) | Our AUC | Our P@k | Our AP@k | Published AUC | Published P@k | Published MAP@k |',
             '|---|---|---|---|---|---|---|---|---|']
    for run in summary['runs']:
        k, ours, ref = run['num_tabs'], run['test_metrics'], run['published_reference_metrics']
        counts = ' / '.join(str(run['splits'][split]['count']) for split in COUNTS)
        values = [ours['AUC'], ours[f'P@{k}'], ours[f'AP@{k}'], ref['AUC'], ref[f'P@{k}'], ref[f'MAP@{k}']]
        lines.append(f"| {k} | {counts} | {run['best_validation_epoch']['value']} | " + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', 'Epochs are transcribed from training logs as supplied by the user; see JSON for corroboration and provenance.', '',
              'Released AP@k and manuscript MAP@k both equal sum(P@i, i=1..k)/k (equations 9–10); this is not conventional relevance-weighted AP. Final journal equations were not independently verified. Published AUC averaging remains uncertain; no AUC delta is asserted.', '',
              f"Reference: {summary['reference']['citation']} [Accessible manuscript, Table II](https://arxiv.org/html/2501.12622v1).", '',
              '## Recorded run settings and checkpoint identities', '']
    for run in summary['runs']:
        lines += [f"### {run['num_tabs']}-tab", '',
                  f"Checkpoint SHA-256: `{run['artifacts']['checkpoint']['sha256']}`", '',
                  '```json', json.dumps(run['training_parameters'], indent=2, sort_keys=True), '```', '',
                  f"Missing training metadata: {', '.join(run['missing_training_metadata']) or 'none among the extracted fields'}.", '']
    lines += ['## Limitations', ''] + ['- ' + item for item in summary['limitations']]
    return '\n'.join(lines) + '\n'


def output_paths(root):
    base = Path(root) / 'results' / 'ares_closed_world_2to5_summary'
    return base.with_suffix('.json'), base.with_suffix('.md')


def require_new_outputs(paths):
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f'Refusing to overwrite {path}')


def write_outputs(root, summary):
    paths = output_paths(root)
    require_new_outputs(paths)
    texts = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n', markdown(summary))
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ares-summary-', dir=paths[0].parent) as staging:
        for index, (path, text) in enumerate(zip(paths, texts)):
            temporary = Path(staging) / str(index)
            temporary.write_text(text)
            os.link(temporary, path)  # Atomic individual publication; never overwrite.
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    require_new_outputs(output_paths(args.repo_root))
    summary = build_summary(args.repo_root)
    for path in write_outputs(args.repo_root, summary):
        print(path)


if __name__ == '__main__':
    main()
