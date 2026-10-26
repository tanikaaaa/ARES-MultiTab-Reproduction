"""Controlled 4-tab BN sensitivity, not a reproduction of a collapsed epoch.

Three independent models are evaluated sequentially to bound memory. Both
training-mode probes keep dropout/DropPath active, as in reference training.
No optimizer, backward pass, or checkpoint write occurs.
"""
import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from WFlib.tools.evaluator import measurement


def select_indices(train_size, valid_size, subset_size, seed):
    if train_size < 64 or not 1 <= subset_size <= valid_size:
        raise ValueError("Insufficient samples")
    train_rng = np.random.RandomState(seed)
    valid_rng = np.random.RandomState(seed + 1)
    batch64 = train_rng.choice(train_size, 64, replace=False).tolist()
    validation = valid_rng.choice(valid_size, subset_size, replace=False).tolist()
    return batch64, batch64[:4], validation


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def bn_snapshot(model):
    return {name: {key: getattr(layer, key).detach().cpu().clone()
                   for key in ('running_mean', 'running_var', 'num_batches_tracked')}
            for name, layer in model.named_modules()
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) and layer.track_running_stats}


def summarize_tensor(value):
    value = value.double()
    return {'min': value.min().item(), 'max': value.max().item(), 'mean': value.mean().item(),
            'l2': torch.linalg.vector_norm(value).item()}


def bn_changes(before, after):
    changes = {}
    for name in before:
        entry = {'batches_before': int(before[name]['num_batches_tracked']),
                 'batches_after': int(after[name]['num_batches_tracked'])}
        for key in ('running_mean', 'running_var'):
            old, new = before[name][key].double(), after[name][key].double()
            delta = new - old
            entry[key] = {
                'before': summarize_tensor(old), 'after': summarize_tensor(new),
                'max_absolute_change': delta.abs().max().item(),
                'mean_absolute_change': delta.abs().mean().item(),
                'relative_l2_change': (torch.linalg.vector_norm(delta) /
                                       torch.linalg.vector_norm(old).clamp_min(1e-12)).item(),
            }
        changes[name] = entry
    return changes


def update_bn_once(model, features, seed):
    """One training-mode forward, verify only BN buffers may change."""
    initial = cpu_state(model)
    before = bn_snapshot(model)
    if not before:
        raise ValueError('Model has no tracked BatchNorm layers')
    seed_all(seed)
    model.train()
    with torch.no_grad():
        outputs = model(features)
        if not torch.isfinite(outputs).all().item():
            raise ValueError('Non-finite training-mode probe outputs')
    after = bn_snapshot(model)
    allowed = {f'{name}.{key}' if name else key for name in before for key in before[name]}
    for name, value in model.state_dict().items():
        if name not in allowed and not torch.equal(initial[name], value.detach().cpu()):
            raise AssertionError(f'Unexpected non-BN state change: {name}')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError('Unexpected parameter gradients')
    for name in before:
        if int(after[name]['num_batches_tracked'] - before[name]['num_batches_tracked']) != 1:
            raise AssertionError(f'Expected exactly one BN update: {name}')
        for key in ('running_mean', 'running_var'):
            if not torch.isfinite(after[name][key]).all():
                raise ValueError(f'Non-finite BN buffer: {name}.{key}')
    model.eval()
    return bn_changes(before, after)


def predict(model, loader, device):
    initial = cpu_state(model)
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for features, target in loader:
            output = model(features.to(device))
            if not torch.isfinite(output).all().item():
                raise ValueError('Non-finite validation logits')
            scores.append(torch.sigmoid(output).cpu().numpy())
            labels.append(target.numpy())
    for name, value in model.state_dict().items():
        if not torch.equal(initial[name], value.detach().cpu()):
            raise AssertionError(f'Evaluation changed state: {name}')
    return np.concatenate(labels), np.concatenate(scores)


def subset_metrics(y, scores):
    if y.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid predictions')
    if not np.isin(y, [0, 1]).all() or not np.all(y.sum(axis=1) == 4):
        raise ValueError('Expected four positive binary labels per sample')
    result = measurement(y, scores, ['P@4', 'AP@4'], num_tabs=4)
    supported = (y.sum(axis=0) > 0) & (y.sum(axis=0) < len(y))
    # Never silently substitute a different population for all-class macro AUC.
    result['AUC'] = measurement(y, scores, ['AUC'], num_tabs=4)['AUC'] if supported.all() else None
    result['auc_undefined_class_indices'] = np.flatnonzero(~supported).tolist()
    result['eligible_class_macro_AUC'] = (
        measurement(y[:, supported], scores[:, supported], ['AUC'], num_tabs=4)['AUC']
        if supported.any() else None)
    return result


def prediction_changes(baseline, changed):
    old_rank = np.argsort(baseline, axis=1)[:, -4:]
    new_rank = np.argsort(changed, axis=1)[:, -4:]
    delta = np.abs(changed.astype(np.float64) - baseline.astype(np.float64))
    return {
        'mean_absolute_probability_change': float(delta.mean()),
        'max_absolute_probability_change': float(delta.max()),
        'top1_changed_fraction': float(np.mean(old_rank[:, -1] != new_rank[:, -1])),
        'top4_set_changed_fraction': float(np.mean(np.any(
            np.sort(old_rank, axis=1) != np.sort(new_rank, axis=1), axis=1))),
    }


def write_report(path, report):
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bn-report-', dir=path.parent) as staging:
        temporary = Path(staging) / 'report.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.link(temporary, path)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--valid-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subset-size', type=int, default=1024, choices=range(512, 1025))
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = parser.parse_args()
    if os.path.lexists(args.output):
        parser.error('Output exists; refusing to overwrite')
    if not 0 <= args.seed < 2**32 - 1:
        parser.error('Seed must be in [0, 2**32-2]')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable; CPU may be selected explicitly')
    from WFlib.models.ARES import ARES
    from WFlib.tools.mtaf_dataset import MTAFDataset
    from exp.test_stream import validate_run_config
    validate_run_config(args.checkpoint, 4)
    source_paths = [args.checkpoint, args.checkpoint.parent / 'run_config.json']
    source_paths += [directory / name for directory in (args.train_dir, args.valid_dir)
                     for name in ('X.npy', 'y.npy', 'COMPLETE.json')]
    def source_metadata():
        return {str(path): (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
                for path in source_paths}
    metadata_before = source_metadata()
    checkpoint_hash = digest(args.checkpoint)
    start = time.perf_counter()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    report = {'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_hash,
              'train_dir': str(args.train_dir.resolve()), 'valid_dir': str(args.valid_dir.resolve()),
              'seed': args.seed, 'device': args.device, 'validation_batch_size': 64,
              'torch_version': torch.__version__, 'numpy_version': np.__version__,
              'design': 'Fresh independent checkpoint models; sequential execution; 4 training indices nested in 64; one no-grad train-mode forward; no optimizer',
              'limitations': ['Best checkpoint is not a collapsed-epoch state.',
                  'Single seeded random batch pair; representativeness is not guaranteed.',
                  'Dropout and DropPath remain active during probes. Equal seeds do not give identical masks at different batch sizes.',
                  'Differences reflect BN buffer sensitivity, not proof that batch size caused historical collapses.',
                  'A negative result does not rule out training-time BN instability.'],
              'conditions': {}}
    with ExitStack() as stack:
        train, valid = MTAFDataset(args.train_dir), None
        stack.callback(train.close)
        valid = MTAFDataset(args.valid_dir)
        stack.callback(valid.close)
        if len(train) != 46980 or len(valid) != 5220:
            raise ValueError('Expected 46980 training and 5220 validation rows')
        batch64, batch4, validation = select_indices(len(train), len(valid), args.subset_size, args.seed)
        report['indices'] = {'train_64': batch64, 'train_4': batch4, 'validation': validation}
        loader = DataLoader(Subset(valid, validation), batch_size=64, shuffle=False, num_workers=0)
        baseline = baseline_y = baseline_metrics = None
        for name, indices in (('baseline', []), ('batch64', batch64), ('batch4', batch4)):
            seed_all(args.seed)
            model = ARES(num_classes=100)
            model.load_state_dict(state, strict=True)
            for key, value in model.state_dict().items():
                if not torch.equal(value, state[key]):
                    raise AssertionError('Checkpoint initialization mismatch')
            model.to(args.device)
            condition = {'training_samples': len(indices)}
            if indices:
                features, labels = next(iter(DataLoader(Subset(train, indices), batch_size=len(indices), num_workers=0)))
                if not torch.all((labels == 0) | (labels == 1)) or not torch.all(labels.sum(1) == 4):
                    raise ValueError('Invalid training probe labels')
                condition['bn_changes'] = update_bn_once(model, features.to(args.device), args.seed + 2)
                condition['parameters_and_non_bn_buffers_unchanged'] = True
                del features, labels
            y, scores = predict(model, loader, args.device)
            condition['metrics'] = subset_metrics(y, scores)
            if baseline is None:
                baseline, baseline_y, baseline_metrics = scores, y, condition['metrics']
            else:
                np.testing.assert_array_equal(y, baseline_y)
                condition['prediction_changes'] = prediction_changes(baseline, scores)
                condition['metric_deltas'] = {
                    key: condition['metrics'][key] - baseline_metrics[key]
                    if baseline_metrics[key] is not None else None for key in ('AUC', 'P@4', 'AP@4')}
                condition['descriptive_sensitivity_flag'] = (
                    abs(condition['metric_deltas']['AP@4']) >= .01 or
                    condition['prediction_changes']['top4_set_changed_fraction'] >= .10)
            condition['evaluation_state_unchanged'] = True
            report['conditions'][name] = condition
            print(f"{name}: {condition['metrics']}", flush=True)
            del model
            gc.collect()
            if args.device == 'mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()
    if metadata_before != source_metadata() or checkpoint_hash != digest(args.checkpoint):
        raise AssertionError('Source metadata or checkpoint content changed')
    report['source_metadata_and_checkpoint_hash_unchanged'] = True
    report['sensitivity_flag_definition'] = 'Descriptive only, not statistical significance: >=0.01 absolute AP@4 change OR >=10% samples with changed top4 set'
    report['elapsed_seconds'] = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_process_rss_bytes'] = rss if sys.platform == 'darwin' else rss * 1024
    write_report(args.output, report)
    print(json.dumps({name: {key: value for key, value in condition.items() if key != 'bn_changes'}
                      for name, condition in report['conditions'].items()}, indent=2))
    print(f'Report: {args.output}. Sensitivity only; historical cause is unproven.')


if __name__ == '__main__':
    main()
