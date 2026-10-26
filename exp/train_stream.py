"""ARES streaming training. Run as: python -m exp.train_stream --help.

Training uses drop_last=False to include every row: an explicit deviation from
the released loader's drop_last=True. Batch size 16 also differs from ARES.sh's
512. Architecture, learning rate, loss implementation and metrics are unchanged.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from WFlib.models.ARES import ARES
from WFlib.tools import model_utils
from WFlib.tools.mtaf_dataset import MTAFDataset


def tab_metrics(num_tabs):
    if type(num_tabs) is not int or num_tabs not in (2, 3, 4, 5):
        raise ValueError("num_tabs must be one of 2, 3, 4, 5")
    return ["AUC", f"P@{num_tabs}", f"AP@{num_tabs}"]


def validate_labels(directory, num_tabs):
    """Check labels in bounded chunks without reading any feature rows.

    Call after MTAFDataset has validated the completion marker and array headers.
    """
    tab_metrics(num_tabs)
    labels = np.load(Path(directory) / "y.npy", mmap_mode="r", allow_pickle=False)
    try:
        for start in range(0, len(labels), 1024):
            chunk = labels[start:start + 1024]
            valid = np.all((chunk == 0) | (chunk == 1), axis=1) & (chunk.sum(axis=1) == num_tabs)
            if not np.all(valid):
                row = start + int(np.flatnonzero(~valid)[0])
                raise ValueError(f"{directory}: row {row} must have exactly {num_tabs} binary positive labels")
    finally:
        labels._mmap.close()


def setup_device(name):
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return device


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_measurements(device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024}
    if device.type == "mps":
        result["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    return result


def benchmark(model, optimizer, loader, device, steps):
    """Exactly steps updates; step 1 is warm-up, with no extra warm-up updates.

Step timing includes device transfer, zero_grad, forward, loss, backward and
optimizer.step; data loading is timed separately. Memory is sampled after each
synchronized step, not an assertion about the peak within a GPU operation.
"""
    if len(loader.dataset) < steps * loader.batch_size:
        raise ValueError("Benchmark requires enough rows for all full batches")
    model.train()
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    iterator = iter(loader)
    records = []
    for step in range(1, steps + 1):
        loading_start = time.perf_counter()
        features, labels = next(iterator)
        loading_seconds = time.perf_counter() - loading_start
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad()
        outputs = model(features.to(device))
        if outputs.shape != (loader.batch_size, 100):
            raise ValueError(f"Unexpected output shape: {outputs.shape}")
        loss = criterion(outputs, labels.to(device))
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at benchmark step {step}")
        loss.backward()
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - start
        record = {"step": step, "loss": loss.item(), "step_seconds": elapsed,
                  "data_loading_seconds": loading_seconds, **memory_measurements(device)}
        records.append(record)
        print(json.dumps(record), flush=True)
        del outputs, loss, features, labels
    steady = [record["step_seconds"] for record in records[1:]]
    print(json.dumps({
        "benchmark": "PASS", "device": str(device), "batch_size": loader.batch_size,
        "optimizer_steps": steps, "checkpoint_saved": False,
        "steady_state_steps": list(range(2, steps + 1)),
        "steady_state_mean_seconds": float(np.mean(steady)),
        "steady_state_median_seconds": float(np.median(steady)),
        "steady_state_min_seconds": min(steady), "steady_state_max_seconds": max(steady),
        "steady_state_samples_per_second": loader.batch_size / float(np.mean(steady)),
        "max_observed_memory": {key: max(record[key] for record in records)
                                for key in memory_measurements(device)},
    }, indent=2))


def create_run_directory(args):
    """Exclusively create persistent output; never clean it up on interruption."""
    args.run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.run_dir / "best.pth"
    config = {
        "train_dir": str(args.train_dir.resolve()), "valid_dir": str(args.valid_dir.resolve()),
        "run_dir": str(args.run_dir.resolve()), "checkpoint": str(checkpoint.resolve()),
        "model": "ARES", "num_classes": 100, "num_tabs": args.num_tabs,
        "device": args.device, "batch_size": args.batch_size, "epochs": args.epochs,
        "seed": 2024, "workers": 0, "loss": "MultiLabelSoftMarginLoss",
        "optimizer": "AdamW", "learning_rate": 0.002,
        "optimizer_defaults": {"weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
        "metrics": tab_metrics(args.num_tabs), "save_metric": f"AP@{args.num_tabs}",
        "scheduler": {"name": "StepLR", "step_size": 30, "gamma": 0.74},
        "train_shuffle": True, "valid_shuffle": False, "drop_last": False,
        "deviations": ["Training drop_last=False; released loader uses True",
                       "Configurable batch size; released ARES.sh uses 512"],
        "torch_version": torch.__version__, "numpy_version": np.__version__,
    }
    with (args.run_dir / "run_config.json").open("x") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return checkpoint


def smoke_test(model, optimizer, loader, device):
    synchronize(device)
    start = time.perf_counter()
    features, labels = next(iter(loader))
    if features.shape != (16, 8, 8000) or labels.shape != (16, 100):
        raise ValueError("Smoke test requires one full batch of 16 real rows")
    model.train()
    optimizer.zero_grad()
    outputs = model(features.to(device))
    if outputs.shape != (16, 100):
        raise ValueError(f"Unexpected output shape: {outputs.shape}")
    loss = torch.nn.MultiLabelSoftMarginLoss()(outputs, labels.to(device))
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite smoke-test loss")
    loss.backward()
    optimizer.step()  # Exactly one step; no validation or checkpoint saving.
    synchronize(device)
    report = {
        "smoke_test": "PASS", "loss": loss.item(),
        "output_shape": list(outputs.shape), "device": str(device),
        "elapsed_seconds": time.perf_counter() - start,
        "optimizer_steps": 1, "checkpoint_saved": False,
    }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_process_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    if device.type == "mps":
        report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tabs", type=int, choices=(2, 3, 4, 5), default=2)
    parser.add_argument("--train-dir", type=Path, help="Default: datasets/closed_<num-tabs>tab/mtaf_train_stream")
    parser.add_argument("--valid-dir", type=Path, help="Default: datasets/closed_<num-tabs>tab/mtaf_valid_stream")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--run-dir", type=Path,
                        help="Unique NEW persistent run directory; contains best.pth and run_config.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--benchmark-steps", type=int, choices=range(2, 11),
                       help="Run only 2–10 real training batches; no checkpoint or validation")
    args = parser.parse_args()
    base = Path(f"datasets/closed_{args.num_tabs}tab")
    args.train_dir = args.train_dir if args.train_dir is not None else base / "mtaf_train_stream"
    args.valid_dir = args.valid_dir if args.valid_dir is not None else base / "mtaf_valid_stream"
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("Batch size and epochs must be positive")
    if args.smoke_test and args.batch_size != 16:
        parser.error("Smoke mode requires --batch-size 16")
    bounded = args.smoke_test or args.benchmark_steps is not None
    if not bounded and args.run_dir is None:
        parser.error("--run-dir is required for training")
    if bounded and args.run_dir is not None:
        parser.error("Smoke/benchmark modes do not accept --run-dir or save files")
    if args.run_dir is not None and os.path.lexists(args.run_dir):
        parser.error("Run directory already exists; refusing to reuse it")
    device = setup_device(args.device)
    with ExitStack() as stack:
        train = MTAFDataset(args.train_dir)
        stack.callback(train.close)
        validate_labels(args.train_dir, args.num_tabs)
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                  drop_last=False, num_workers=0)
        model = ARES(num_classes=100).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        if args.smoke_test:
            smoke_test(model, optimizer, train_loader, device)
            return
        if args.benchmark_steps is not None:
            benchmark(model, optimizer, train_loader, device, args.benchmark_steps)
            return
        valid = MTAFDataset(args.valid_dir)
        stack.callback(valid.close)
        validate_labels(args.valid_dir, args.num_tabs)
        valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False,
                                  drop_last=False, num_workers=0)
        checkpoint = create_run_directory(args)
        print("Training: shuffle=True, drop_last=False (released loader used True). "
              "StepLR: step_size=30, gamma=0.74; lr=0.002; seed=2024.")
        model_utils.model_train(
            model=model, optimizer=optimizer, train_iter=train_loader,
            valid_iter=valid_loader, loss_name="MultiLabelSoftMarginLoss",
            save_metric=f"AP@{args.num_tabs}", eval_metrics=tab_metrics(args.num_tabs),
            train_epochs=args.epochs, out_file=str(checkpoint),
            num_classes=100, num_tabs=args.num_tabs, device=device, lradj="StepLR",
        )
        if not checkpoint.is_file():
            raise RuntimeError("Reference training loop did not save a checkpoint")
        print(f"Saved best validation AP@{args.num_tabs} checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
