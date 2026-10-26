"""ARES streaming evaluation. Run as: python -m exp.test_stream --help.

Loads the selected checkpoint, uses all test rows without shuffling, and reuses
the reference common evaluator. Existing result files are never overwritten.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader

from WFlib.models.ARES import ARES
from WFlib.tools import model_utils
from WFlib.tools.mtaf_dataset import MTAFDataset
from exp.train_stream import setup_device, tab_metrics, validate_labels


def validate_run_config(checkpoint, num_tabs):
    """Require the original run metadata beside the selected checkpoint."""
    path = checkpoint.parent / "run_config.json"
    with path.open() as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or type(config.get("num_tabs")) is not int or config["num_tabs"] != num_tabs:
        raise ValueError(f"Configured num_tabs={num_tabs} does not match {path}")
    if config.get("metrics") != tab_metrics(num_tabs) or config.get("save_metric") != f"AP@{num_tabs}":
        raise ValueError(f"Metrics in {path} do not match num_tabs={num_tabs}")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tabs", type=int, choices=(2, 3, 4, 5), default=2)
    parser.add_argument("--test-dir", type=Path, help="Default: datasets/closed_<num-tabs>tab/mtaf_test_stream")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True, help="New JSON result path")
    args = parser.parse_args()
    if args.test_dir is None:
        args.test_dir = Path(f"datasets/closed_{args.num_tabs}tab/mtaf_test_stream")
    if args.batch_size <= 0:
        parser.error("Batch size must be positive")
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    if os.path.lexists(args.result):
        parser.error("Result path already exists; refusing to overwrite")
    validate_run_config(args.checkpoint, args.num_tabs)
    device = setup_device(args.device)
    dataset = MTAFDataset(args.test_dir)
    try:
        validate_labels(args.test_dir, args.num_tabs)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=0)
        model = ARES(num_classes=100)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        model.to(device)
        args.result.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".ares-eval-", dir=args.result.parent) as staging:
            staged_result = Path(staging) / "result.json"
            model_utils.model_eval(
                model=model, test_iter=loader, valid_iter=None,
                eval_method="common", eval_metrics=tab_metrics(args.num_tabs),
                out_file=str(staged_result), num_classes=100,
                ckp_path=str(args.checkpoint.parent), scenario="Closed-world",
                num_tabs=args.num_tabs, device=device,
            )
            os.link(staged_result, args.result)
        print(f"Saved results: {args.result}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
