# ARES Multi-Tab Website Fingerprinting — Reproduction

An independent B.Tech research reproduction of the released **ARES** baseline on closed-world 2-, 3-, 4-, and 5-tab website-fingerprinting datasets. Website fingerprinting infers visited websites from encrypted traffic patterns; the multi-tab setting requires identifying multiple websites within one mixed trace. ARES combines multi-channel temporal aggregation features (MTAF), local traffic profiling, and attention-based modeling for multi-label prediction.

This repository is a separate copy of [FIND-Lab's Website-Fingerprinting-Library (WFlib)](https://github.com/FIND-Lab/Website-Fingerprinting-Library), based on upstream commit `537ec8e`. The ARES architecture, reference feature calculations, training utilities, and evaluation formulas are upstream work. The reproduction adds memory-safe extraction and loading, streaming experiment entry points, validation diagnostics, and consolidated experiment records. It does **not** claim a new model or exact reproduction of published scores. Other upstream models and scripts remain included for provenance.

## Research and data sources

- Xinhao Deng et al., **Toward Robust Multi-Tab Website Fingerprinting**, IEEE Transactions on Networking, 2026. [DOI: 10.1109/TON.2026.3666721](https://doi.org/10.1109/TON.2026.3666721). The [accessible manuscript](https://arxiv.org/html/2501.12622v1) provides the reviewed metric equations and Table II comparison values.
- Original ARES work: **Robust Multi-tab Website Fingerprinting Attacks in the Wild**, IEEE Symposium on Security and Privacy, 2023. See the [authors' multi-tab dataset repository](https://github.com/Xinhao-Deng/Multitab-WF-Datasets) for research and dataset provenance.
- WFlib dataset distribution: [Zenodo record 13732130](https://zenodo.org/records/13732130). Obtain the official closed-world multi-tab datasets from the upstream sources and follow their usage terms; data is not redistributed here.
- WFlib attribution: Xinhao Deng, Qi Li, and Ke Xu, **Robust and Reliable Early-Stage Website Fingerprinting Attacks via Spatial-Temporal Distribution Analysis**, ACM CCS, 2024. Cite both the original ARES research and WFlib when using their implementation.

## Experimental scope and representation

Each tab count uses 58,000 traces and 100 monitored website classes, with exactly `k` positive labels per trace. The split is **46,980 training / 5,220 validation / 5,800 test** samples. The split script uses seed 2024, a 90%/10% train/test split followed by a 90%/10% train/validation split, without stratification for these multi-label experiments.

`exp/dataset_process/gen_mtaf_stream.py` calls the unchanged reference `process_MTAF()` and `agg_interval2()`:

- Signed timestamps preserve packet direction and packet order. Float32 timestamps are multiplied in place by 1000 for millisecond scaling.
- Aggregation uses 20 ms intervals, relative to the first packet's absolute timestamp, with 8,000 positions. The final position includes all remaining packets.
- Eight channels contain positive/negative packet counts, sums of successive same-direction timestamp differences, positive/negative burst counts, and mean positive/negative burst lengths.
- Features are stored as float32 `(N, 8, 8000)` arrays. Labels are preserved on disk and returned as float32 multi-hot tensors by `MTAFDataset`.
- Extraction writes disk-backed `X.npy`, `y.npy`, and a completion marker `COMPLETE.json`. Loading requires the marker and uses read-only memory maps with writable per-row tensor copies.

No timestamp sorting or cleaning is applied. The reference uses `np.searchsorted`, which assumes sorted absolute timestamps; preserving timestamp reversals preserves reference behavior rather than resolving that limitation. Feature arrays alone require about 13.83 GiB per complete 58,000-sample dataset, so lazy loading is essential on the reproduction machine.

## Recorded environment and training settings

The experiment machine was an Apple Silicon MacBook Air with 16 GiB unified memory (experiment context). The [consolidated JSON record](results/ares_closed_world_2to5_summary.json) records the following settings for all four runs:

| Setting | Value |
|---|---|
| Backend / framework | MPS / PyTorch 2.2.2 |
| NumPy | 1.26.4 |
| Model | Released ARES, 100 output classes |
| Epochs / batch size / seed | 30 / 64 / 2024 |
| Loss | MultiLabelSoftMarginLoss |
| Optimizer | AdamW, learning rate 0.002 |
| AdamW defaults | Weight decay 0.01; betas (0.9, 0.999); epsilon 1e-8 |
| Scheduler | StepLR, step_size=30, gamma=0.74 |
| Loader workers | 0 |
| Shuffle | Training: true; validation/test: false |
| Training drop_last | false |
| Checkpoint selection | Highest validation AP@k |
| Evaluation | Released common evaluator; AUC, P@k, AP@k |

Python 3.9.6 was reported for the experiment environment, but is not independently recorded in the consolidated run settings. The dependency set is not fully version-locked. Batch size 64 and `drop_last=False` differ from the released ARES shell configuration/loader (batch size 512 and `drop_last=True`). With 46,980 training rows, each epoch has 734 full batches and one four-sample batch.

## Held-out results and published comparison

Values below are copied from the saved summary; no inference was rerun to prepare this repository.

| Tabs | Reproduced AUC | Reproduced P@k | Reproduced AP@k | Published AUC | Published P@k | Published MAP@k |
|---|---|---|---|---|---|---|
| 2 | 0.9901 | 0.8344 | 0.8938 | 0.9950 | 0.9040 | 0.9380 |
| 3 | 0.9796 | 0.7853 | 0.8665 | 0.9900 | 0.8730 | 0.9160 |
| 4 | 0.9756 | 0.7831 | 0.8602 | 0.9890 | 0.8840 | 0.9220 |
| 5 | 0.9640 | 0.7162 | 0.8138 | 0.9880 | 0.8690 | 0.9090 |

The released evaluator computes P@k as the mean fraction of correct labels in the top `k`. Its AP@k is `sum(P@i for i=1..k) / k`, matching MAP@k in equations 9–10 of the reviewed manuscript. This is **not conventional relevance-weighted average precision**. The final journal equations were not independently verified. Repository AUC is macro-averaged across classes; published AUC averaging remains uncertain, so the table is descriptive rather than proof of identical evaluation protocols.

Best validation epochs are **18, 28, 27, and 24** for 2–5 tabs respectively, **zero-indexed and transcribed from training logs**, not recovered from checkpoint tensors. Selection uses validation data; the table reports held-out test results. These are single-seed runs with rounded metrics, without confidence intervals.

## Reproduction commands

Run from this repository root. Commands below are instructions, not steps executed during publication preparation. Create a local environment with a compatible Python version:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.2.2' 'numpy==1.26.4'
python -m pip install -e .
```

Other dependencies come from `setup.py`; their original versions are not fully recorded. MPS commands require a supported Apple Silicon/PyTorch installation. Download the official data separately. The examples use lowercase local filenames (`datasets/closed_2tab.npz` etc.); upstream downloads may use different capitalization.

For **new local artifacts only**, choose `k=2`, `3`, `4`, or `5`:

```sh
k=2
python -m exp.dataset_process.dataset_split \
  --dataset "closed_${k}tab" --use_stratify False

for split in train valid test; do
  python -m exp.dataset_process.gen_mtaf_stream \
    --input "datasets/closed_${k}tab/${split}.npz" \
    --output-dir "datasets/closed_${k}tab/mtaf_${split}_stream"
done
```

The upstream split script eagerly loads raw data and can overwrite existing split files: run it only when those outputs do not exist. Streaming extraction refuses existing output directories. Allow disk space for raw splits, temporary raw arrays, and generated features.

Optional bounded checks before a new run (each performs optimizer steps but saves no checkpoint):

```sh
python -m exp.train_stream --num-tabs "$k" --device mps \
  --batch-size 16 --smoke-test
python -m exp.train_stream --num-tabs "$k" --device mps \
  --batch-size 64 --benchmark-steps 5
```

Full training and held-out evaluation, when intentionally requested:

```sh
python -m exp.train_stream --num-tabs "$k" --device mps \
  --batch-size 64 --epochs 30 \
  --train-dir "datasets/closed_${k}tab/mtaf_train_stream" \
  --valid-dir "datasets/closed_${k}tab/mtaf_valid_stream" \
  --run-dir "checkpoints/ares_${k}tab_30epochs_bs64"

python -m exp.test_stream --num-tabs "$k" --device mps \
  --batch-size 64 \
  --test-dir "datasets/closed_${k}tab/mtaf_test_stream" \
  --checkpoint "checkpoints/ares_${k}tab_30epochs_bs64/best.pth" \
  --result "checkpoints/ares_${k}tab_30epochs_bs64/test_results.json"
```

Choose a new run directory for repeats. Training refuses an existing run directory; testing refuses an existing result and checks tab count against the adjacent `run_config.json`. Learning rate, loss, optimizer, and seed are fixed in the wrapper; the commands use the recorded settings. Do not tune experiments against held-out test scores.

## Records and diagnostics

- [`results/ares_closed_world_2to5_summary.json`](results/ares_closed_world_2to5_summary.json): saved run configurations, split metadata, metric provenance, and checkpoint SHA-256 identities.
- [`results/ares_closed_world_2to5_summary.md`](results/ares_closed_world_2to5_summary.md): readable results and parameter record.
- [`diagnostics/ares_2tab_epoch18_validation.json`](diagnostics/ares_2tab_epoch18_validation.json): validation ranking, class recall, and missed-pair diagnostics from `exp/diagnose_ares_valid.py` (2-tab-specific).
- [`diagnostics/ares_4tab_bn_sensitivity_seed2024.json`](diagnostics/ares_4tab_bn_sensitivity_seed2024.json): controlled BatchNorm-buffer sensitivity probe from `exp/diagnose_ares_bn.py`, using a fixed validation subset and training-mode forwards without optimizer updates.
- `exp/test_*.py` and `exp/dataset_process/test_*.py`: focused tests for the added wrappers, diagnostics, summaries, and data path. The original `exp/test.py` is an evaluation entry point, not a unit test.

`python -m exp.summarize_ares_runs` generates the summary only when all four local runs and split manifests exist and the summary output paths are absent. It refuses to overwrite the records included here. Saved records retain historical absolute paths as provenance; those paths are not portable and should not be followed to locate artifacts in another checkout. Checkpoint hashes identify artifacts, but saved test JSON alone does not cryptographically bind scores to an evaluation invocation.

Datasets, processed features, and checkpoint binaries are **excluded from GitHub** and are not included in this copy. Obtain data from its original source and train locally; the public records alone cannot rerun inference. Source, Markdown reports, JSON summaries, and intended diagnostic records remain trackable.

## Limitations and attribution

Recurring validation-score collapses remain unresolved. For example, 3-tab AP@3 fell from 0.7690 to 0.1588 before recovering, and 4-tab AP@4 fell from 0.8578 to 0.0767 and 0.0827 late in training (reported training-log observations). BatchNorm sensitivity, small final batches, optimization dynamics, and backend effects are hypotheses, not established causes. The diagnostic uses the best checkpoint rather than a collapsed-epoch state; it cannot establish the historical cause, and a negative probe does not rule out training-time instability. Stochastic training-mode layers also remain active during that probe.

The results fall short of the published reference values. Differences in batch size, final-batch handling, backend, incomplete environment locking, and unresolved training instability limit exact reproducibility. No architecture or feature-definition change is claimed as part of this reproduction.

The original implementation belongs to its upstream authors. The existing [MIT license](LICENSE), including the copyright notice for **Xinhao Deng@Tsinghua University (2024)**, is preserved unchanged. Retain that notice when redistributing the code, and credit the original research, WFlib, and dataset authors. This repository documents reproduction engineering and experimental observations, not independent authorship of upstream algorithms.
