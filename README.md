# ARES: Multi-Tab Website Fingerprinting — Reproduction


## 1. Overview

This repository documents the reproduction and experimental evaluation of **ARES**, a deep-learning framework for multi-tab website fingerprinting.

The reproduction is based on the research paper:

**Toward Robust Multi-Tab Website Fingerprinting**  
Xinhao Deng et al., *IEEE Transactions on Networking*, 2026.

- [Research Paper](https://arxiv.org/abs/2501.12622)
- [Original Implementation — Website-Fingerprinting-Library](https://github.com/FIND-Lab/Website-Fingerprinting-Library)
- [Dataset — Zenodo](https://zenodo.org/records/13732130)

The objective is to reproduce the closed-world experiments involving **2, 3, 4, and 5 simultaneously active browser tabs**, evaluate the ARES model, and compare the reproduced results with those reported in the paper.

This project focuses on experimental reproduction and analysis. It does not claim to introduce a new website-fingerprinting architecture.

---

## 2. Research Background

### What is Website Fingerprinting?

Website fingerprinting (WF) is a traffic-analysis technique that attempts to identify visited websites using observable characteristics of encrypted network traffic.

Although encryption protects packet contents, an observer may still analyze traffic characteristics such as:

- Packet directions
- Packet timing
- Traffic bursts
- Packet sequences
- Temporal traffic patterns

These characteristics can reveal information about browsing activity.

### Multi-Tab Website Fingerprinting

Traditional website-fingerprinting approaches often assume that a user visits one website at a time.

In realistic browsing scenarios, however, multiple websites may be open simultaneously.

The resulting network traffic combines packets associated with different websites, making individual website identification more challenging.

ARES addresses this multi-label classification problem by learning representations from mixed browsing traffic.

### Research Objective

The central question investigated in this reproduction is:

> How effectively can ARES identify multiple simultaneously visited websites from their combined network traffic, and how does performance change as the number of active tabs increases?

---

## 3. Reproduction Scope

This project reproduces four closed-world experimental settings.

| Experiment | Active Tabs | Website Classes | Samples |
|---|---:|---:|---:|
| CW-2 | 2 | 100 | 58,000 |
| CW-3 | 3 | 100 | 58,000 |
| CW-4 | 4 | 100 | 58,000 |
| CW-5 | 5 | 100 | 58,000 |

Each sample has exactly `k` positive website labels, where `k` is the number of active tabs.

### Reproduction Highlights

- Acquired and validated the publicly released multi-tab datasets.
- Implemented memory-efficient, disk-backed MTAF preprocessing.
- Preserved the original ARES model implementation.
- Adapted the training and evaluation workflow for Apple Silicon MPS.
- Added configurable 2–5-tab experimental support.
- Trained four separate ARES models for 30 epochs each.
- Evaluated the best validation checkpoint on held-out test data.
- Generated consolidated experimental results and checkpoint hashes.
- Investigated recurring validation instability through a controlled BatchNorm diagnostic.

---

## 4. Dataset

The experiments use the publicly available multi-tab website-fingerprinting datasets.

**Dataset source:** [Zenodo — Website Fingerprinting Attack Evaluation](https://zenodo.org/records/13732130)

The following datasets were used:

```text
closed_2tab.npz
closed_3tab.npz
closed_4tab.npz
closed_5tab.npz
```

Each dataset contains:

| Array | Shape | Data Type |
|---|---|---|
| X | (58000, 10000) | float32 |
| y | (58000, 100) | int8 |

`X` contains signed packet-timing traces, while `y` contains multi-label website indicators.

### Dataset Splitting

The same splitting procedure was applied to all four datasets.

| Split | Samples | Percentage |
|---|---:|---:|
| Training | 46,980 | 81% |
| Validation | 5,220 | 9% |
| Testing | 5,800 | 10% |
| Total | 58,000 | 100% |

All splits were verified to contain 100 website classes and the expected number of positive labels per sample.

### Dataset Validation

The following checks were performed:

- Verified input and label shapes.
- Confirmed binary labels.
- Verified exactly `k` positive labels per sample.
- Checked numerical finiteness.
- Inspected packet directions and trace lengths.
- Identified timestamp-order irregularities.
- Confirmed that all dataset splits retained the expected labels.

Some traces contain small decreases in packet timestamps. These irregularities were documented, and the original data was preserved.

---

## 5. Methodology

### 5.1 MTAF Preprocessing

ARES uses a Multi-channel Time-based Aggregate Feature (MTAF) representation.

The original packet traces were transformed into fixed-size feature tensors.

**Preprocessing configuration:**

| Parameter | Value |
|---|---|
| Feature channels | 8 |
| Sequence length | 8,000 |
| Time interval | 20 ms |
| Output dtype | float32 |
| Processing | Disk-backed streaming |

Each processed sample has shape:

```text
(8, 8000)
```

The complete dataset is stored as NumPy arrays, allowing memory-efficient loading during training.

The preprocessing implementation was verified against the original `process_MTAF` function using exact numerical comparisons on selected samples.

### 5.2 Model

The experiments use the ARES implementation provided in the original Website-Fingerprinting-Library.

ARES processes MTAF traffic representations and produces scores for 100 website classes.

Because multiple websites may be present simultaneously, the task is formulated as multi-label classification.

The original model architecture was retained.

### 5.3 Training Configuration

| Parameter | Configuration |
|---|---|
| Model | ARES |
| Epochs | 30 |
| Batch size | 64 |
| Learning rate | 0.002 |
| Optimizer | AdamW |
| Loss | MultiLabelSoftMarginLoss |
| Scheduler | StepLR |
| Scheduler step size | 30 |
| Scheduler gamma | 0.74 |
| Random seed | 2024 |
| Device | Apple MPS |
| Training shuffle | True |
| Drop last batch | False |

The experiments used PyTorch 2.2.2 and NumPy 1.26.4.

The batch size and `drop_last` behavior differ from the released reference training configuration.

The model checkpoint was selected using the best validation AP@k score.

---

## 6. Experimental Results

All four models were trained for 30 epochs and evaluated on their respective held-out test sets.

### 6.1 Reproduced Results

| Tabs | AUC | P@k | AP@k | Best Epoch |
|---|---:|---:|---:|---:|
| 2 | 0.9901 | 0.8344 | 0.8938 | 18 |
| 3 | 0.9796 | 0.7853 | 0.8665 | 28 |
| 4 | 0.9756 | 0.7831 | 0.8602 | 27 |
| 5 | 0.9640 | 0.7162 | 0.8138 | 24 |

Best epochs are zero-indexed.

### 6.2 Comparison With Published Results

| Tabs | Published P@k | Reproduced P@k | Published MAP@k | Reproduced AP@k |
|---|---:|---:|---:|---:|
| 2 | 0.9040 | 0.8344 | 0.9380 | 0.8938 |
| 3 | 0.8730 | 0.7853 | 0.9160 | 0.8665 |
| 4 | 0.8840 | 0.7831 | 0.9220 | 0.8602 |
| 5 | 0.8690 | 0.7162 | 0.9090 | 0.8138 |

The reviewed repository AP@k and manuscript MAP@k definitions use the average of P@i over ranks 1 through k. This differs from conventional relevance-weighted average precision.

Published AUC aggregation was not independently established as identical to the reproduced implementation.

The reproduced results are consistently below the published values.

---

## 7. Results and Observations

### Observation 1 — Performance Decreases With More Tabs

ARES achieved its strongest results in the 2-tab setting.

As the number of simultaneously visited websites increased, identification performance generally decreased.

For example:

- 2-tab AP@2: **0.8938**
- 5-tab AP@5: **0.8138**

This illustrates the increased difficulty of separating website identities from more heavily mixed traffic.

### Observation 2 — Strong Ranking Performance

Across all four settings, the reproduced models achieved relatively high AUC and AP@k scores.

However, identifying the exact set of active websites remained more difficult, particularly for the 5-tab experiment.

### Observation 3 — Validation Instability

Some training epochs showed severe validation-score fluctuations despite progressively decreasing training loss.

For example, during 5-tab training:

| Epoch | Validation AP@5 |
|---|---:|
| 23 | 0.7974 |
| 24 | 0.8132 |
| 25 | 0.7136 |
| 26 | 0.1248 |
| 27 | 0.7655 |

A controlled BatchNorm sensitivity diagnostic was conducted to investigate one possible explanation.

The diagnostic demonstrated that small-batch updates could change predictions, but did not reproduce the severe validation collapse.

The underlying cause remains unresolved.

### Observation 4 — Best Checkpoint Selection

Selecting the checkpoint using the highest validation AP@k helped preserve the strongest model encountered during training.

Final test scores were computed using these selected checkpoints rather than the models from the last training epoch.

---

## 8. Project Structure

```text
ARES-MultiTab-Reproduction/
│
├── WFlib/
│   ├── models/
│   │   └── ARES.py
│   └── tools/
│       ├── data_processor.py
│       ├── evaluator.py
│       ├── model_utils.py
│       └── mtaf_dataset.py
│
├── exp/
│   ├── dataset_process/
│   │   ├── dataset_split.py
│   │   └── gen_mtaf_stream.py
│   ├── train_stream.py
│   ├── test_stream.py
│   ├── diagnose_ares_valid.py
│   ├── diagnose_ares_bn.py
│   └── summarize_ares_runs.py
│
├── diagnostics/
│   ├── ares_2tab_epoch18_validation.json
│   └── ares_4tab_bn_sensitivity_seed2024.json
│
├── results/
│   ├── ares_closed_world_2to5_summary.json
│   └── ares_closed_world_2to5_summary.md
│
├── scripts/
├── figures/
├── README.md
├── LICENSE
└── setup.py
```

The repository also retains supporting components of the original Website-Fingerprinting-Library.

---

## 9. Reproducibility

### 9.1 Environment

The experiments were performed using:

- Python 3.9.6
- PyTorch 2.2.2
- NumPy 1.26.4
- Apple Silicon MPS

Install the dependencies required by the upstream library and configure a compatible Python environment before running the experiments.

### 9.2 Dataset Preparation

Download the corresponding closed-world dataset from Zenodo and place it in:

```text
datasets/closed_2tab.npz
```

For other settings, replace `2` with `3`, `4`, or `5`.

Split the dataset:

```bash
python exp/dataset_process/dataset_split.py \
  --dataset closed_2tab \
  --use_stratify False
```

### 9.3 Generate MTAF Features

```bash
python -m exp.dataset_process.gen_mtaf_stream \
  --input datasets/closed_2tab/train.npz \
  --output-dir datasets/closed_2tab/mtaf_train_stream
```

Repeat for the validation and test splits using their respective input files and output directories.

### 9.4 Train ARES

```bash
python -m exp.train_stream \
  --num-tabs 2 \
  --train-dir datasets/closed_2tab/mtaf_train_stream \
  --valid-dir datasets/closed_2tab/mtaf_valid_stream \
  --batch-size 64 \
  --epochs 30 \
  --device mps \
  --run-dir checkpoints/ares_2tab_30epochs_bs64
```

### 9.5 Evaluate ARES

```bash
python -m exp.test_stream \
  --num-tabs 2 \
  --checkpoint checkpoints/ares_2tab_30epochs_bs64/best.pth \
  --test-dir datasets/closed_2tab/mtaf_test_stream \
  --result checkpoints/ares_2tab_30epochs_bs64/test_results.json \
  --batch-size 64 \
  --device mps
```

Replace the tab count and corresponding paths for the remaining experiments.

The custom scripts refuse to overwrite certain existing outputs. Use fresh output paths when repeating experiments.

---

## 10. Evaluation Metrics

The following metrics were used:

**AUC:** Measures the model's ability to rank positive website classes above negative classes.

**P@k:** Measures how many of the k highest-ranked predictions correspond to the k active websites.

**AP@k:** Uses the repository's rank-based definition:

\[
AP@k = \frac{1}{k}\sum_{i=1}^{k}P@i
\]

This definition was compared against the accessible manuscript's reported MAP@k formulation.

---

## 11. Reproducibility Records

The repository includes:

- Consolidated experimental results.
- Training configurations recorded in the results summary.
- Best-epoch information.
- Checkpoint SHA-256 hashes.
- Dataset validation and preprocessing scripts.
- Diagnostic reports.
- Synthetic unit tests.

Detailed records are available in:

```text
results/ares_closed_world_2to5_summary.md
results/ares_closed_world_2to5_summary.json
```

Large datasets, generated MTAF arrays, and trained model checkpoints are excluded from GitHub due to storage requirements.

The checkpoint hashes document the identities of the locally retained model artifacts.

---

## 12. Limitations

This reproduction has several limitations:

1. **Numerical differences:** Reproduced metrics do not exactly match the published results.
2. **Hardware differences:** Experiments were performed on Apple MPS rather than the original CUDA environment.
3. **Batch-size differences:** Batch size 64 was used instead of the released script's batch size 512.
4. **Training instability:** Some epochs experienced severe validation-score collapses whose cause remains unresolved.
5. **Limited scope:** Only the closed-world 2–5-tab experiments were reproduced.
6. **No repeated-seed evaluation:** Each configuration was trained once using seed 2024.
7. **Excluded model files:** Dataset archives and model checkpoints are not distributed through this repository.

The project therefore represents a documented baseline reproduction rather than an exact replication of every result in the paper.

---

## 13. References and Attribution

### Research Paper

Xinhao Deng et al.  
**Toward Robust Multi-Tab Website Fingerprinting.**  
*IEEE Transactions on Networking*, 2026.

- [Paper](https://arxiv.org/abs/2501.12622)
- [DOI](https://doi.org/10.1109/TON.2026.3666721)

### Original Implementation

FIND-Lab — Website-Fingerprinting-Library

https://github.com/FIND-Lab/Website-Fingerprinting-Library

### Dataset

Dataset for Website Fingerprinting Attack Evaluation

https://zenodo.org/records/13732130

### License

This repository retains code originating from the Website-Fingerprinting-Library.

The original MIT license and copyright notice are preserved in [LICENSE](LICENSE).

The ARES architecture and upstream implementation are credited to their original authors.

The reproduction workflow, memory-efficient data pipeline, training and evaluation adaptations, diagnostic scripts, and experimental analysis were developed as part of this reproduction project.

---

## 14. Project Status

**Completed:**

- [x] Closed-world 2-tab reproduction
- [x] Closed-world 3-tab reproduction
- [x] Closed-world 4-tab reproduction
- [x] Closed-world 5-tab reproduction
- [x] Dataset validation and MTAF preprocessing
- [x] Training and held-out evaluation
- [x] Initial training-stability investigation
- [x] Consolidated experimental results
- [x] GitHub documentation

**Future Work:**

- Investigate the source of validation instability.
- Evaluate additional random seeds.
- Examine sensitivity to batch size and training configuration.
- Extend experiments to open-world or other multi-tab scenarios.
- Compare ARES with other website-fingerprinting approaches.

