# ARES closed-world 2–5-tab results

| Tabs | Train / valid / test | Best epoch (0-based) | Our AUC | Our P@k | Our AP@k | Published AUC | Published P@k | Published MAP@k |
|---|---|---|---|---|---|---|---|---|
| 2 | 46980 / 5220 / 5800 | 18 | 0.9901 | 0.8344 | 0.8938 | 0.9950 | 0.9040 | 0.9380 |
| 3 | 46980 / 5220 / 5800 | 28 | 0.9796 | 0.7853 | 0.8665 | 0.9900 | 0.8730 | 0.9160 |
| 4 | 46980 / 5220 / 5800 | 27 | 0.9756 | 0.7831 | 0.8602 | 0.9890 | 0.8840 | 0.9220 |
| 5 | 46980 / 5220 / 5800 | 24 | 0.9640 | 0.7162 | 0.8138 | 0.9880 | 0.8690 | 0.9090 |

Epochs are transcribed from training logs as supplied by the user; see JSON for corroboration and provenance.

Released AP@k and manuscript MAP@k both equal sum(P@i, i=1..k)/k (equations 9–10); this is not conventional relevance-weighted AP. Final journal equations were not independently verified. Published AUC averaging remains uncertain; no AUC delta is asserted.

Reference: Xinhao Deng et al., Toward Robust Multi-Tab Website Fingerprinting, IEEE Transactions on Networking, 2026. DOI: 10.1109/TON.2026.3666721 [Accessible manuscript, Table II](https://arxiv.org/html/2501.12622v1).

## Recorded run settings and checkpoint identities

### 2-tab

Checkpoint SHA-256: `f444f0ac11a3f61a1bdd7667aeb1b4acb57ac6222e8382f31c1aee05003e86c1`

```json
{
  "batch_size": 64,
  "deviations": [
    "Training drop_last=False; released loader uses True",
    "Configurable batch size; released ARES.sh uses 512"
  ],
  "device": "mps",
  "drop_last": false,
  "epochs": 30,
  "learning_rate": 0.002,
  "loss": "MultiLabelSoftMarginLoss",
  "numpy_version": "1.26.4",
  "optimizer": "AdamW",
  "optimizer_defaults": {
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.01
  },
  "scheduler": {
    "gamma": 0.74,
    "name": "StepLR",
    "step_size": 30
  },
  "seed": 2024,
  "torch_version": "2.2.2",
  "train_shuffle": true,
  "valid_shuffle": false,
  "workers": 0
}
```

Missing training metadata: none among the extracted fields.

### 3-tab

Checkpoint SHA-256: `2d79342b0f52190bf8bf677a42125816f66238bcf1a9da64bbb5ac77ac7f43f9`

```json
{
  "batch_size": 64,
  "deviations": [
    "Training drop_last=False; released loader uses True",
    "Configurable batch size; released ARES.sh uses 512"
  ],
  "device": "mps",
  "drop_last": false,
  "epochs": 30,
  "learning_rate": 0.002,
  "loss": "MultiLabelSoftMarginLoss",
  "numpy_version": "1.26.4",
  "optimizer": "AdamW",
  "optimizer_defaults": {
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.01
  },
  "scheduler": {
    "gamma": 0.74,
    "name": "StepLR",
    "step_size": 30
  },
  "seed": 2024,
  "torch_version": "2.2.2",
  "train_shuffle": true,
  "valid_shuffle": false,
  "workers": 0
}
```

Missing training metadata: none among the extracted fields.

### 4-tab

Checkpoint SHA-256: `d1d204fae879aa7c3aa07c541f1aa4e078034df15ef28bd68231972b41a8db0a`

```json
{
  "batch_size": 64,
  "deviations": [
    "Training drop_last=False; released loader uses True",
    "Configurable batch size; released ARES.sh uses 512"
  ],
  "device": "mps",
  "drop_last": false,
  "epochs": 30,
  "learning_rate": 0.002,
  "loss": "MultiLabelSoftMarginLoss",
  "numpy_version": "1.26.4",
  "optimizer": "AdamW",
  "optimizer_defaults": {
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.01
  },
  "scheduler": {
    "gamma": 0.74,
    "name": "StepLR",
    "step_size": 30
  },
  "seed": 2024,
  "torch_version": "2.2.2",
  "train_shuffle": true,
  "valid_shuffle": false,
  "workers": 0
}
```

Missing training metadata: none among the extracted fields.

### 5-tab

Checkpoint SHA-256: `dcb540edbd4111a1f7e5cfd9f24f24cb14587fbbb58b79d0e7eb36f1670e5b02`

```json
{
  "batch_size": 64,
  "deviations": [
    "Training drop_last=False; released loader uses True",
    "Configurable batch size; released ARES.sh uses 512"
  ],
  "device": "mps",
  "drop_last": false,
  "epochs": 30,
  "learning_rate": 0.002,
  "loss": "MultiLabelSoftMarginLoss",
  "numpy_version": "1.26.4",
  "optimizer": "AdamW",
  "optimizer_defaults": {
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.01
  },
  "scheduler": {
    "gamma": 0.74,
    "name": "StepLR",
    "step_size": 30
  },
  "seed": 2024,
  "torch_version": "2.2.2",
  "train_shuffle": true,
  "valid_shuffle": false,
  "workers": 0
}
```

Missing training metadata: none among the extracted fields.

## Limitations

- Recurring validation-score collapses remain unresolved; BatchNorm has not been proven responsible.
- Best epochs are user-supplied training-log transcriptions unless also recorded in configuration.
- Only saved rounded aggregate metrics are available; no uncertainty or significance is inferred.
- Checkpoint hashes identify current artifacts; result JSON does not independently bind scores to a checkpoint hash or test invocation.
- Metadata paths and manifests are checked; datasets and checkpoint tensors are not deserialized.
