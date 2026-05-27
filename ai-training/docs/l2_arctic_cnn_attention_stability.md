# L2-ARCTIC CNN Attention Stability Check

## Purpose

This stability check verifies whether the CNN attention model consistently improves over CNN V2 across multiple random seeds. The previous CNN attention result was promising, but one strong run is not enough to select a model confidently because initialization, sampling order, and optimizer dynamics can change results.

## Setup

Dataset:

`ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv`

Seeds tested:

- 42
- 123
- 2026

Training setup:

- Original segment crop: `start_time` to `end_time`
- Log-mel spectrogram input
- CNN V2-style feature extractor
- Temporal attention pooling
- `WeightedRandomSampler` only
- Normal unweighted `CrossEntropyLoss`
- No weighted loss plus weighted sampler combination
- CUDA used when available

CNN V2 baseline:

- test macro F1: 0.4835
- test addition F1: 0.1240

Single-run CNN attention result before this check:

- test macro F1: 0.5105
- test addition F1: 0.1754

Confidence values remain classifier confidence, not pronunciation correctness.

## Per-Seed Results

| Seed | Best Epoch | Val Macro F1 | Val Addition F1 | Test Accuracy | Test Macro F1 | Test Addition F1 | Test Deletion F1 | Test Substitution F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1 | 0.5390 | 0.1856 | 0.6126 | 0.4913 | 0.1471 | 0.6484 | 0.6784 |
| 123 | 8 | 0.5659 | 0.2444 | 0.6517 | 0.5341 | 0.2264 | 0.6848 | 0.6910 |
| 2026 | 11 | 0.5374 | 0.2047 | 0.6096 | 0.5119 | 0.2078 | 0.6772 | 0.6507 |

Generated files:

- `ai-training/datasets/l2-arctic/evaluation/cnn_attention_stability_runs.csv`
- `ai-training/datasets/l2-arctic/evaluation/cnn_attention_stability_summary.csv`
- `ai-training/datasets/l2-arctic/evaluation/cnn_attention_stability_metrics.json`

Local checkpoints were generated under `ai-training/models/` and must not be committed.

## Mean and Standard Deviation

| Metric | Mean | Std |
| --- | ---: | ---: |
| test_accuracy | 0.6246 | 0.0235 |
| test_macro_f1 | 0.5124 | 0.0214 |
| test_weighted_f1 | 0.6449 | 0.0152 |
| test_addition_f1 | 0.1938 | 0.0415 |
| test_deletion_f1 | 0.6701 | 0.0192 |
| test_substitution_f1 | 0.6734 | 0.0206 |
| val_macro_f1 | 0.5474 | 0.0160 |
| val_addition_f1 | 0.2116 | 0.0300 |

The mean test macro F1 is 0.5124, which is above CNN V2's 0.4835. The mean test addition F1 is 0.1938, which is above CNN V2's 0.1240. All three seeds also individually exceed CNN V2 on test macro F1 and addition F1.

## Decision

CNN attention can replace CNN V2 as the current main model candidate for phone-level error type classification.

The stability check supports the single-run finding: CNN attention improves overall macro F1 while also improving the rare `addition` class. The result is not just one lucky seed within this 3-seed check.

## Limitations

- Only three seeds were tested.
- The dataset remains strongly imbalanced.
- Addition test support is small, with only 19 examples.
- The same train/validation/test split was used for all seeds, so this checks seed stability, not split stability.
- Checkpoint confidence is model confidence, not pronunciation correctness.
- The model still classifies known phone-level error segments; it is not a full end-to-end pronunciation assessment system.

## Next Steps

- Update `AI_EXPERIMENT_REPORT.md` to mark CNN attention as the selected current model.
- Keep CNN V2 in the experiment history as the previous baseline.
- Review addition false positives and false negatives from CNN attention.
- Consider a future split-level or speaker-holdout stability check if more data becomes available.
