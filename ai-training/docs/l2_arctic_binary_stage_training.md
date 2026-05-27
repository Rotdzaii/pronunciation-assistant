# L2-ARCTIC Binary-Stage Phone Error Classifiers

## Purpose

This experiment tests whether separating rare `addition` detection from the more common `deletion` versus `substitution` decision improves Vietnamese L2-ARCTIC phone-level error diagnosis.

The clean v2 dataset is strongly imbalanced:

| Class | Rows |
|---|---:|
| addition | about 198 |
| deletion | about 1566 |
| substitution | about 3155 |

The direct 3-class classifiers struggled most on `addition`. The sampler-only model confirmed that `WeightedRandomSampler` without weighted loss avoids over-balancing collapse, but addition F1 remained weak.

## Setup

Stage 1:

```txt
addition vs non_addition
```

Stage 2:

```txt
deletion vs substitution
```

Both stages use original `start_time` to `end_time` crops, log-mel spectrogram features, the same small CNN family as the existing sampler-only baseline, CUDA when available, and normal `CrossEntropyLoss`.

Stage 1 uses `WeightedRandomSampler` only because `addition` is rare. Stage 2 uses an unweighted loss and a sampler to keep deletion/substitution batches balanced without doubling the imbalance correction in the loss.

Model confidence is model confidence for the predicted class. It is not pronunciation correctness.

## Stage 1 Results

Stage 1 checkpoint:

```txt
ai-training/models/l2_arctic_addition_binary_cnn.pt
```

Validation:

| Metric | Value |
|---|---:|
| Accuracy | 0.8278 |
| Macro F1 | 0.5656 |
| Addition F1 | 0.2281 |
| Non-addition F1 | 0.9031 |

Test:

| Metric | Value |
|---|---:|
| Accuracy | 0.8048 |
| Macro F1 | 0.5000 |
| Addition F1 | 0.1096 |
| Non-addition F1 | 0.8904 |

## Stage 2 Results

Stage 2 checkpoint:

```txt
ai-training/models/l2_arctic_del_sub_binary_cnn.pt
```

Validation:

| Metric | Value |
|---|---:|
| Accuracy | 0.8046 |
| Macro F1 | 0.7850 |
| Deletion F1 | 0.7202 |
| Substitution F1 | 0.8498 |

Test:

| Metric | Value |
|---|---:|
| Accuracy | 0.7293 |
| Macro F1 | 0.7020 |
| Deletion F1 | 0.6119 |
| Substitution F1 | 0.7922 |

## Full Pipeline Results

The binary-stage pipeline first predicts `addition` versus `non_addition`. If Stage 1 predicts `addition`, the final output is `addition`. Otherwise, Stage 2 predicts `deletion` or `substitution`.

Validation:

| Metric | Value |
|---|---:|
| Accuracy | 0.6712 |
| Macro F1 | 0.5589 |
| Addition F1 | 0.2281 |
| Deletion F1 | 0.7139 |
| Substitution F1 | 0.7346 |

Test:

| Metric | Value |
|---|---:|
| Accuracy | 0.5946 |
| Macro F1 | 0.4662 |
| Addition F1 | 0.1096 |
| Deletion F1 | 0.6063 |
| Substitution F1 | 0.6828 |

## Comparison

The comparison table is saved to:

```txt
ai-training/datasets/l2-arctic/evaluation/binary_stage_comparison.csv
```

| Run | Test macro F1 | Test addition F1 | Test deletion F1 | Test substitution F1 |
|---|---:|---:|---:|---:|
| baseline | 0.4657 | 0.0000 | 0.6341 | 0.7631 |
| v2 | 0.4835 | 0.1240 | 0.6444 | 0.6821 |
| sampler_only | 0.4803 | 0.0541 | 0.6555 | 0.7315 |
| binary_stage_pipeline | 0.4662 | 0.1096 | 0.6063 | 0.6828 |

Ranking by test macro F1:

1. `v2`
2. `sampler_only`
3. `binary_stage_pipeline`
4. `baseline`

Ranking by test addition F1:

1. `v2`
2. `binary_stage_pipeline`
3. `sampler_only`
4. `baseline`

## Limitations

`addition` remains data-limited, so validation and test metrics are sensitive to a small number of examples. The two-stage setup can also compound errors: any true deletion/substitution sample incorrectly caught by Stage 1 as `addition` cannot be corrected by Stage 2.

The pipeline improved test addition F1 over sampler-only, but it reduced deletion/substitution F1 enough that overall macro F1 stayed below both V2 and sampler-only.

## Selection Decision

Do not select the binary-stage pipeline as the main model from this run. Keep it as an experiment because it confirms that explicit addition detection can raise addition F1 versus sampler-only, but the full pipeline does not improve test macro F1 and does not beat V2 on addition F1.

## Next Recommended Step

Collect or generate more reliable addition examples before adding more model complexity. The current bottleneck is likely minority-class data volume and label quality rather than CNN capacity.
