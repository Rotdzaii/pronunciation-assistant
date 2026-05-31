# L2-ARCTIC Vietnamese Speaker-Disjoint Context Experiment

## Purpose

This experiment tests whether a controlled audio context window improves phone-error classification for unseen Vietnamese speakers. The target weakness is the `addition` class, which remained low in Vietnamese speaker-disjoint evaluation after all-speaker training.

This experiment does not claim to fully model Vietnamese pronunciation characteristics. It only evaluates whether adding local acoustic context around the annotated phone-error segment improves the same three-class schema:

- `addition`
- `deletion`
- `substitution`

## Motivation

The previous Vietnamese speaker-disjoint CNN Attention baseline had weak addition generalization:

| Run | Mean macro F1 | Mean addition F1 | Mean deletion F1 | Mean substitution F1 |
|---|---:|---:|---:|---:|
| Baseline speaker-disjoint CNN Attention | 0.5022 +/- 0.0210 | 0.0881 +/- 0.0391 | 0.6881 +/- 0.0266 | 0.7303 +/- 0.0286 |
| Addition-focused sampler variant | 0.4715 +/- 0.0299 | 0.0958 +/- 0.0348 | 0.6347 +/- 0.0225 | 0.6839 +/- 0.0376 |

The sampler-only variant slightly improved addition F1 but reduced macro F1 and the other class scores, so it was not selected. A controlled context window is a different hypothesis: addition errors may need neighboring acoustic cues rather than only stronger class sampling.

Prior context-window attempts did not reliably improve overall performance, so this feature treats context as an experiment, not an assumed improvement.

## Dataset

Dataset:

`ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv`

The dataset uses all available L2-ARCTIC speakers with the same clean schema as the Vietnamese v2 dataset. Vietnamese-specific evaluation remains separate because the all-speaker data includes non-Vietnamese L1 groups.

## Fold Design

The protocol is leave-one-Vietnamese-speaker-out:

| Fold | Test speaker |
|---|---|
| 1 | HQTV |
| 2 | PNV |
| 3 | THV |
| 4 | TLV |

For each fold:

- Test set: all rows from the held-out Vietnamese speaker.
- Training set: original `train` rows from all other speakers, including non-Vietnamese speakers.
- Validation set: original `val` rows from all other speakers.
- The held-out Vietnamese speaker is excluded from both training and validation.

## Context Settings

The script supports these context modes:

| Mode | Context per side |
|---|---:|
| `original_segment` | 0.00 seconds |
| `context_0_05` | 0.05 seconds |
| `context_0_10` | 0.10 seconds |
| `context_0_15` | 0.15 seconds |

This run tested `context_0_10` only. Running all modes would require four times the training cost, and the goal of this feature was to validate one strong context candidate after sampler-only oversampling failed to provide a good tradeoff.

## Model And Config

Script:

`ai-training/scripts/run_l2_arctic_vietnamese_speaker_disjoint_cnn_attention_context.py`

Model:

- CNN Attention architecture reused from the speaker-disjoint baseline.
- Temporal attention pooling.
- Label order: `addition`, `deletion`, `substitution`.
- Inverse-frequency weighted sampler, same as the baseline speaker-disjoint run.
- Cross-entropy loss without extra class weighting.

Training config:

| Setting | Value |
|---|---:|
| Epochs | 12 |
| Batch size | 8 |
| Learning rate | 0.0001 |
| Random seed | 42 |
| Sample rate | 16000 |
| Mel bins | 64 |
| Max crop length | 1.0 second |
| Dropout | 0.2 |

Classifier confidence remains model confidence only. It is not a pronunciation correctness score.

## Per-Fold Metrics

| Context | Held-out speaker | Accuracy | Macro F1 | Addition F1 | Deletion F1 | Substitution F1 |
|---|---|---:|---:|---:|---:|---:|
| `context_0_10` | HQTV | 0.7158 | 0.5573 | 0.1676 | 0.7384 | 0.7659 |
| `context_0_10` | PNV | 0.6786 | 0.4982 | 0.0945 | 0.6353 | 0.7649 |
| `context_0_10` | THV | 0.6612 | 0.5218 | 0.1250 | 0.6763 | 0.7641 |
| `context_0_10` | TLV | 0.6186 | 0.4938 | 0.1115 | 0.6704 | 0.6996 |

Best fold by macro F1: HQTV.

Worst fold by macro F1: TLV.

## Aggregate Metrics

| Context | Mean accuracy | Mean macro F1 | Mean addition F1 | Mean deletion F1 | Mean substitution F1 |
|---|---:|---:|---:|---:|---:|
| `context_0_10` | 0.6685 +/- 0.0349 | 0.5178 +/- 0.0252 | 0.1246 +/- 0.0271 | 0.6801 +/- 0.0371 | 0.7486 +/- 0.0283 |

## Comparison

| Run | Mean macro F1 | Mean addition F1 | Mean deletion F1 | Mean substitution F1 |
|---|---:|---:|---:|---:|
| Baseline speaker-disjoint CNN Attention | 0.5022 +/- 0.0210 | 0.0881 +/- 0.0391 | 0.6881 +/- 0.0266 | 0.7303 +/- 0.0286 |
| Addition-focused sampler variant | 0.4715 +/- 0.0299 | 0.0958 +/- 0.0348 | 0.6347 +/- 0.0225 | 0.6839 +/- 0.0376 |
| Context-window CNN Attention `context_0_10` | 0.5178 +/- 0.0252 | 0.1246 +/- 0.0271 | 0.6801 +/- 0.0371 | 0.7486 +/- 0.0283 |

Compared with the baseline speaker-disjoint run:

- Macro F1 changed by `+0.0156`.
- Addition F1 changed by `+0.0365`.
- Deletion F1 changed by `-0.0080`.
- Substitution F1 changed by `+0.0183`.

Compared with the addition-focused sampler variant:

- Macro F1 improved by `+0.0463`.
- Addition F1 improved by `+0.0288`.
- Deletion F1 improved by `+0.0454`.
- Substitution F1 improved by `+0.0647`.

## Outputs

Generated evaluation outputs:

- `ai-training/datasets/l2-arctic/evaluation/vietnamese_speaker_disjoint_context_results.json`
- `ai-training/datasets/l2-arctic/evaluation/vietnamese_speaker_disjoint_context_per_fold.csv`
- `ai-training/datasets/l2-arctic/evaluation/vietnamese_speaker_disjoint_context_summary.csv`
- `ai-training/datasets/l2-arctic/evaluation/vietnamese_speaker_disjoint_context_per_class.csv`
- `ai-training/datasets/l2-arctic/evaluation/vietnamese_speaker_disjoint_context_confusion_matrices.csv`
- `ai-training/datasets/l2-arctic/evaluation/vietnamese_speaker_disjoint_context_misclassified_examples.csv`
- `ai-training/datasets/l2-arctic/evaluation/vietnamese_speaker_disjoint_context_comparison.csv`

## Checkpoint Note

Local checkpoints were generated under `ai-training/models/`:

- `l2_arctic_cnn_attention_speaker_disjoint_context_context_0_10_HQTV.pt`
- `l2_arctic_cnn_attention_speaker_disjoint_context_context_0_10_PNV.pt`
- `l2_arctic_cnn_attention_speaker_disjoint_context_context_0_10_THV.pt`
- `l2_arctic_cnn_attention_speaker_disjoint_context_context_0_10_TLV.pt`

These checkpoint files are local experiment artifacts and must not be committed.

## Limitations

- Only four Vietnamese speakers are available, so fold variance remains important.
- Only `context_0_10` was trained in this feature due runtime cost.
- The all-speaker training data includes non-Vietnamese L1 groups, so Vietnamese-specific conclusions must remain limited.
- Class imbalance remains severe, especially for `addition`.
- Model confidence is not a pronunciation score and should not be presented as correctness.

## Decision

The `context_0_10` experiment is a better candidate than the sampler-only addition-focused variant. It improved mean addition F1 and mean macro F1 over the speaker-disjoint baseline in this run.

Recommendation: keep `context_0_10` as the leading speaker-disjoint robustness candidate, but do not declare it final until at least one follow-up confirms whether the gain is stable across random seeds or across additional context settings.
