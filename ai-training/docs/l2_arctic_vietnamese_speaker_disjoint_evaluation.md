# Vietnamese Speaker-Disjoint CNN Attention Evaluation

## Purpose

This experiment evaluates whether the L2-ARCTIC CNN Attention phone-error classifier generalizes to unseen Vietnamese speakers.

The previous all-speaker result used the original utterance split. That split can include the same speaker in train, validation, and test, so it may overestimate generalization to a new learner. Speaker-disjoint evaluation is stricter because the held-out Vietnamese speaker is absent from both training and validation.

## Dataset

Metadata:

`ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv`

Label schema:

- `addition`
- `deletion`
- `substitution`

Vietnamese speakers:

- `HQTV`
- `PNV`
- `THV`
- `TLV`

All local audio paths were available before running this experiment.

## Fold Design

The protocol is leave-one-Vietnamese-speaker-out. For each fold:

- test: all rows from one held-out Vietnamese speaker
- train: original `train` split rows where `speaker_id != held_out_speaker`
- validation: original `val` split rows where `speaker_id != held_out_speaker`

The held-out Vietnamese speaker is never present in train or validation.

Non-Vietnamese L1 groups are included in train/validation through the original split rows. This keeps the all-speaker training setup while making the Vietnamese test speaker unseen.

## Model And Training

The experiment reuses the CNN Attention architecture:

- 16 kHz audio segment
- 64-bin log-mel spectrogram
- 1.0 second max segment length with padding/truncation
- CNN channels: 1 -> 16 -> 32 -> 64 -> 96
- temporal attention pooling
- dropout plus linear classifier

Training config:

- epochs per fold: 12
- batch size: 8
- optimizer: Adam
- learning rate: 1e-4
- balancing: `WeightedRandomSampler`
- loss: unweighted `CrossEntropyLoss`
- random seed: 42

Classifier confidence remains model confidence only. It is not a pronunciation correctness score.

## Per-Fold Metrics

| Held-out speaker | Test rows | Accuracy | Macro F1 | Weighted F1 | Addition F1 | Deletion F1 | Substitution F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HQTV | 1358 | 0.6841 | 0.5168 | 0.7052 | 0.0800 | 0.7161 | 0.7544 |
| PNV | 893 | 0.6663 | 0.4818 | 0.7126 | 0.0426 | 0.6481 | 0.7548 |
| THV | 1399 | 0.6526 | 0.5286 | 0.6941 | 0.1506 | 0.7078 | 0.7274 |
| TLV | 1269 | 0.6288 | 0.4814 | 0.6627 | 0.0793 | 0.6802 | 0.6848 |

Best fold by macro F1: `THV`

Worst fold by macro F1: `TLV`

## Aggregate Metrics

Mean/std across four held-out Vietnamese speaker folds:

- accuracy: 0.6580 +/- 0.0202
- macro F1: 0.5022 +/- 0.0210
- weighted F1: 0.6937 +/- 0.0191
- addition F1: 0.0881 +/- 0.0391
- deletion F1: 0.6881 +/- 0.0266
- substitution F1: 0.7303 +/- 0.0286

## Comparison

| Run | Scope | Macro F1 | Addition F1 | Deletion F1 | Substitution F1 |
|---|---|---:|---:|---:|---:|
| Vietnamese-only CNN Attention | Original Vietnamese test split | 0.5105 | 0.1754 | 0.6667 | 0.6893 |
| All-speaker CNN Attention | Original Vietnamese subset test split | 0.5420 | 0.2769 | 0.6667 | 0.6824 |
| Vietnamese speaker-disjoint CNN Attention | Mean held-out Vietnamese speaker | 0.5022 +/- 0.0210 | 0.0881 +/- 0.0391 | 0.6881 +/- 0.0266 | 0.7303 +/- 0.0286 |

The non-disjoint all-speaker result improved Vietnamese subset macro F1 and addition F1, but the speaker-disjoint result does not preserve that improvement. Addition generalization is especially weak when the Vietnamese speaker is unseen.

## Limitations

Only four Vietnamese speakers are available in L2-ARCTIC, so fold-level variance is expected and the estimate remains limited.

The training data includes non-Vietnamese L1 groups. This helps increase general phone-error examples, but it does not prove Vietnamese-specific pronunciation modeling.

Class imbalance remains substantial. `addition` has much lower support than `substitution`, and the speaker-disjoint addition F1 is weak.

This experiment evaluates error-type classification only. Classifier confidence must not be treated as a pronunciation score or as pronunciation correctness.

## Decision

Recommendation: do not replace the current main CNN Attention model with the all-speaker model solely based on the previous non-disjoint improvement.

The speaker-disjoint result is a caution signal:

- mean held-out Vietnamese macro F1 is 0.5022, below the non-disjoint all-speaker Vietnamese subset macro F1 of 0.5420
- mean held-out Vietnamese addition F1 is 0.0881, below both previous comparison points
- unseen-speaker Vietnamese generalization remains unresolved

The all-speaker model should remain an experiment until the training strategy improves speaker-disjoint addition performance.

## Checkpoints

Local fold checkpoints were generated at:

- `ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_HQTV.pt`
- `ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_PNV.pt`
- `ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_THV.pt`
- `ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_TLV.pt`

These files are local artifacts and must not be committed.
