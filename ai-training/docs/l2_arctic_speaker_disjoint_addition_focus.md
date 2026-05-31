# Speaker-Disjoint Addition-Focused CNN Attention

## Purpose

This experiment investigates weak `addition` performance under Vietnamese speaker-disjoint evaluation for L2-ARCTIC phone-error classification.

Previous speaker-disjoint CNN Attention result:

- mean macro F1: 0.5022 +/- 0.0210
- mean addition F1: 0.0881 +/- 0.0391
- mean deletion F1: 0.6881 +/- 0.0266
- mean substitution F1: 0.7303 +/- 0.0286

The target was to improve unseen Vietnamese speaker addition detection without making overall macro F1 unacceptable.

## Dataset And Folds

Dataset:

`ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv`

Fold design remained leave-one-Vietnamese-speaker-out:

- held-out speakers: `HQTV`, `PNV`, `THV`, `TLV`
- train: original `train` rows from every speaker except the held-out speaker
- validation: original `val` rows from every speaker except the held-out speaker
- test: all rows from the held-out Vietnamese speaker

The held-out speaker does not appear in train or validation.

## Addition Weakness Analysis

Total addition rows: 1,092 out of 18,610 total rows.

Vietnamese addition rows: 198.

Vietnamese addition count by speaker:

- `HQTV`: 72
- `PNV`: 16
- `THV`: 67
- `TLV`: 43

Vietnamese addition count in the original test split:

- `HQTV`: 5
- `PNV`: 1
- `THV`: 5
- `TLV`: 8

Top addition labels are mostly inserted phones after `sil`, especially `AH0`, `R`, `G`, `AX`, and `S`.

Baseline speaker-disjoint confusion involving addition showed two problems:

- true additions are often predicted as `deletion` or `substitution`
- many non-additions are also predicted as `addition`, especially substitutions

This means addition is not only under-detected; the decision boundary is unstable for unseen Vietnamese speakers.

## Config Change

The model architecture and audio preprocessing were unchanged:

- same CNN Attention architecture
- same label order: `addition`, `deletion`, `substitution`
- same 16 kHz, 64-bin log-mel, 1.0 second segment preprocessing
- same unweighted cross-entropy loss

Only the sampler changed:

- baseline: inverse-frequency weighted sampler
- addition-focused: inverse-frequency weighted sampler with extra `addition` multiplier of 1.5

An initial 3.0 multiplier was tested briefly and produced an immediate validation collapse, so it was reduced before the completed experiment.

## Per-Fold Results

| Held-out speaker | Accuracy | Macro F1 | Weighted F1 | Addition F1 | Deletion F1 | Substitution F1 |
|---|---:|---:|---:|---:|---:|---:|
| HQTV | 0.5376 | 0.4718 | 0.6246 | 0.1199 | 0.6284 | 0.6672 |
| PNV | 0.5364 | 0.4357 | 0.6296 | 0.0458 | 0.6089 | 0.6524 |
| THV | 0.6412 | 0.5181 | 0.6963 | 0.1354 | 0.6707 | 0.7481 |
| TLV | 0.5603 | 0.4603 | 0.6352 | 0.0823 | 0.6310 | 0.6677 |

## Aggregate Results

Addition-focused mean/std across folds:

- accuracy: 0.5689 +/- 0.0428
- macro F1: 0.4715 +/- 0.0299
- weighted F1: 0.6464 +/- 0.0291
- addition F1: 0.0958 +/- 0.0348
- deletion F1: 0.6347 +/- 0.0225
- substitution F1: 0.6839 +/- 0.0376

Best fold by macro F1: `THV`

Worst fold by macro F1: `PNV`

## Comparison To Baseline Speaker-Disjoint

| Metric | Baseline | Addition-focused | Delta |
|---|---:|---:|---:|
| Mean accuracy | 0.6580 | 0.5689 | -0.0891 |
| Mean macro F1 | 0.5022 | 0.4715 | -0.0307 |
| Mean addition F1 | 0.0881 | 0.0958 | +0.0077 |
| Mean deletion F1 | 0.6881 | 0.6347 | -0.0533 |
| Mean substitution F1 | 0.7303 | 0.6839 | -0.0465 |

Per-fold addition F1 changes:

- `HQTV`: +0.0399
- `PNV`: +0.0032
- `THV`: -0.0152
- `TLV`: +0.0030

## Decision

Recommendation: keep the addition-focused sampler as an experiment. Do not replace the baseline speaker-disjoint model.

The extra addition sampling slightly improved mean addition F1, but the improvement was too small and came with a material drop in macro F1, accuracy, deletion F1, and substitution F1.

The result suggests that sampler-only addition boosting is not enough. The addition class likely needs better feature context or a different training strategy rather than simply increasing its sampling probability.

## Limitations

Only four Vietnamese speakers are available, and `addition` support is very small for some folds, especially PNV.

The all-speaker training data includes non-Vietnamese L1 groups, so these results do not prove Vietnamese-specific pronunciation modeling.

Class imbalance remains severe. Classifier confidence remains model confidence only and must not be treated as pronunciation correctness or as a pronunciation score.

## Checkpoints

Local checkpoints were generated at:

- `ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_addition_focus_HQTV.pt`
- `ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_addition_focus_PNV.pt`
- `ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_addition_focus_THV.pt`
- `ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_addition_focus_TLV.pt`

These are local artifacts and must not be committed.
