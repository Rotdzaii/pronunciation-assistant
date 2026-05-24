# L2-ARCTIC Error Type CNN Evaluation

## Dataset Summary

This evaluation uses `ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification.csv`, which contains phone-level L2-ARCTIC Vietnamese annotation segments with explicit error labels:

- `addition`
- `deletion`
- `substitution`

The evaluator uses the existing train/validation/test split from the metadata. The model was evaluated on validation and test rows only.

Validation distribution:

- `addition`: 58
- `deletion`: 354
- `substitution`: 609

Test distribution:

- `addition`: 38
- `deletion`: 222
- `substitution`: 396

## Model And Task

The checkpoint at `ai-training/models/l2_arctic_error_type_cnn.pt` is a segment-level mel-spectrogram CNN baseline. It classifies one cropped phone segment into one of three explicit error types.

Preprocessing matches training:

- sample rate: 16 kHz
- segment length: 1 second after crop and padding
- features: 64-bin log mel-spectrogram
- classes: `addition`, `deletion`, `substitution`

This is a baseline for phone-level error type classification. It is closer to real mispronunciation diagnosis than native-vs-non-native classification because it evaluates explicit time-aligned phone error labels rather than corpus source identity.

## Overall Metrics

Validation:

- accuracy: 0.7571
- macro precision: 0.4993
- macro recall: 0.5228
- macro F1: 0.5104

Test:

- accuracy: 0.6966
- macro precision: 0.4533
- macro recall: 0.4790
- macro F1: 0.4657

The gap between accuracy and macro F1 is important. Accuracy is dominated by the larger `substitution` and `deletion` classes, while macro F1 exposes that the model currently fails on `addition`.

## Per-Class Metrics

Validation:

| class | precision | recall | F1 | support |
| --- | ---: | ---: | ---: | ---: |
| addition | 0.0000 | 0.0000 | 0.0000 | 58 |
| deletion | 0.7229 | 0.7147 | 0.7188 | 354 |
| substitution | 0.7750 | 0.8539 | 0.8125 | 609 |

Test:

| class | precision | recall | F1 | support |
| --- | ---: | ---: | ---: | ---: |
| addition | 0.0000 | 0.0000 | 0.0000 | 38 |
| deletion | 0.6245 | 0.6441 | 0.6341 | 222 |
| substitution | 0.7354 | 0.7929 | 0.7631 | 396 |

Worst class: `addition`.

## Confusion Matrix Interpretation

Validation confusion matrix:

| actual \ predicted | addition | deletion | substitution |
| --- | ---: | ---: | ---: |
| addition | 0 | 8 | 50 |
| deletion | 0 | 253 | 101 |
| substitution | 0 | 89 | 520 |

Test confusion matrix:

| actual \ predicted | addition | deletion | substitution |
| --- | ---: | ---: | ---: |
| addition | 0 | 4 | 34 |
| deletion | 0 | 143 | 79 |
| substitution | 0 | 82 | 314 |

The model does not predict `addition` for any validation or test sample. Most addition segments are predicted as `substitution`. This likely reflects class imbalance and acoustic ambiguity in short segment crops.

## Per-Speaker Notes

Validation accuracy:

- HQTV: 0.7700 over 287 samples
- PNV: 0.7742 over 186 samples
- THV: 0.7464 over 276 samples
- TLV: 0.7426 over 272 samples

Test accuracy:

- HQTV: 0.7427 over 171 samples
- PNV: 0.7419 over 124 samples
- THV: 0.7318 over 179 samples
- TLV: 0.5879 over 182 samples

TLV has the weakest test accuracy in this run. This should be investigated before interpreting the baseline as speaker-robust.

## Limitations

- The class distribution is imbalanced, especially `addition`.
- The model currently fails to recover the smallest class.
- The split may contain the same speakers across train, validation, and test. Stronger speaker-independent splits are needed for a stricter generalization claim.
- Segment-only crops may remove useful word and phonetic context.
- Metrics evaluate explicit annotation labels only. Model confidence must not be treated as pronunciation correctness.
- This is not user feedback generation and should not produce vague pronunciation feedback.

## Next Recommended Improvements

1. Add class balancing or weighted loss for `addition`.
2. Evaluate a speaker-independent split, such as leave-one-speaker-out.
3. Include target phone, realized phone, neighboring phones, and word context as model inputs.
4. Compare segment-level CNN performance against a simple label-prior baseline and a context-aware acoustic model.
5. Track macro F1 and per-class recall as primary metrics, not only accuracy.
