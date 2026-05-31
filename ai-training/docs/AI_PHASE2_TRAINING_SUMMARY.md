# AI Phase 2 Training Summary

## Purpose

Phase 2 expanded the L2-ARCTIC phone-error classification data and tested whether the CNN Attention model generalizes better to unseen Vietnamese speakers. The research target remains automatic English pronunciation error diagnosis based on Deep Learning, with the phone-level labels:

- `addition`
- `deletion`
- `substitution`

This phase does not claim to solve Vietnamese pronunciation modeling fully. It improves and evaluates a phone-error classifier using L2-ARCTIC annotations.

## Dataset Expansion

Phase 1 used the Vietnamese-only clean v2 dataset:

`ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv`

Phase 2 created the all-speaker L2-ARCTIC metadata:

`ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv`

All-speaker dataset statistics:

| Item | Value |
|---|---:|
| Total rows | 18,610 |
| Speakers | 24 |
| L1 groups | 6 |
| Addition rows | 1,092 |
| Deletion rows | 3,420 |
| Substitution rows | 14,098 |
| Train rows | 16,274 |
| Validation rows | 1,339 |
| Test rows | 997 |

L1 distribution:

| L1 group | Rows |
|---|---:|
| Vietnamese | 4,919 |
| Spanish | 3,415 |
| Chinese | 3,244 |
| Hindi | 2,871 |
| Arabic | 2,202 |
| Korean | 1,959 |

The all-speaker audio was extracted and validated locally for training. Raw audio, extracted dataset folders, and checkpoints are local artifacts and are not committed.

## Why Speaker-Disjoint Evaluation Was Needed

The all-speaker model improved the non-disjoint Vietnamese subset metrics, but non-disjoint splits can be optimistic because the same speakers may appear across train, validation, and test partitions.

For the app target, the more important question is whether the model generalizes to unseen Vietnamese learners. Phase 2 therefore added Vietnamese leave-one-speaker-out evaluation:

- Hold out one Vietnamese speaker for test: `HQTV`, `PNV`, `THV`, `TLV`.
- Train on original `train` rows from all other speakers.
- Validate on original `val` rows from all other speakers.
- The held-out Vietnamese speaker must not appear in train or validation.

## Experiment Sequence

| Step | Experiment | Decision |
|---|---|---|
| 1 | All-speaker metadata and audio preparation | Selected as the Phase 2 training data base. |
| 2 | All-speaker CNN Attention, non-disjoint Vietnamese subset | Promising, but not enough for model selection because speaker overlap may inflate results. |
| 3 | Vietnamese speaker-disjoint CNN Attention baseline | Required evaluation protocol; addition remained weak. |
| 4 | Addition-focused sampler | Not selected; small addition gain with large macro F1 loss. |
| 5 | `context_0_10` speaker-disjoint CNN Attention | Selected as leading robustness candidate. |
| 6 | Multi-seed `context_0_10` stability | Confirmed the context result across seeds 42, 123, and 2026. |

## Key Results

| Experiment | Macro F1 | Addition F1 | Decision |
|---|---:|---:|---|
| Vietnamese-only CNN Attention | 0.5105 | 0.1754 | Previous candidate. |
| All-speaker CNN Attention on non-disjoint Vietnamese subset | 0.5420 | 0.2769 | Useful signal, but not speaker-disjoint. |
| Speaker-disjoint baseline | 0.5022 +/- 0.0210 | 0.0881 +/- 0.0391 | Baseline for unseen Vietnamese speakers. |
| Addition-focused sampler | 0.4715 +/- 0.0299 | 0.0958 +/- 0.0348 | Not selected. |
| `context_0_10` speaker-disjoint single-seed | 0.5178 +/- 0.0252 | 0.1246 +/- 0.0271 | Leading candidate before stability. |
| `context_0_10` speaker-disjoint multi-seed | 0.5170 +/- 0.0338 | 0.1251 +/- 0.0473 | Final Phase 2 research candidate. |

Multi-seed `context_0_10` also reached accuracy `0.6618 +/- 0.0324`.

## Final Phase 2 Candidate

The final Phase 2 research candidate is:

`CNN Attention with context_0_10 under Vietnamese speaker-disjoint evaluation`

It is selected over the previous Vietnamese-only CNN Attention candidate because it is evaluated under the stricter unseen-speaker protocol and improves the speaker-disjoint baseline on both macro F1 and addition F1.

It is selected over the all-speaker non-disjoint result because non-disjoint Vietnamese subset metrics are not sufficient evidence of unseen-speaker generalization.

It is selected over the addition-focused sampler because sampler-only balancing harmed macro F1 and the common classes.

## Limitations

- Only four Vietnamese L2-ARCTIC speakers are available.
- Addition remains sparse and high variance.
- The all-speaker training pool includes non-Vietnamese L1 groups.
- The model classifies known phone-error segments; it is not a complete end-to-end pronunciation assessment system.
- Classifier confidence is not pronunciation correctness and must not be used as a pronunciation score.
- Local checkpoints were generated for experiments but are not committed.

## Integration Implication

AI Worker integration should preserve the selected `context_0_10` crop behavior. For each aligned phone segment, inference should expand the original segment by 0.10 seconds on the left and right, clamp the crop to the audio boundaries, and keep the original segment boundary as the user-facing location.

The final app feedback should remain assistive. It should not present the classifier as a teacher replacement, and it must keep classifier confidence separate from pronunciation scoring.

## Next Work

Recommended next phase:

`feature/ai-phase3-context-model-integration-plan`

Goals:

- Decide how the context model should be packaged for AI Worker use.
- Keep confidence separate from pronunciation scoring.
- Preserve Vietnamese-specific evaluation reporting.
- Add deployment-safe documentation for checkpoint handling and model versioning.
