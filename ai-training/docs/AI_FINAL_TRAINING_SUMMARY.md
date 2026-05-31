# AI Final Training Summary

## Phase 2 Status

Phase 2 is complete from a research-summary perspective.

Completed work:

- Expanded L2-ARCTIC phone-error metadata from Vietnamese-only to all 24 speakers.
- Extracted and validated all-speaker audio locally.
- Trained all-speaker CNN Attention.
- Evaluated Vietnamese speaker-disjoint generalization.
- Tested an addition-focused sampler variant.
- Tested `context_0_10` speaker-disjoint CNN Attention.
- Confirmed `context_0_10` stability with a 3-seed, 4-fold evaluation.

## Dataset

Primary Phase 2 metadata:

`ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv`

Summary:

| Item | Value |
|---|---:|
| Total rows | 18,610 |
| Speakers | 24 |
| L1 groups | 6 |
| Addition rows | 1,092 |
| Deletion rows | 3,420 |
| Substitution rows | 14,098 |

Vietnamese subset:

| Item | Value |
|---|---:|
| Rows | 4,919 |
| Speakers | 4 |
| Addition rows | 198 |
| Deletion rows | 1,566 |
| Substitution rows | 3,155 |

## Final Phase 2 Candidate

Selected Phase 2 research candidate:

`CNN Attention with context_0_10`

Evaluation protocol:

Vietnamese leave-one-speaker-out with held-out speakers `HQTV`, `PNV`, `THV`, and `TLV`.

Multi-seed stability:

Seeds `42`, `123`, and `2026`.

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | 0.6618 | 0.0324 |
| Macro F1 | 0.5170 | 0.0338 |
| Addition F1 | 0.1251 | 0.0473 |
| Deletion F1 | 0.6819 | 0.0382 |
| Substitution F1 | 0.7439 | 0.0316 |

## Model Selection Decision

The previous candidate was Vietnamese-only CNN Attention. Phase 2 moves the leading research candidate to `context_0_10` CNN Attention because it was evaluated under the stricter unseen Vietnamese speaker protocol and showed stable improvements over the speaker-disjoint baseline.

The all-speaker non-disjoint Vietnamese subset result remains useful but is not enough for final selection because it may be optimistic.

The addition-focused sampler is not selected because it hurt macro F1 and the common classes.

## Local Checkpoints

Training generated local checkpoint files under:

`ai-training/models/`

Expected Phase 2 checkpoint families include:

- `l2_arctic_all_speakers_cnn_attention.pt`
- `l2_arctic_cnn_attention_speaker_disjoint_*.pt`
- `l2_arctic_cnn_attention_speaker_disjoint_addition_focus_*.pt`
- `l2_arctic_cnn_attention_speaker_disjoint_context_*.pt`
- `l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_*.pt`

These files are local artifacts and must not be committed.

## Remaining Limitations

- Only four Vietnamese speakers are available.
- Addition remains sparse and unstable.
- The classifier predicts phone-error type for known error segments; it is not an end-to-end pronunciation scoring system.
- Model confidence is not pronunciation correctness.
- More deployment work is needed before AI Worker integration should switch model versions.

## Next Recommended Phase

Recommended next phase:

`feature/ai-phase3-context-model-integration-plan`

The next phase should plan checkpoint packaging, inference compatibility, model version naming, and AI Worker integration safeguards without mixing classifier confidence into pronunciation scoring.
