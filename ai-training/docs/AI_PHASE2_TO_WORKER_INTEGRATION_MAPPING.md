# AI Phase 2 To Worker Integration Mapping

## Purpose

This document maps the Phase 2 selected research candidate to the AI Worker integration requirements. It is intended to prevent accidental changes in inference behavior when moving from training scripts to worker scoring.

## Selected Model

Selected Phase 2 research candidate:

`CNN Attention with context_0_10`

Selection evidence:

- Vietnamese speaker-disjoint multi-seed macro F1: `0.5170 +/- 0.0338`
- Vietnamese speaker-disjoint multi-seed addition F1: `0.1251 +/- 0.0473`
- Vietnamese speaker-disjoint multi-seed accuracy: `0.6618 +/- 0.0324`

This is a leading research candidate, not a final production pronunciation model.

## Checkpoint

Expected local checkpoint family:

`ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_<SEED>_<SPEAKER>.pt`

The exact deployment checkpoint should be chosen explicitly during implementation. Checkpoints are local artifacts and must not be committed.

## Training Context Setting

Training/evaluation context mode:

`context_0_10`

Meaning:

- Expand the annotated phone-error segment by 0.10 seconds on the left.
- Expand the annotated phone-error segment by 0.10 seconds on the right.
- Clamp the crop to the audio boundaries.
- Pad or truncate the crop to the 1.0 second CNN input length.

The training script computes this from the original segment `start_time` and `end_time`. The metadata also has `context_start_time`, `context_end_time`, and `context_duration`, but worker inference should compute the crop from live alignment boundaries because app inputs do not come from the L2-ARCTIC metadata CSV.

## Inference Context Requirement

For each aligned phone segment in the worker:

| Field | Meaning |
|---|---|
| `segment_start_time` | Original aligned phone start time. |
| `segment_end_time` | Original aligned phone end time. |
| `crop_start_time` | `max(0.0, segment_start_time - 0.10)`. |
| `crop_end_time` | `min(audio_duration, segment_end_time + 0.10)`. |

The model should infer on `crop_start_time` to `crop_end_time`, but the output should continue to show the original segment start/end for user-facing location.

## Label Order

The label order must remain:

1. `addition`
2. `deletion`
3. `substitution`

The worker should prefer checkpoint-provided `index_to_label` or `label_to_index` mappings when available, and fall back to this order only if the checkpoint does not include mappings.

## Expected Worker Output Fields

The context scorer should preserve the normalized AI result contract:

- `status`
- `score`
- `score_note`
- `pronunciation_score_source`
- `problem_phonemes`
- `predicted_error_type`
- `diagnosis`
- `feedback`
- `scorer`
- `metadata`
- `scoring`
- `segments`

The worker should continue to populate:

- `diagnosis.primary_error_type`
- `diagnosis.class_probabilities`
- `diagnosis.diagnosis_confidence`
- `diagnosis.confidence_note`
- `feedback.ai_result`

## Proposed Context Metadata

Add metadata at segment level where possible:

```json
{
  "context_mode": "context_0_10",
  "context_left_seconds": 0.10,
  "context_right_seconds": 0.10,
  "segment_start_time": 0.42,
  "segment_end_time": 0.49,
  "crop_start_time": 0.32,
  "crop_end_time": 0.59
}
```

Aggregated result metadata should also include:

- `context_mode`
- `context_left_seconds`
- `context_right_seconds`
- `segment_level_inference=true`
- `model_output_is_scoring=false`

## What To Preserve From Phase 2

Preserve:

- CNN Attention architecture.
- Label order.
- 16 kHz mono audio preprocessing.
- 64-bin log-mel spectrogram input.
- 1.0 second pad/truncate behavior.
- `context_0_10` crop behavior.
- Speaker-disjoint evaluation numbers in documentation.

## What Not To Claim In Worker Output

Do not claim:

- The model fully solves Vietnamese pronunciation modeling.
- The model confidence is pronunciation correctness.
- The heuristic score is real GOP/CaGOP.
- Fallback alignment boundaries are precise.
- Addition performance is solved.

Safe phrasing:

- "Classifier confidence, not pronunciation correctness."
- "Pronunciation score unavailable until a learned quality scorer is trained and validated."
- "Location reliability depends on alignment quality."

## Recommended Worker Strategy

Use a new worker scorer mode:

`SCORER_MODE=cnn_attention_context`

Use context-specific checkpoint configuration:

`CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH`

This avoids changing the behavior of the existing `cnn_attention` mode and allows side-by-side validation before switching defaults.
