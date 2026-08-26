# Context CNN Attention Integration Plan

## 1. Purpose

Phase 2 selected CNN Attention with `context_0_10` as the leading research candidate for L2-ARCTIC phone-error classification. This document plans how to integrate that candidate into the AI Worker without replacing the existing CNN Attention scorer prematurely.

This is an integration plan only. It does not train a model, add a checkpoint, or implement the scorer replacement.

## 2. Selected Candidate

Selected candidate:

`CNN Attention + context_0_10`

Evaluation result:

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | 0.6618 | 0.0324 |
| Macro F1 | 0.5170 | 0.0338 |
| Addition F1 | 0.1251 | 0.0473 |

These metrics come from Vietnamese speaker-disjoint multi-seed stability evaluation. The model is the leading research candidate, not a final production pronunciation model.

## 3. Current AI Worker Scorer

Current scorer path:

`ai-worker/app/scorers/cnn_attention_scorer.py`

Current worker mode:

`SCORER_MODE=cnn_attention`

Current checkpoint override:

`CNN_ATTENTION_CHECKPOINT_PATH`

Current default checkpoint path:

`ai-training/models/l2_arctic_error_type_cnn_attention.pt`

Current scorer metadata:

```json
{
  "name": "cnn_attention",
  "type": "phone_error_classifier",
  "version": "cnn_attention_selected_baseline"
}
```

Current preprocessing:

- Load mono audio with `librosa`.
- Resample to 16 kHz.
- Crop either the full clip first second or the provided aligned segment.
- Pad or truncate to 1.0 second.
- Convert to 64-bin log-mel spectrogram.

Current segment behavior:

- `predict_segment(...)` and `predict_segments(...)` accept original segment `start` and `end` boundaries.
- Segment inference crops exactly `start_time` to `end_time`.
- There is no `context_start_time` / `context_end_time` inference branch yet.
- Aligned inference passes boundaries from the alignment contract through `get_alignment_segments(...)`.

Current output behavior:

- `predicted_error_type`, `class_probabilities`, and `diagnosis_confidence` are mapped into the normalized AI result contract.
- `diagnosis_confidence` is classifier confidence, not pronunciation correctness.
- `score` is generated separately through demo/heuristic scoring and must not be treated as the classifier confidence.
- Segment-level predictions are preserved under `segments`.
- Rich metadata and diagnosis are preserved under `feedback.ai_result` by the webhook payload path.

## 4. Required Context Inference Change

`context_0_10` means each target phone segment should be inferred with 0.10 seconds of left context and 0.10 seconds of right context.

For each aligned phone segment:

1. Preserve the original segment boundaries:
   - `segment_start_time`
   - `segment_end_time`
2. Compute context crop boundaries:
   - `crop_start_time = max(0.0, segment_start_time - 0.10)`
   - `crop_end_time = min(audio_duration, segment_end_time + 0.10)`
3. Run CNN Attention inference on the expanded crop.
4. Keep the original phone start/end in user-facing issue location fields.
5. Store the context crop boundaries separately in metadata.

The worker should not display context crop boundaries as the phone location. The crop is only an inference input window.

## 5. Output Contract Compatibility

The normalized output contract should remain compatible with the current backend/frontend path.

Fields to preserve:

- `status`
- `score`
- `problem_phonemes`
- `predicted_error_type`
- `diagnosis`
- `scorer`
- `metadata`
- `feedback.ai_result`

Proposed metadata additions:

```json
{
  "context_mode": "context_0_10",
  "context_left_seconds": 0.10,
  "context_right_seconds": 0.10,
  "segment_start_time": 1.23,
  "segment_end_time": 1.31,
  "crop_start_time": 1.13,
  "crop_end_time": 1.41
}
```

For segment-level output, each segment prediction should keep:

- original `start`
- original `end`
- `predicted_error_type`
- `class_probabilities`
- `diagnosis_confidence`
- `confidence_note`
- context crop metadata

The score path must remain separate from the classifier output. Classifier confidence must not be shown as pronunciation correctness.

## 6. Environment Variables

Recommended safer strategy:

Use a new scorer mode:

```dotenv
SCORER_MODE=cnn_attention_context
```

Recommended context-specific variables:

```dotenv
CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH=C:\path\to\l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_42_HQTV.pt
CNN_ATTENTION_CONTEXT_MODE=context_0_10
CNN_ATTENTION_CONTEXT_LEFT_SECONDS=0.10
CNN_ATTENTION_CONTEXT_RIGHT_SECONDS=0.10
```

This is safer than silently reusing `SCORER_MODE=cnn_attention` because the existing mode points to the earlier selected baseline checkpoint and exact segment crop behavior. A distinct mode lets the worker run both scorers side by side during validation.

Alternative:

Reuse `SCORER_MODE=cnn_attention` with `CONTEXT_MODE=context_0_10`.

This is less safe because a configuration mistake could change inference behavior without an obvious scorer-mode change.

## 7. Migration Plan

1. Keep the existing `cnn_attention` scorer unchanged.
2. Add a new `cnn_attention_context` scorer mode or a clearly isolated config branch.
3. Add a demo script for context scorer inference.
4. Validate normalized output contract fields and segment metadata.
5. Run backend webhook dry-run.
6. Run local backend POST verification.
7. Compare output shape with the current CNN Attention scorer.
8. Only then consider switching default scorer mode.

Recommended implementation branch:

`feature/ai-worker-context-cnn-attention-scorer`

## 8. Risks And Limitations

- Fallback alignment is approximate.
- Context inference quality depends on alignment boundary quality.
- GOP/CaGOP is not the Phoenix v2 roadmap.
- Public score is unavailable until a learned quality scorer is trained and
  validated; heuristic data remains internal diagnostics.
- Classifier confidence is not pronunciation correctness.
- The selected checkpoint is local only and must not be committed.
- Vietnamese speaker coverage is limited to four L2-ARCTIC speakers.
- Addition remains sparse and high variance.
- The model classifies known phone-error segments; it is not a complete end-to-end pronunciation assessment system.

## 9. Recommended Implementation Branch

`feature/ai-worker-context-cnn-attention-scorer`

Implementation should focus on a minimal, reversible integration:

- add a separate context scorer mode,
- preserve output contract compatibility,
- add context crop metadata,
- validate webhook payloads,
- keep confidence separate from pronunciation score.
