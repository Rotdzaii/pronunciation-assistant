# CNN Attention Context Scorer

## Purpose

The CNN Attention context scorer adds a new AI Worker mode for the Phase 2 selected research candidate:

`CNN Attention with context_0_10`

This scorer keeps the existing `cnn_attention` mode unchanged and adds a separate mode:

```dotenv
SCORER_MODE=cnn_attention_context
```

## Why Context_0_10

Phase 2 Vietnamese speaker-disjoint multi-seed stability selected `context_0_10` as the leading research candidate:

- macro F1: `0.5170 +/- 0.0338`
- addition F1: `0.1251 +/- 0.0473`
- accuracy: `0.6618 +/- 0.0324`

The model is still a research candidate, not a final production pronunciation model.

## Environment Variables

```dotenv
SCORER_MODE=cnn_attention_context
CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH=C:\path\to\context-checkpoint.pt
CNN_ATTENTION_CONTEXT_MODE=context_0_10
CNN_ATTENTION_CONTEXT_LEFT_SECONDS=0.10
CNN_ATTENTION_CONTEXT_RIGHT_SECONDS=0.10
```

Default local checkpoint path:

```text
ai-training/models/l2_arctic_cnn_attention_context_0_10.pt
```

Checkpoint files are local artifacts and must not be committed.

## Context Crop Behavior

For each aligned phone segment:

1. Preserve the original segment location:
   - `segment_start_time`
   - `segment_end_time`
2. Compute the model crop:
   - `crop_start_time = max(0.0, segment_start_time - 0.10)`
   - `crop_end_time = min(audio_duration, segment_end_time + 0.10)`
3. Run CNN Attention on the context-expanded crop.
4. Keep the original segment start/end as the user-facing location.
5. Store context metadata separately.

The context crop is an inference input window. It is not the user-facing phone location.

## Output Metadata

The final AI result metadata includes:

- `context_mode`
- `context_used`
- `context_left_seconds`
- `context_right_seconds`
- `segment_start_time`
- `segment_end_time`
- `crop_start_time`
- `crop_end_time`
- `model_output_is_scoring=false`

When alignment runs through `ALIGNMENT_MODE=mfa`, the final AI result should also preserve:

- `alignment_status`
- `alignment_method`
- `is_forced_alignment`
- `mfa_used`
- `textgrid_parse_success`
- `word_segments_count`
- `phone_segments_count`
- `fallback_alignment`

Each segment prediction can also include a `context` object with the same crop fields.

Classifier confidence remains diagnosis confidence only. It is not pronunciation correctness.

Local TextGrid or temporary MFA paths must not appear in the final AI result or webhook payload.

## Demo

Run with generated temporary audio:

```powershell
python ai-worker/scripts/demo_cnn_attention_context_scorer.py
```

Run with local audio:

```powershell
python ai-worker/scripts/demo_cnn_attention_context_scorer.py path\to\audio.wav --prompt-text "example" --phones EH G Z AE M P AH L
```

If `torch`, audio dependencies, or the checkpoint are missing, the demo prints a clear message and exits without creating committed artifacts.

For MFA-aligned validation, see:

```text
ai-worker/docs/CNN_ATTENTION_CONTEXT_MFA_ALIGNED_INFERENCE.md
```

## Limitations

- Fallback alignment is approximate and not real forced alignment.
- MFA-aligned context inference depends on local MFA setup and TextGrid parsing.
- Context quality depends on alignment boundary quality.
- GOP/CaGOP is not the Phoenix v2 roadmap.
- Public score is unavailable until a learned quality scorer is trained and
  validated.
- Classifier confidence is not pronunciation correctness.
- The context checkpoint is local only and must not be committed.
- Vietnamese speaker coverage remains limited.
- Addition remains sparse and high variance.
