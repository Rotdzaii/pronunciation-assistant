# CNN Attention Context MFA-Aligned Inference

## Purpose

This document validates `SCORER_MODE=cnn_attention_context` with `ALIGNMENT_MODE=mfa` so the context scorer uses real MFA word and phone timing when MFA succeeds.

The goal is to keep fallback alignment available while making the metadata explicit about whether timing came from MFA or fallback scaffolding.

## How This Differs From Fallback Context Inference

Fallback context inference:

- uses approximate even-split timing
- may operate at word level when canonical phones are unavailable
- must be treated as limited location evidence

MFA-aligned context inference:

- uses MFA TextGrid timing parsed into the existing alignment contract
- prefers real phone segments when the TextGrid contains them
- preserves word and phone segment counts in final metadata

Alignment timing is evidence about estimated segment location only. It is not pronunciation correctness.

## Required Environment Variables

```dotenv
SCORER_MODE=cnn_attention_context
ALIGNMENT_MODE=mfa
ALLOW_ALIGNMENT_FALLBACK=true
MFA_CONDA_ENV=mfa
MFA_DICTIONARY_PATH=english_us_mfa
MFA_ACOUSTIC_MODEL_PATH=english_mfa
CNN_ATTENTION_CONTEXT_MODE=context_0_10
CNN_ATTENTION_CONTEXT_LEFT_SECONDS=0.10
CNN_ATTENTION_CONTEXT_RIGHT_SECONDS=0.10
```

Optional:

```dotenv
CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH=C:\path\to\l2_arctic_cnn_attention_context_0_10.pt
MFA_COMMAND=mfa
MFA_TEMP_DIR=C:\path\to\temp
```

Checkpoint files, MFA models, dictionaries, audio files, and temporary alignment outputs are local artifacts and must not be committed.

## Demo Commands

Dry-run only:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_mfa_aligned_inference.py --dry-run
```

Local audio validation:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_mfa_aligned_inference.py --audio-path path\to\architecture.wav --transcript "Architecture"
```

The demo does not run real MFA when `--dry-run` is used or when `--audio-path` is omitted.

## Expected Output

The demo prints:

- `CONFIG`
- `ALIGNMENT SUMMARY`
- `SCORER RESULT SUMMARY`
- `VALIDATION`
- `METADATA SAFETY CHECK`

On MFA success, the alignment summary should show:

- `alignment_method=mfa`
- `alignment_status=success`
- `is_forced_alignment=true`
- `mfa_used=true`
- `textgrid_parse_success=true`
- `word_segments_count`
- `phone_segments_count`
- `fallback_alignment=false`

If the AI result validator is available, `VALIDATION` also reports whether the normalized AI result passes contract and safety checks.

## Metadata Fields

Important final AI result metadata fields:

- `alignment_status`
- `alignment_method`
- `alignment_note`
- `requested_alignment_mode`
- `is_forced_alignment`
- `mfa_used`
- `mfa_attempted`
- `mfa_exit_code`
- `textgrid_parse_success`
- `fallback_alignment`
- `fallback_reason`
- `word_segments_count`
- `phone_segments_count`
- `context_mode`
- `context_used`

Local TextGrid and temporary MFA paths are stripped from the final AI result metadata and webhook payload.

## Fallback Behavior

When `ALIGNMENT_MODE=mfa` and MFA fails:

- if `ALLOW_ALIGNMENT_FALLBACK=true`, the scorer falls back to approximate alignment
- final metadata keeps `fallback_alignment=true`
- final metadata preserves `fallback_reason`
- final metadata keeps `requested_alignment_mode=mfa`

When fallback is used, the result must not claim forced alignment.

## Limitations

- MFA timing is alignment evidence, not pronunciation correctness.
- Heuristic score is not real GOP.
- Classifier confidence is not pronunciation correctness.
- Real GOP/CaGOP remains Phase 4B.
