# MFA Integration

## Purpose

Montreal Forced Aligner (MFA) is used to produce word and phone time boundaries from known prompt text and submitted speech audio. The CNN Attention classifier can then run over aligned segments instead of only the full clip.

This scaffold does not install MFA, download acoustic models, or commit dictionaries/models.

## Fallback vs MFA

Fallback alignment evenly splits audio duration across words or canonical phones. It is approximate scaffolding only and is not real forced alignment.

MFA forced alignment uses a local MFA installation, pronunciation dictionary, acoustic model, prompt transcript, and audio file to generate a TextGrid with estimated speech boundaries.

The worker must not claim MFA was used unless `alignment_method` is `mfa` and `metadata.is_forced_alignment` is `true`.

## Environment Variables

```dotenv
ALIGNMENT_MODE=fallback
ALLOW_ALIGNMENT_FALLBACK=true
MFA_COMMAND=mfa
MFA_DICTIONARY_PATH=C:\path\to\dictionary.dict
MFA_ACOUSTIC_MODEL_PATH=C:\path\to\acoustic-model.zip
MFA_TEMP_DIR=C:\path\to\temp
```

Modes:

- `ALIGNMENT_MODE=fallback`: default local-development mode
- `ALIGNMENT_MODE=mfa`: attempt local MFA alignment, then fallback if allowed
- `ALIGNMENT_MODE=none`: return a failed/no-alignment result

`ALLOW_ALIGNMENT_FALLBACK=true` keeps development and worker demos from failing when MFA is not installed or configured.

## Local Paths

MFA dictionaries and acoustic models are local runtime artifacts. They can be large and environment-specific, so they must not be committed to Git.

Expected local inputs for MFA mode:

- audio file path
- prompt text
- dictionary path
- acoustic model path
- writable temp directory

## TextGrid Mapping

`ai-worker/app/alignment/textgrid_parser.py` parses common MFA/Praat TextGrid files without heavy external libraries.

Supported tier names:

- word tiers: `words`, `word`, `Words`
- phone tiers: `phones`, `phone`, `Phones`

The parser maps TextGrid intervals into the alignment contract:

- `alignment_status: success`
- `alignment_method: mfa`
- `words`
- `phones`
- `segments`
- `metadata.is_forced_alignment: true`
- `metadata.is_fallback: false`
- `metadata.mfa_used: true`
- `metadata.textgrid_parse_success: true`
- `metadata.word_segments_count`
- `metadata.phone_segments_count`

Phone segments are preferred for downstream aligned CNN Attention inference when present.

When `ALIGNMENT_MODE=mfa` falls back because MFA fails and `ALLOW_ALIGNMENT_FALLBACK=true`, the fallback result should preserve:

- `metadata.requested_alignment_mode: mfa`
- `metadata.fallback_alignment: true`
- `metadata.fallback_reason`

The final AI result and webhook payload must not expose local TextGrid or MFA temporary paths.

## Current Status

The scaffold is implemented:

- TextGrid parser
- MFA wrapper
- alignment service
- fallback behavior
- demos

Actual MFA execution requires a local MFA installation and local dictionary/acoustic model configuration. No MFA dependency, dictionary, or model is installed by this feature.

The context scorer validation entry point is:

```text
ai-worker/scripts/demo_context_mfa_aligned_inference.py
```

## Future Learned Scoring

MFA phone boundaries are the timing source for CNN + Attention + Context
segment selection and localization. The selected future direction is an
audited-correct-samples correctness head and, when quality labels exist, a
learned quality/scoring head. GOP/CaGOP is not the Phoenix v2 roadmap.
