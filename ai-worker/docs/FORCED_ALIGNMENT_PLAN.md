# Forced Alignment Plan

## Current Scaffold

The AI worker now routes prompt-based alignment through `ai-worker/app/alignment/alignment_service.py`.

MFA local validation has passed with MFA `3.3.9`, `conda run -n mfa mfa`, dictionary `english_us_mfa`, acoustic model `english_mfa`, and parsed TextGrid output with `alignment_method=mfa` and `is_forced_alignment=true`.

The service selects alignment using:

```dotenv
ALIGNMENT_MODE=fallback
ALLOW_ALIGNMENT_FALLBACK=true
```

Supported modes:

- `fallback`: use approximate fallback alignment
- `mfa`: try MFA, then fallback when `ALLOW_ALIGNMENT_FALLBACK=true`
- `none`: return a failed/no-alignment result

## Components

- `alignment_service.py`: mode selection and graceful fallback
- `mfa_aligner.py`: local MFA wrapper, no installation or downloads
- `textgrid_parser.py`: minimal TextGrid parser for MFA/Praat interval tiers
- `fallback_aligner.py`: approximate even-split scaffold

## Fallback Behavior

Fallback alignment remains the default to avoid breaking local development and tests. It must be labeled as approximate and must not be described as real forced alignment.

When MFA mode fails because MFA is missing, audio is missing, or dictionary/acoustic model paths are incomplete, the service returns fallback alignment if allowed. The fallback reason is recorded in metadata.

## TextGrid Parser

The TextGrid parser supports common tier names:

- words, word, Words
- phones, phone, Phones

It returns the normalized alignment contract with `alignment_method=mfa`, `alignment_status=success`, `words`, `phones`, and metadata marking the result as forced alignment.

## Next Steps

- introduce worker MFA mode with sanitized metadata and safe fallback behavior
- feed MFA phone boundaries into GOP/CaGOP scoring
- keep CNN Attention confidence separate from pronunciation correctness scores
