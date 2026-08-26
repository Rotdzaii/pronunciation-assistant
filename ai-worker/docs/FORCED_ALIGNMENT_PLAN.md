# Forced Alignment Plan

## Current Scaffold

The AI worker now routes prompt-based alignment through `ai-worker/app/alignment/alignment_service.py`.

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

- configure local MFA installation and model paths outside Git
- validate MFA TextGrid output on safe local audio
- feed MFA phone boundaries into learned CNN + Attention + Context heads for
  segment selection and localization
- keep CNN Attention confidence separate from pronunciation correctness scores
