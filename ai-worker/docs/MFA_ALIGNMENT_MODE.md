# MFA Alignment Mode

## Purpose

`ALIGNMENT_MODE=mfa` lets the AI Worker use Montreal Forced Aligner for real word and phone timing boundaries before segment-level pronunciation diagnosis. Alignment timing is not pronunciation correctness; it only provides locations for downstream scoring and diagnosis.

## Environment Variables

```dotenv
ALIGNMENT_MODE=mfa
ALLOW_ALIGNMENT_FALLBACK=true
MFA_COMMAND=mfa
MFA_CONDA_ENV=mfa
MFA_DICTIONARY_PATH=english_us_mfa
MFA_ACOUSTIC_MODEL_PATH=english_mfa
MFA_TEMP_DIR=
```

- `ALIGNMENT_MODE=fallback|mfa|none` selects the alignment provider.
- `ALLOW_ALIGNMENT_FALLBACK=true|false` controls whether MFA failures fall back to approximate alignment.
- `MFA_CONDA_ENV=mfa` runs MFA as `conda run -n mfa mfa`.
- `MFA_DICTIONARY_PATH` and `MFA_ACOUSTIC_MODEL_PATH` may be MFA model names or local paths.
- `MFA_TEMP_DIR` optionally chooses the local scratch root.

## Local Validation vs Worker Mode

`validate_mfa_local_alignment.py` is a standalone readiness check for one local audio file. `ALIGNMENT_MODE=mfa` is the worker service path used by AI scoring flows such as CNN Attention context inference.

## Dry Run

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_mfa_alignment_mode.py --dry-run
```

Dry-run prints the effective configuration and does not run real alignment.

## Real Local Audio Validation

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_mfa_alignment_mode.py `
  --audio-path "C:\path\to\local_sample.wav" `
  --transcript "Architecture" `
  --alignment-mode mfa `
  --allow-fallback `
  --conda-env mfa `
  --dictionary-path english_us_mfa `
  --acoustic-model-path english_mfa
```

## Expected Output

- `CONFIG`
- `ALIGNMENT RESULT`
- `alignment_status`
- `alignment_method`
- word segment count
- phone segment count
- first word segments
- first phone segments
- sanitized metadata

## Fallback Behavior

When `ALIGNMENT_MODE=mfa` fails and `ALLOW_ALIGNMENT_FALLBACK=true`, the worker returns approximate fallback alignment with:

- `alignment_status=fallback`
- `alignment_method=fallback_even_split`
- `requested_alignment_method=mfa`
- `fallback_alignment=true`
- sanitized `fallback_reason`

When fallback is disabled, the worker returns a failed alignment result.

## Metadata Fields

Successful MFA metadata is restricted to safe fields:

- `alignment_method`
- `alignment_status`
- `is_forced_alignment`
- `requested_alignment_method`
- `fallback_alignment`
- `mfa_used`
- `mfa_exit_code`
- `textgrid_parse_success`
- `word_segments_count`
- `phone_segments_count`

Fallback metadata may also include sanitized `fallback_reason`.

## Safety Notes

- Do not commit audio files, TextGrid files, temporary MFA folders, checkpoints, `.env` files, secrets, or signed URLs.
- Do not expose local audio paths, TextGrid paths, or temporary directory paths in backend webhook payloads.
- Fallback alignment remains approximate and must not be described as real forced alignment.
- MFA boundaries are timing evidence only, not pronunciation correctness.

## Troubleshooting

`MFA command not found`

Use `MFA_CONDA_ENV=mfa` when MFA works through `conda run -n mfa mfa`.

`MFA alignment falls back`

Check local MFA installation, dictionary/model names, transcript coverage, and audio format. The worker records a sanitized fallback reason.

`TextGrid generated but parsing fails`

Treat this as parser or integration validation, not as proof that MFA execution failed.
