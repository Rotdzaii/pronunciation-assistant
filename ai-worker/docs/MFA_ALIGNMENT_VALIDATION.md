# MFA Alignment Validation

## Fast validation

Run the unit suite without MFA installed:

```powershell
cd ai-worker
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run one local alignment and retain its TextGrid plus reports:

```powershell
python -m app.alignment.validate_mfa `
  --audio path\to\audio.wav `
  --text "cat" `
  --output-dir .\artifacts\mfa `
  --json-output .\artifacts\mfa\report.json `
  --segments-json .\artifacts\mfa\segments.json
```

Expected output has this shape:

```text
Alignment status: success
Source: mfa
Words: 1
Phones: 3
Coverage: 0.91
OOV: 0
Issues: none
```

The CLI exits with code `2` when MFA cannot create a reliable alignment. `--keep-debug` preserves the failed temporary corpus for local inspection. Do not commit generated audio, TextGrid, reports, or debug directories.

## Optional integration test

Integration runs are off by default, so CI and local unit testing do not require MFA models. Supply a known local audio file and transcript only on a machine with MFA configured:

```powershell
$env:RUN_MFA_INTEGRATION_TESTS="1"
$env:MFA_INTEGRATION_AUDIO="C:\data\cat.wav"
$env:MFA_INTEGRATION_TEXT="cat"
.\.venv\Scripts\python.exe -m unittest tests.test_mfa_alignment.MfaIntegrationTest -v
```

Record the output of `mfa version`, `mfa model inspect acoustic english_mfa`, and the CLI JSON report with the experiment. The present code change completed unit validation only; it did not execute MFA because the executable is absent in this environment.

## Quality report

`quality` is alignment integrity, never `pronunciation_score`:

```json
{
  "status": "ok",
  "quality_score": 0.91,
  "issues": [],
  "metrics": {
    "audio_duration": 1.2,
    "aligned_duration": 1.1,
    "speech_coverage_ratio": 0.917,
    "number_of_words": 1,
    "number_of_phones": 3,
    "oov_count": 0,
    "empty_interval_count": 0,
    "overlap_count": 0,
    "out_of_bounds_count": 0,
    "very_short_phone_count": 0,
    "very_long_phone_count": 0
  }
}
```

The status is `failed` for no phones/words, non-positive phone duration, overlaps, out-of-bounds boundaries, or coverage below `MFA_MIN_COVERAGE_RATIO`. It is `warning` for moderate coverage, OOV, long phones, or too many very short phones. Thresholds live in `.env.example`, not in scorer code.

When `ALLOW_ALIGNMENT_FALLBACK=true`, a failed MFA attempt returns the existing `fallback_even_split` result with `alignment_source=fallback`, quality warning, zero alignment confidence, and a limitation note. It is not forced alignment. With fallback disabled, the context scorer stops before inference and the worker emits a normal failed webhook result instead of inventing phone boundaries.

## Common failures

| Error category | Meaning | Action |
| --- | --- | --- |
| `mfa_not_installed` | `mfa` or the configured Conda environment is unavailable. | Install MFA or correct `MFA_COMMAND` / `MFA_CONDA_ENV`. |
| `dictionary_missing` / `acoustic_model_missing` | A configured path is missing, or MFA cannot resolve a model. | Download the matching model pair and inspect it. |
| `oov` | Transcript words are absent from the dictionary. | Add reviewed pronunciations or use MFA G2P as an explicit data-preparation step. |
| `audio_invalid` / `audio_silent` | Decode failed, sample data is invalid, or recording is silent. | Re-record or inspect the source audio. |
| `mfa_timeout` | Alignment exceeded `MFA_ALIGNMENT_TIMEOUT_SECONDS`. | Inspect debug artifact, resource limits, and model setup. |
| `mfa_temp_unavailable` / `mfa_output_unavailable` | The worker cannot create its temporary or requested artifact directory. | Correct permissions or `MFA_TEMP_DIR`. |
| `textgrid_missing` / `textgrid_invalid` | MFA did not create usable phone timings. | Keep debug artifacts and inspect MFA stdout/stderr and TextGrid tiers. |

The worker log emits key-value events with job id, audio duration, transcript, selected MFA model/dictionary name, runtime, word/phone counts, OOV count, quality status, and error category. It never logs audio bytes and webhook sanitization removes local artifact paths.
