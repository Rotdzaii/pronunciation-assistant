# MFA Backend Payload Validation

## Purpose

Historical scoring note: any heuristic score in this validation record predates
the Deep Learning First scoring decision. It is not a current contract example;
public score is now `null` with `score_type: "unavailable"` until a learned
quality scorer is trained and validated.

This document validates that an MFA-aligned `cnn_attention_context` AI result can be converted into a backend webhook payload compatible with:

```text
POST /practice/webhook/ai-result
```

The validation focuses on payload compatibility, metadata preservation, and payload safety after real or representative MFA-aligned inference.

## Why Payload Validation Is Needed After MFA Alignment

MFA adds richer alignment metadata such as forced-alignment status, TextGrid parse status, segment counts, and alignment reliability.

That metadata is useful for research and debugging, but the payload sent to the existing backend route must still:

- preserve legacy fields required by the current backend
- keep the full normalized result under `feedback.ai_result`
- avoid leaking local audio, TextGrid, checkpoint, or temporary MFA paths
- avoid leaking signed URL tokens

Alignment timing is not pronunciation correctness. Classifier confidence is not pronunciation correctness. Heuristic score is not real GOP.

## Demo Script

Script:

```text
ai-worker/scripts/demo_mfa_backend_payload.py
```

The script:

- sets `SCORER_MODE=cnn_attention_context`
- sets `ALIGNMENT_MODE=mfa`
- sets the selected `context_0_10` inference configuration
- uses a representative safe sample AI result when `--dry-run` is used or when `--audio-path` is omitted
- runs real local MFA-aligned context inference only when `--audio-path` is provided and `--dry-run` is not used
- validates the normalized AI result
- builds the backend webhook payload
- validates the webhook payload
- checks payload safety markers before any optional POST

## Dry-Run Command

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_mfa_backend_payload.py --dry-run
```

Dry-run does not execute MFA, does not run the scorer, and does not POST. It builds a representative safe sample AI result with MFA-style metadata and validates the payload conversion path.

## Real Local Audio Command

```powershell
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="C:\path\to\l2_arctic_cnn_attention_context_0_10.pt"
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_mfa_backend_payload.py --audio-path path\to\architecture.wav --transcript "Architecture" --checkpoint-path "$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH"
```

This command runs real local MFA-aligned context inference when local dependencies, checkpoint, and MFA resources are available.

For real local validation, set `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` or pass `--checkpoint-path`. Do not commit checkpoint files.

## Optional POST Command

POST is opt-in only:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_mfa_backend_payload.py --audio-path path\to\architecture.wav --transcript "Architecture" --checkpoint-path "$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH" --job-id <existing-practice-history-uuid> --post --webhook-url http://localhost:8000/practice/webhook/ai-result --secret <local-ai-webhook-secret>
```

Do not run `--post` unless:

- the backend is running
- the job id already exists in `practice_history`
- the secret matches the backend configuration
- payload validation and safety checks pass

## Expected Payload Fields

Legacy-compatible top-level fields:

- `job_id`
- `status`
- `score`
- `problem_phonemes`
- `feedback`

Rich worker fields that should remain preserved safely:

- `predicted_error_type`
- `diagnosis`
- `scorer`
- `metadata`
- `ai_result`

The full normalized result should also remain available under:

- `feedback.ai_result`

## Expected Metadata Fields

For MFA-aligned success, payload metadata preserved through `feedback.ai_result.metadata` should include:

- `alignment_method=mfa`
- `is_forced_alignment=true`
- `mfa_used=true`
- `textgrid_parse_success=true`
- `fallback_alignment=false`
- `location_reliability=forced_alignment`

Typical supporting metadata also includes:

- `alignment_status=success`
- `requested_alignment_mode=mfa`
- `word_segments_count`
- `phone_segments_count`
- `context_mode=context_0_10`
- `context_used=true`

## Recorded Safe Real Local Validation Result

Real local MFA backend payload validation passed with:

- `script=ai-worker/scripts/demo_mfa_backend_payload.py`
- `execution_mode=real_mfa_inference`
- `transcript=Architecture`
- `SCORER_MODE=cnn_attention_context`
- `ALIGNMENT_MODE=mfa`
- `context_mode=context_0_10`
- `alignment_status=success`
- `alignment_method=mfa`
- `is_forced_alignment=true`
- `mfa_used=true`
- `textgrid_parse_success=true`
- `fallback_alignment=false`
- `word_segments_count=1`
- `phone_segments_count=9`
- `segments_count=9`
- `location_reliability=forced_alignment`
- `score=67.1`
- `score_note=Heuristic/demo score, not production GOP.`
- `pronunciation_score_source=heuristic_gop`
- `predicted_error_type=deletion`
- `diagnosis_confidence=0.6575304269790649`
- `problem_phonemes=["ɹ", "tʰ", "k"]`
- `ai_result_valid=true`
- `payload_valid=true`
- `metadata_safety_check passed=true`
- `metadata_safety_check findings=[]`
- `post_requested=false`
- `post_attempted=false`

Checkpoint note for this validation:

- the first run failed because the default checkpoint path was missing
- the successful run used `--checkpoint-path "$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH"`
- checkpoint files are local artifacts and must not be committed

## Safety Notes

- No local paths should appear in the backend webhook payload.
- No local audio filenames or TextGrid filenames should appear in the backend webhook payload.
- No temporary MFA folder paths should appear in the backend webhook payload.
- No signed URL token fragments should appear in the backend webhook payload.
- Classifier confidence is not pronunciation correctness.
- Heuristic score is not real GOP.
- Alignment timing is not pronunciation correctness.

The demo checks the payload for markers such as:

- `C:\`
- `/tmp/`
- `AppData\Local\Temp`
- `.TextGrid`
- `.wav`
- `.webm`
- `.m4a`
- signed URL token markers such as `token=` or `x-amz-signature=`

If any marker is found, the safety check fails and POST must be skipped.

## Troubleshooting

Missing checkpoint:

- Provide `--checkpoint-path` or set `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH`.
- For real local validation, prefer setting `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` and passing `--checkpoint-path "$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH"` explicitly.
- Do not commit checkpoint files.

Missing MFA:

- Confirm MFA is installed locally and the configured environment is available.
- Confirm `MFA_CONDA_ENV`, `MFA_DICTIONARY_PATH`, and `MFA_ACOUSTIC_MODEL_PATH`.

Missing audio path:

- Use `--audio-path` for real inference.
- Without `--audio-path`, the script stays in representative sample mode.

Backend refused connection:

- Confirm the backend is running on the expected host and port.
- Confirm `--webhook-url` points to `/practice/webhook/ai-result`.

Webhook secret mismatch:

- Confirm the provided `--secret` matches backend `AI_WEBHOOK_SECRET`.
- The script never prints the secret value.

Payload safety check failed:

- Inspect the AI result generation path for leaked local artifact fields or string values.
- Confirm TextGrid, temp directory, checkpoint, local audio, and signed URL token data are sanitized before POST.
