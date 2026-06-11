# Phoenix v2 Worker Hardening

## Purpose

This document records the AI Worker hardening for Phoenix v2 Stable deploy/demo operation.

Phoenix v2 remains Deep Learning-first and production-safe. The worker safety layer validates configuration, normalizes output, records failed scoring safely, and protects queue/webhook behavior. It does not train models, modify checkpoints, or replace the selected Deep Learning scorer with rule-based scoring.

Selected Phoenix v2 runtime:

```dotenv
SCORER_MODE=cnn_attention_context
ALIGNMENT_MODE=mfa
MODEL_VERSION=phoenix_v2_stable
CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH=<local-compatible-checkpoint>
```

## What Was Hardened

- Worker startup and per-job logs now include `scorer_mode`, `alignment_mode`, and `model_version`.
- Phoenix v2 failure types are standardized at the worker boundary.
- `cnn_attention_context` requires an explicit `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH`.
- Missing context checkpoint env/path returns a safe failed scorer result with `error_type=checkpoint_missing`.
- Scorer exceptions are converted into failed webhook payloads instead of crashing the job path.
- Failed scorer results keep `score=null` and `problem_phonemes=[]`.
- Worker-normalized feedback includes `model_version`, `scorer_mode`, `alignment_method`, `is_forced_alignment`, `summary`, `details`, and `warnings` where available.
- Webhook payload validation happens before POST.
- Queue archive is attempted only after a successful 2xx webhook response.
- Archive failures are logged and do not mark the job as processed.
- Loop mode continues after a per-job failure.
- Once mode processes at most one message and exits without crashing on a handled per-job failure.

## Failure Behavior

Standard worker-level `feedback.error_type` values:

| Error type | Meaning |
|---|---|
| `audio_decode_failed` | Submitted audio could not be decoded. |
| `audio_preprocess_failed` | Audio preprocessing or feature extraction failed. |
| `alignment_failed` | MFA/alignment failed and no acceptable scoring path completed. |
| `checkpoint_missing` | Required checkpoint env/path is missing. |
| `checkpoint_incompatible` | Checkpoint exists but does not match the scorer architecture/state dict. |
| `scorer_timeout` | Scorer execution timed out. |
| `scorer_failed` | Scorer crashed or returned invalid output. |
| `webhook_failed` | Backend webhook POST failed or returned non-2xx. |
| `unknown_error` | Failure did not match a known category after sanitization. |

Failed scoring output sent to the webhook must follow the Phoenix v2 output contract:

```json
{
  "status": "failed",
  "score": null,
  "problem_phonemes": [],
  "feedback": {
    "model_version": "phoenix_v2_stable",
    "scorer_mode": "cnn_attention_context",
    "error_type": "scorer_failed",
    "summary": "Phoenix v2 could not produce a model score for this attempt."
  }
}
```

The worker must not create fake successful model scores when the Deep Learning scorer fails.

## Archive Policy

- If scoring succeeds and the webhook returns 2xx, archive the queue message.
- If scoring fails but the failed-status webhook returns 2xx, archive the queue message.
- If webhook payload validation fails, do not POST and do not archive.
- If webhook POST fails or returns non-2xx, do not archive.
- If archive RPC fails after a successful webhook, log `archive_success=false`; the worker does not claim the job was processed.

## Webhook Policy

The worker sends backend-compatible fields to:

```text
POST /practice/webhook/ai-result
```

Required fields:

- `job_id`
- `status`
- `score`
- `problem_phonemes`
- `feedback`

Current FastAPI behavior accepts `score=null` for `status=failed` and requires a numeric score for `status=completed`.

Webhook payloads are sanitized before submission. The worker must not send secrets, signed URLs, local audio paths, checkpoint paths, TextGrid paths, or temporary MFA paths.

## Known Limitations

- Worker error classification is based on known exception types and sanitized message text.
- Python thread/process hard timeout enforcement is not yet implemented; timeout classification is ready for future timeout wrappers.
- `cnn_attention_context` still depends on local torch/audio dependencies and a compatible local checkpoint.
- MFA requires local setup. If MFA falls back, location reliability remains limited and must be reported separately from scoring.
- Existing scorer internals may still mark some score paths as demo/heuristic; Phoenix v2 Stable must not present heuristic replacement scores as model scores.

## Manual Validation Checklist

Before deploy/demo:

- Set `SCORER_MODE=cnn_attention_context`.
- Set `ALIGNMENT_MODE=mfa`.
- Set `MODEL_VERSION=phoenix_v2_stable`.
- Set `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` to a compatible local checkpoint.
- Run `WORKER_MODE=once` against one safe queue job.
- Confirm worker logs include `msg_id`, `job_id`, `scorer_mode`, `alignment_mode`, `model_version`, webhook status, and archive result.
- Confirm successful result includes numeric `score`, `problem_phonemes`, and `feedback`.
- Confirm failed scorer path sends `status=failed`, `score=null`, `problem_phonemes=[]`, and `feedback.error_type`.
- Confirm backend webhook returns 2xx before archive.
- Confirm non-2xx webhook response leaves the queue message unarchived.
- Confirm logs do not expose service-role keys, webhook secrets, signed URLs, checkpoint paths, local audio paths, or TextGrid paths.

