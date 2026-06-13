# Phoenix v2 Runtime Validation

## Purpose

This document defines the manual runtime validation plan for Phoenix v2 Stable after AI Worker hardening.

Phoenix v2 runtime validation confirms that the hardened worker can run the selected Deep Learning scorer safely, send backend-compatible webhook payloads, and archive queue messages only after successful webhook delivery.

This plan does not train models, download datasets, modify checkpoints, modify the database schema, or require a real queue unless the local environment is already configured.

Selected runtime:

```dotenv
SCORER_MODE=cnn_attention_context
ALIGNMENT_MODE=mfa
MODEL_VERSION=phoenix_v2_stable
```

## References

- [Phoenix v2 Worker Hardening](PHOENIX_V2_WORKER_HARDENING.md)
- [Phoenix v2 Output Contract](../../ai-training/docs/PHOENIX_V2_OUTPUT_CONTRACT.md)
- [Context Worker Loop Validation](CONTEXT_WORKER_LOOP_VALIDATION.md)
- [MFA Worker Once Validation](MFA_WORKER_ONCE_VALIDATION.md)
- [Context CNN Attention Scorer](CNN_ATTENTION_CONTEXT_SCORER.md)

## Shared Environment

Use local values only. Do not commit `.env`, service-role keys, webhook secrets, signed URLs, local audio files, TextGrid files, or checkpoints.

From the repository root in PowerShell:

```powershell
$env:SUPABASE_URL="https://your-project.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY="<local-service-role-key>"
$env:QUEUE_NAME="practice_jobs"

$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="<local-ai-webhook-secret>"

$env:SCORER_MODE="cnn_attention_context"
$env:ALIGNMENT_MODE="mfa"
$env:MODEL_VERSION="phoenix_v2_stable"
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="<local-compatible-context-checkpoint>"
$env:CNN_ATTENTION_CONTEXT_MODE="context_0_10"
$env:CNN_ATTENTION_CONTEXT_LEFT_SECONDS="0.10"
$env:CNN_ATTENTION_CONTEXT_RIGHT_SECONDS="0.10"

$env:WORKER_MODE="once"
```

If MFA is expected to run rather than fall back, configure local MFA resources:

```powershell
$env:MFA_DICTIONARY_PATH="<local-dictionary-or-mfa-dictionary-name>"
$env:MFA_ACOUSTIC_MODEL_PATH="<local-acoustic-model-or-mfa-model-name>"
$env:MFA_COMMAND="mfa"
```

## A. Successful Worker Once Flow

Goal: verify that one queued practice job completes with the selected Phoenix v2 scorer and archives only after webhook success.

Prerequisites:

- FastAPI backend is running and accepts `POST /practice/webhook/ai-result`.
- Supabase env values are configured locally.
- PGMQ `practice_jobs` has one safe message with `job_id`, `student_id`, `target_word`, and `audio_url`.
- `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` points to a compatible local checkpoint.
- AI Worker virtual environment has required runtime dependencies.

Run:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\worker.py
```

Expected worker logs:

```text
worker_mode=once
scorer_mode=cnn_attention_context
alignment_mode=mfa
model_version=phoenix_v2_stable
msg_id=<message-id>
job_id=<job-id>
webhook_payload_valid=true
webhook_status_code=200
archive_success=true msg_id=<message-id>
```

Expected result behavior:

- Worker reads one queue message.
- Scorer mode is `cnn_attention_context`.
- Alignment mode is `mfa`.
- Model checkpoint loads successfully.
- Result status is `completed`.
- `score` is numeric from `0` to `100`.
- `problem_phonemes` is an array.
- `feedback` includes `model_version`, `scorer_mode`, and `alignment_method`.
- Backend webhook returns 2xx.
- Queue message is archived only after successful webhook.

Suggested SQL verification:

```sql
select
  id,
  status,
  score,
  problem_phonemes,
  feedback->>'model_version' as model_version,
  feedback->>'scorer_mode' as scorer_mode,
  feedback->>'alignment_method' as alignment_method,
  updated_at
from public.practice_history
where id = '<job-id>';
```

If the worker preserves richer nested output under `feedback.ai_result`, also inspect:

```sql
select
  feedback->'ai_result'->'scorer' as scorer,
  feedback->'ai_result'->'metadata'->>'context_mode' as context_mode,
  feedback->'ai_result'->'metadata'->>'alignment_method' as alignment_method,
  feedback->'ai_result'->'metadata'->>'is_forced_alignment' as is_forced_alignment
from public.practice_history
where id = '<job-id>';
```

## B. Failed Scorer Flow

Goal: verify that scorer failure is safely recorded as failed status and does not create a fake successful model score.

Safe missing-checkpoint scenario:

```powershell
$env:SCORER_MODE="cnn_attention_context"
$env:ALIGNMENT_MODE="mfa"
$env:MODEL_VERSION="phoenix_v2_stable"
Remove-Item Env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH -ErrorAction SilentlyContinue
$env:WORKER_MODE="once"
.\ai-worker\.venv\Scripts\python.exe ai-worker\worker.py
```

Alternative incompatible-checkpoint scenario:

```powershell
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="<local-incompatible-checkpoint>"
$env:WORKER_MODE="once"
.\ai-worker\.venv\Scripts\python.exe ai-worker\worker.py
```

Expected worker logs:

```text
scorer_mode=cnn_attention_context
alignment_mode=mfa
model_version=phoenix_v2_stable
result_status=failed
error_type=checkpoint_missing
webhook_payload_valid=true
webhook_status_code=200
archive_success=true msg_id=<message-id>
```

Expected result behavior:

- Missing or incompatible checkpoint is handled safely.
- Worker does not crash.
- Result status is `failed`.
- `score` is `null`.
- `problem_phonemes` is `[]`.
- `feedback.error_type` is set, such as `checkpoint_missing` or `checkpoint_incompatible`.
- `feedback.summary` is safe for the app.
- Queue message is archived only if the failed-status webhook returns 2xx.

Suggested SQL verification:

```sql
select
  id,
  status,
  score,
  problem_phonemes,
  feedback->>'error_type' as error_type,
  feedback->>'model_version' as model_version,
  feedback->>'scorer_mode' as scorer_mode,
  feedback->>'summary' as summary,
  updated_at
from public.practice_history
where id = '<job-id>';
```

Expected row:

- `status = failed`
- `score is null`
- `problem_phonemes = []`
- `error_type` is populated

## C. Webhook Failure Flow

Goal: verify that webhook failure prevents archive and does not lose the queue message.

Use a deliberately invalid local webhook URL or wrong webhook secret:

```powershell
$env:NODE_WEBHOOK_URL="http://localhost:9999/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="<local-ai-webhook-secret>"
$env:SCORER_MODE="cnn_attention_context"
$env:ALIGNMENT_MODE="mfa"
$env:MODEL_VERSION="phoenix_v2_stable"
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="<local-compatible-context-checkpoint>"
$env:WORKER_MODE="once"
.\ai-worker\.venv\Scripts\python.exe ai-worker\worker.py
```

Expected worker logs:

```text
webhook_payload_valid=true
Webhook request failed. Message <message-id> was not archived. error_type=webhook_failed
```

or:

```text
webhook_status_code=401
Webhook failed. Message <message-id> was not archived. error_type=webhook_failed
```

Expected behavior:

- Webhook failure is logged.
- Queue message is not archived.
- Worker once mode exits after at most one message.
- Worker loop mode can continue after the failure.

Queue verification depends on exposed PGMQ access. If archive tables are queryable, the message should not appear in `pgmq.a_practice_jobs`:

```sql
select msg_id, archived_at
from pgmq.a_practice_jobs
where msg_id = <message-id>;
```

Expected: no archived row for the failed webhook attempt.

After the visibility timeout, the same message should become readable again unless it is otherwise handled.

## Safe Static Validation

Run from the repository root:

```powershell
python -m compileall ai-worker/worker.py ai-worker/app ai-worker/audio ai-worker/scorers ai-worker/scripts
```

This validates Python syntax/import compilation for worker source paths without walking `ai-worker/.venv`.

Do not run:

```powershell
python -m compileall ai-worker
```

That command walks the local virtual environment when `ai-worker/.venv` exists and can be slow or noisy.

## Pass Criteria

- Static compile validation passes.
- Successful once flow returns completed result and archives after webhook 2xx.
- Failed scorer flow returns failed result with `score=null` and archives only after failed-status webhook 2xx.
- Webhook failure flow logs failure and does not archive.
- Logs include `msg_id`, `job_id`, `scorer_mode`, `alignment_mode`, `model_version`, webhook status, and archive result.
- Logs and persisted feedback do not expose secrets, signed URLs, local checkpoint paths, local audio paths, or TextGrid paths.

## Fail Criteria

- Worker crashes on missing/incompatible checkpoint.
- Failed scorer path emits a fake completed score.
- Failed result has missing `feedback.error_type`.
- Completed result has `score=null`.
- Webhook fails but queue message is archived.
- Payload validation fails for the Phoenix v2 output contract.
- Logs expose secrets or local artifact paths.

