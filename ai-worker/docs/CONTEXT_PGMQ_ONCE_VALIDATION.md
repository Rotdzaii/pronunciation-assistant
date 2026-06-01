# Context CNN Attention PGMQ Once Validation

## Purpose

This document describes a safe once-only PGMQ validation flow for `SCORER_MODE=cnn_attention_context`.

The script reads at most one message from the Supabase PGMQ queue, runs the context CNN Attention scorer, validates the AI result, builds and validates the backend webhook payload, and optionally POSTs and archives the message.

Default behavior is safe:

- no POST by default
- no archive by default
- no infinite loop

## Prerequisites

- `ai-worker/.venv` exists with worker dependencies installed.
- Supabase environment variables are configured.
- Queue `practice_jobs` exists.
- At least one practice job is queued if you want to test a real message.
- FastAPI backend is running if `--post` is used.
- A local context checkpoint exists.

The checkpoint is local only and must not be committed.

## Environment Variables

Required for queue read:

```powershell
$env:SUPABASE_URL="https://your-project.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY="<service-role-key>"
```

Required for context scorer:

```powershell
$env:SCORER_MODE="cnn_attention_context"
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="C:\path\to\context-checkpoint.pt"
$env:CNN_ATTENTION_CONTEXT_MODE="context_0_10"
$env:CNN_ATTENTION_CONTEXT_LEFT_SECONDS="0.10"
$env:CNN_ATTENTION_CONTEXT_RIGHT_SECONDS="0.10"
```

Required for POST:

```powershell
$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="<local-ai-webhook-secret>"
```

Do not commit `.env`, service-role keys, webhook secrets, checkpoints, or audio files.

## Dry-Run Command

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_pgmq_once.py
```

Optional queue override:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_pgmq_once.py --queue-name practice_jobs
```

Optional checkpoint override:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_pgmq_once.py `
  --checkpoint-path C:\path\to\context-checkpoint.pt
```

Expected dry-run output:

- `CONFIG`
- `QUEUE MESSAGE`
- `JOB PAYLOAD`
- `SCORER RESULT SUMMARY`
- `VALIDATION`
- `WEBHOOK PAYLOAD SUMMARY`
- `POST RESULT`
- `ARCHIVE RESULT`
- `NEXT VERIFY STEPS`

Expected dry-run behavior:

- Reads at most one queue message.
- Runs inference if a message, checkpoint, and audio are available.
- Prints `ai_result_valid=True` and `payload_valid=True` on success.
- Prints `post_attempted=False`.
- Prints `archive_attempted=False`.

If no queue message is available, the script prints a clear message and exits without error.

## Real POST Command

POST is explicit:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_pgmq_once.py `
  --post `
  --webhook-url http://localhost:8000/practice/webhook/ai-result `
  --secret <local-ai-webhook-secret>
```

This sends the validated payload to the backend but does not archive the PGMQ message.

## Archive Command

Archive is explicit and should normally be used only after POST:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_pgmq_once.py `
  --post `
  --archive `
  --webhook-url http://localhost:8000/practice/webhook/ai-result `
  --secret <local-ai-webhook-secret>
```

The script refuses to archive without `--post`, and only archives after a successful POST.

## Supabase And Backend Verification

After `--post`, verify the related `practice_history` row:

- `status=completed`
- `score` is set
- `problem_phonemes` is set
- `feedback.ai_result.scorer.name=cnn_attention_context`
- `feedback.ai_result.metadata.context_used=true`
- `feedback.ai_result.metadata.context_mode=context_0_10`
- `feedback.ai_result.metadata.crop_start_time` is present
- `feedback.ai_result.metadata.crop_end_time` is present

After `--post --archive`, verify the PGMQ message was archived.

## Troubleshooting

No queue message:

- Confirm a practice job was enqueued in `practice_jobs`.
- Run again after creating a practice job.

Missing Supabase environment:

- Set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.
- The script also loads `ai-worker/.env` if present.

Signed audio URL expired:

- Re-enqueue a job with a fresh audio URL.
- Or use a local path in the queue payload for local validation.

Missing checkpoint:

- Set `--checkpoint-path` or `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH`.
- Do not commit checkpoint files.

Torch missing:

- Install or repair the AI Worker inference dependencies in `ai-worker/.venv`.

Backend refused connection:

- Start FastAPI locally.
- Confirm webhook URL and port.

Secret mismatch:

- Confirm the backend and script use the same `AI_WEBHOOK_SECRET`.
- Do not print or commit secrets.

Archive failed:

- Confirm the Supabase RPC for archive is exposed.
- The script attempts the same archive RPC names as `worker.py`.

Scoring limitations:

- Classifier confidence is not pronunciation correctness.
- Heuristic score is not real GOP.
- Fallback alignment is approximate.
- The context CNN Attention model is a research candidate, not a fully final production pronunciation model.

Real PGMQ validation result:

- msg_id=24
- job_id=95a38bdd-02fa-421e-a619-cf6586f8dfbb
- inference_ran=true
- ai_result_valid=True
- payload_valid=True
- post_success=True
- archive_success=True
- scorer.name=cnn_attention_context
- context_mode=context_0_10
- context_used=true