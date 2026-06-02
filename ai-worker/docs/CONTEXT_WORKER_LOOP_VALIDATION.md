# Context Worker Loop Validation

## Purpose

Validate the real `ai-worker/worker.py` flow with `SCORER_MODE=cnn_attention_context`.

This is a production-path validation, not a training run. It confirms that the worker reads a PGMQ message, runs the selected context CNN Attention scorer, posts the webhook payload to FastAPI, and archives the queue message only after a successful webhook response.

## Worker Path Inspection

`WORKER_MODE` is read in `_load_env()` from the shell or `ai-worker/.env`. Valid values are `once` and `loop`.

- `WORKER_MODE=once`: `main()` calls `_process_one_job(...)` once, then exits with status `0`.
- `WORKER_MODE=loop`: `main()` repeatedly calls `_process_one_job(...)`; when no message is found, it sleeps with backoff up to `WORKER_IDLE_BACKOFF_MAX_SECONDS`.

`SCORER_MODE` is read in `_load_env()` from the shell or `ai-worker/.env`. Supported modes are:

```text
mock, wav2vec2, cnn_attention, cnn_attention_context
```

The real worker supports `cnn_attention_context` in `_score(...)` by importing `score_pronunciation_context` from `app.scorers.cnn_attention_scorer`.

PGMQ read behavior:

- `_read_one_job(...)` tries `read_practice_job` first.
- It falls back to common `pgmq_read` wrapper names and parameter shapes.
- `_parse_queue_row(...)` requires `job_id`, `student_id`, `target_word`, and `audio_url`.

Webhook POST behavior:

- `_post_webhook(...)` sends the normalized success or failed webhook payload to `NODE_WEBHOOK_URL`.
- It uses the `x-ai-webhook-secret` header.
- The secret is not printed.

Archive behavior:

- `_archive_job(...)` is called only after the webhook returns a 2xx status code.
- It tries `archive_practice_job` first, then common `pgmq_archive` wrapper names.
- Failed webhook requests leave the queue message unarchived.

Ctrl+C behavior:

- `KeyboardInterrupt` exits cleanly with `Worker stopped by Ctrl+C.` and status `0`.
- The worker does not print a traceback for manual loop shutdown.

## Difference From `demo_context_pgmq_once.py`

`demo_context_pgmq_once.py` is a controlled validation helper:

- forces `SCORER_MODE=cnn_attention_context`
- can dry-run without POST or archive
- redacts audio fields when printing the job payload
- can guard by expected message id
- is useful for proving the scorer and payload contract safely

`worker.py` is the real worker path:

- reads `SCORER_MODE` from env
- reads one message or loops depending on `WORKER_MODE`
- posts to the configured backend webhook
- archives after successful POST
- is the path used for real demo operation

## Prerequisites

- FastAPI backend is running and can accept `POST /practice/webhook/ai-result`.
- Supabase env is configured for the AI Worker.
- PGMQ queue has at least one `practice_jobs` message.
- The queue message includes `job_id`, `student_id`, `target_word`, and `audio_url`.
- `SCORER_MODE=cnn_attention_context`.
- The local context checkpoint path exists.
- The AI Worker virtual environment has `torch` and audio dependencies installed.
- The signed audio URL in the queue has not expired.

Do not commit `.env`, service-role keys, webhook secrets, checkpoints, signed URLs, or audio files.

## PowerShell Setup

Use local values only. Do not paste real secrets into committed files.

```powershell
$env:SCORER_MODE="cnn_attention_context"
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="C:\path\to\context-checkpoint.pt"
$env:CNN_ATTENTION_CONTEXT_MODE="context_0_10"
$env:CNN_ATTENTION_CONTEXT_LEFT_SECONDS="0.10"
$env:CNN_ATTENTION_CONTEXT_RIGHT_SECONDS="0.10"
$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="<local-ai-webhook-secret>"
$env:WORKER_MODE="once"
```

Supabase values can come from the shell or `ai-worker/.env`:

```powershell
$env:SUPABASE_URL="https://your-project.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY="<local-service-role-key>"
$env:QUEUE_NAME="practice_jobs"
```

## Run Worker Once

From the repository root:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\worker.py
```

Or from `ai-worker`:

```powershell
cd ai-worker
.\.venv\Scripts\python.exe worker.py
```

Expected output includes:

```text
Supported scorer modes: mock, wav2vec2, cnn_attention, cnn_attention_context
worker_mode=once
scorer_mode=cnn_attention_context
msg_id=<message-id>
job_id=<practice-history-job-id>
target_word=<target-word>
model_confidence=<classifier-confidence>
webhook_status_code=200
Processed job <practice-history-job-id> and archived message <message-id>.
```

`model_confidence` is classifier diagnosis confidence. It is not pronunciation correctness and must stay separate from `score`.

## Run Worker Loop

Use loop mode only when the backend and queue are ready:

```powershell
$env:WORKER_MODE="loop"
.\ai-worker\.venv\Scripts\python.exe ai-worker\worker.py
```

Expected startup output includes:

```text
worker_mode=loop
scorer_mode=cnn_attention_context
Worker loop started queue=practice_jobs poll_interval_seconds=1.0 idle_backoff_max_seconds=10.0
```

If the queue is empty, expected output includes:

```text
No job found in practice_jobs queue.
```

Press Ctrl+C to stop the loop. Expected shutdown output:

```text
Worker stopped by Ctrl+C.
```

## Supabase Verification Queries

Verify the practice history row was completed:

```sql
select
  id,
  status,
  score,
  problem_phonemes,
  feedback->'ai_result'->'scorer' as scorer,
  feedback->'ai_result'->'diagnosis'->>'diagnosis_confidence' as diagnosis_confidence,
  feedback->'ai_result'->'metadata'->>'context_used' as context_used,
  updated_at
from public.practice_history
where id = '<job-id>';
```

Confirm the result used the context scorer:

```sql
select
  feedback->'ai_result'->'scorer'->>'name' as scorer_name,
  feedback->'ai_result'->'metadata'->>'context_mode' as context_mode,
  feedback->'ai_result'->'metadata'->>'scoring_method' as scoring_method
from public.practice_history
where id = '<job-id>';
```

Archive verification depends on the exposed PGMQ schema and RPC wrappers. If `pgmq` archive tables are queryable:

```sql
select msg_id, archived_at, message
from pgmq.a_practice_jobs
where msg_id = <msg-id>;
```

If only wrapper RPCs are exposed, verify by reading the queue again and confirming the same `msg_id` is no longer returned.

## Troubleshooting

No queue message:

- Confirm the frontend or backend created a practice job.
- Confirm `QUEUE_NAME=practice_jobs`.
- Confirm the message is not hidden by a visibility timeout from a prior read.

Worker uses `wav2vec2` accidentally:

- Check `SCORER_MODE` in the active shell.
- Check `ai-worker/.env`.
- Startup logs should print `scorer_mode=cnn_attention_context`.

Missing checkpoint:

- Confirm `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` points to an existing local `.pt` file.
- Do not commit the checkpoint.

Missing torch:

- Activate the AI Worker virtual environment.
- Install dependencies locally from `ai-worker/requirements.txt`.
- Do not train models during validation.

Signed audio URL expired:

- Create a fresh frontend/backend practice job.
- Do not paste signed URLs into docs, commits, or issue logs.

Backend refused connection:

- Start the FastAPI backend.
- Confirm `NODE_WEBHOOK_URL` points to the correct local route.
- Confirm the webhook secret matches the backend environment.

Ctrl+C behavior:

- In loop mode, Ctrl+C should print `Worker stopped by Ctrl+C.` and exit without a traceback.
