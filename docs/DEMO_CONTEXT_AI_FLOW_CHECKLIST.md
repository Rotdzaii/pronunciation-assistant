# Demo Context AI Flow Checklist

## Purpose

Use this checklist to run the full context AI demo flow:

```text
frontend recording -> FastAPI backend -> Supabase Storage + practice_history -> PGMQ practice_jobs -> AI Worker -> backend webhook -> result/history verification
```

The selected scorer for this demo is `SCORER_MODE=cnn_attention_context`.

## Safety Notes

- Do not train models during the demo.
- Do not commit `.env` files, service-role keys, webhook secrets, signed audio URLs, checkpoints, or audio files.
- Do not paste a real signed URL into this checklist or any committed file.
- Stop old worker terminals before the demo.
- Do not run an old `wav2vec2` worker at the same time as the context worker.
- Classifier confidence is diagnosis confidence, not pronunciation correctness.
- The heuristic score is not real GOP.
- Fallback alignment is approximate.

## Prerequisites

- Supabase project is configured.
- `public.practice_history` exists.
- PGMQ `practice_jobs` queue and read/archive RPC wrappers exist.
- Supabase Storage bucket exists for practice audio.
- FastAPI backend `.env` is configured locally.
- Frontend `.env` is configured locally with frontend-safe values.
- AI Worker `.env` or shell has Supabase and webhook settings.
- Local context checkpoint exists and is not committed.
- AI Worker virtual environment has dependencies installed, including `torch`.

## Terminals To Open

Open three terminals:

- Terminal 1: FastAPI backend
- Terminal 2: Expo frontend
- Terminal 3: AI Worker

Before starting, close any old backend, frontend, or worker terminals from previous demos.

## Backend Startup

From the repo root:

```powershell
cd fastapi-backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Check health:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected: backend returns a healthy response.

## Frontend Startup

Open a second terminal from the repo root:

```powershell
cd frontend
$env:EXPO_PUBLIC_API_BASE_URL="http://localhost:8000"
npm run web
```

Open the Expo Web URL, usually:

```text
http://localhost:8081
```

## AI Worker Env Setup

Open a third terminal from the repo root:

```powershell
$env:SUPABASE_URL="https://your-project.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY="<local-service-role-key>"
$env:QUEUE_NAME="practice_jobs"

$env:SCORER_MODE="cnn_attention_context"
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="C:\path\to\context-checkpoint.pt"
$env:CNN_ATTENTION_CONTEXT_MODE="context_0_10"
$env:CNN_ATTENTION_CONTEXT_LEFT_SECONDS="0.10"
$env:CNN_ATTENTION_CONTEXT_RIGHT_SECONDS="0.10"

$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="<local-ai-webhook-secret>"
$env:WORKER_MODE="once"
```

Do not print or commit real service-role keys, webhook secrets, or checkpoint files.

## Create Practice Job From Frontend

In the frontend:

1. Log in as a student.
2. Open the practice flow.
3. Record audio.
4. Replay the recording if needed.
5. Submit to AI scoring.

Expected backend calls:

```text
POST /practice/upload-audio
POST /practice/create-job
GET /practice/{job_id}
```

The created queue message should contain an `audio_url`. Treat it as a signed URL and do not commit it.

## Verify Latest Practice History Rows

Run in Supabase SQL editor:

```sql
select
  id,
  student_id,
  target_word,
  status,
  score,
  problem_phonemes,
  created_at,
  updated_at
from public.practice_history
order by created_at desc
limit 10;
```

Before the worker runs, the new row is expected to be `processing`.

## Verify Specific Job Id

Replace `<job-id>` with the frontend-created `practice_history.id`.

```sql
select
  id,
  student_id,
  target_word,
  status,
  score,
  problem_phonemes,
  feedback,
  created_at,
  updated_at
from public.practice_history
where id = '<job-id>';
```

## Verify PGMQ Message

If the project exposes a read RPC, use a safe read with visibility timeout awareness. Reading may hide the message temporarily, so avoid this immediately before the worker demo unless needed.

If queue tables are queryable:

```sql
select
  msg_id,
  read_ct,
  vt,
  message,
  enqueued_at
from pgmq.q_practice_jobs
order by msg_id desc
limit 10;
```

Confirm the message payload contains the expected `job_id`, `student_id`, `target_word`, and an `audio_url`. Do not copy the signed URL into docs or commits.

## Run Worker Once

From the AI Worker terminal at the repo root:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\worker.py
```

Expected output:

```text
Supported scorer modes: mock, wav2vec2, cnn_attention, cnn_attention_context
worker_mode=once
scorer_mode=cnn_attention_context
msg_id=<message-id>
job_id=<job-id>
target_word=<target-word>
model_confidence=<classifier-confidence>
webhook_status_code=200
Processed job <job-id> and archived message <message-id>.
```

Known completed validation example:

```text
worker_mode=once
scorer_mode=cnn_attention_context
msg_id=26
job_id=aeaf098f-64cf-4cbd-8469-1d245fd5d93a
webhook_status_code=200
message archived successfully
```

## Verify Backend Webhook Update

Run in Supabase SQL editor:

```sql
select
  id,
  status,
  score,
  problem_phonemes,
  feedback->'ai_result'->'scorer' as scorer,
  feedback->'ai_result'->'diagnosis'->>'diagnosis_confidence' as diagnosis_confidence,
  feedback->'ai_result'->'metadata'->>'context_used' as context_used,
  feedback->'ai_result'->'metadata'->>'context_mode' as context_mode,
  feedback->'ai_result'->'metadata'->>'scoring_method' as scoring_method,
  updated_at
from public.practice_history
where id = '<job-id>';
```

Expected:

- `status` is `completed`.
- `score` is present.
- `feedback.ai_result.scorer.name` is the context CNN Attention scorer.
- `context_used` is `true` when context inference metadata is present.
- `context_mode` is `context_0_10`.

## Verify Archive Message

If archive tables are queryable:

```sql
select
  msg_id,
  archived_at,
  message
from pgmq.a_practice_jobs
where msg_id = <message-id>;
```

Expected: one archived row for the processed `msg_id`.

Also confirm the active queue no longer returns the same message:

```sql
select
  msg_id,
  message,
  enqueued_at
from pgmq.q_practice_jobs
where msg_id = <message-id>;
```

Expected: no active row for the archived message.

## Verify Frontend Or History Result

In the frontend:

1. Return to the practice result screen if it polls `GET /practice/{job_id}`.
2. Open History if available.
3. Confirm the submitted attempt appears as completed.
4. Confirm the displayed score and problem phonemes are populated.

Remember: the displayed score is currently heuristic and is not real GOP.

## Troubleshooting

No queue message:

- Confirm the frontend called `POST /practice/create-job`.
- Check the latest `practice_history` rows.
- Confirm `QUEUE_NAME=practice_jobs`.
- If a read RPC was used manually, wait for the visibility timeout or create a new job.

Worker uses `wav2vec2` accidentally:

- Stop all old worker terminals.
- Check the AI Worker terminal has `$env:SCORER_MODE="cnn_attention_context"`.
- Check `ai-worker/.env` does not override the intended mode.
- Startup output must show `scorer_mode=cnn_attention_context`.

Missing checkpoint:

- Confirm `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` points to an existing local checkpoint.
- Do not commit `.pt`, `.pth`, or `.ckpt` files.

Missing torch or audio dependencies:

- Use the AI Worker virtual environment.
- Install dependencies from `ai-worker/requirements.txt`.
- Do not train models to fix demo setup.

Signed audio URL expired:

- Create a fresh frontend practice job.
- Do not commit `<signed-url>` or paste the real URL into logs that will be committed.

Backend refused connection:

- Confirm FastAPI is running on `http://localhost:8000`.
- Confirm `NODE_WEBHOOK_URL=http://localhost:8000/practice/webhook/ai-result`.
- Confirm `AI_WEBHOOK_SECRET` matches the backend local secret.

Webhook returns non-2xx:

- Check backend logs.
- Confirm the `practice_history` row exists for the `job_id`.
- Confirm the webhook secret is correct.
- The worker should not archive the message after a failed webhook.

Frontend does not show result:

- Query `practice_history` directly.
- Confirm the row status is `completed`.
- Refresh the frontend history screen.
- Check frontend API base URL points to `http://localhost:8000`.

Ctrl+C behavior:

- Use Ctrl+C to stop loop-mode workers after the demo.
- Leave no old worker terminal running before a new demo.
