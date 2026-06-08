# MFA Worker Once Validation

## Purpose

This document defines a production-like once-only validation flow for `ai-worker/worker.py` using:

- `WORKER_MODE=once`
- `SCORER_MODE=cnn_attention_context`
- `ALIGNMENT_MODE=mfa`
- `MFA_CONDA_ENV=mfa`
- `MFA_DICTIONARY_PATH=english_us_mfa`
- `MFA_ACOUSTIC_MODEL_PATH=english_mfa`

The goal is to validate that the real worker entrypoint can read one queue message, prepare queued frontend audio for MFA when needed, run MFA-aligned context inference, POST the backend webhook, and archive the queue message only after a successful POST.

Alignment timing is not pronunciation correctness. Classifier confidence is not pronunciation correctness. Heuristic score is not real GOP.

## Prerequisites

- backend is running locally and can accept `POST /practice/webhook/ai-result`
- queue `practice_jobs` has a message available
- Supabase environment is configured locally
- MFA environment `mfa` is available locally
- `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` points to a real local checkpoint

Do not commit audio files, TextGrid files, temporary folders, checkpoints, secrets, signed URLs, or `.env`.

## PowerShell Environment Setup

```powershell
$env:WORKER_MODE="once"
$env:SCORER_MODE="cnn_attention_context"
$env:ALIGNMENT_MODE="mfa"
$env:ALLOW_ALIGNMENT_FALLBACK="true"
$env:MFA_CONDA_ENV="mfa"
$env:MFA_DICTIONARY_PATH="english_us_mfa"
$env:MFA_ACOUSTIC_MODEL_PATH="english_mfa"
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="C:\path\to\l2_arctic_cnn_attention_context_0_10.pt"
$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="<local-ai-webhook-secret>"
```

Checkpoints are local artifacts and must not be committed.

## Worker Once Command

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\worker.py
```

## Expected Output

Expected worker output includes:

- `Supported scorer modes: mock, wav2vec2, cnn_attention, cnn_attention_context`
- `worker_mode=once`
- `scorer_mode=cnn_attention_context`
- either `No job found in practice_jobs queue.`
- or, for a real message:
- `msg_id=<PGMQ message id>`
- `job_id=<practice_history uuid>`
- `target_word=<target word>`
- `model_confidence=<value or unavailable>`
- `webhook_status_code=200`
- `Processed job <job_id> and archived message <msg_id>.`

Expected behavior:

- queue audio may come from frontend/Supabase as WebM or another browser format
- the scorer/alignment path prepares temporary local WAV input for MFA when needed
- local temp paths, TextGrid paths, signed URLs, and secrets must not appear in payloads
- the worker archives only after a successful webhook response

## practice_history Verification SQL

After a successful once-run, verify:

```sql
select
  id,
  status,
  score,
  problem_phonemes,
  updated_at,
  feedback->'ai_result'->'scorer'->>'name' as scorer_name,
  feedback->'ai_result'->'metadata'->>'alignment_method' as alignment_method,
  feedback->'ai_result'->'metadata'->>'alignment_status' as alignment_status,
  feedback->'ai_result'->'metadata'->>'is_forced_alignment' as is_forced_alignment,
  feedback->'ai_result'->'metadata'->>'mfa_used' as mfa_used,
  feedback->'ai_result'->'metadata'->>'textgrid_parse_success' as textgrid_parse_success,
  feedback->'ai_result'->'metadata'->>'context_mode' as context_mode,
  feedback->'ai_result'->'metadata'->>'location_reliability' as location_reliability
from public.practice_history
where id = '<JOB_ID>';
```

Expected MFA success values:

- `status=completed`
- `scorer_name=cnn_attention_context`
- `alignment_method=mfa`
- `alignment_status=success`
- `is_forced_alignment=true`
- `mfa_used=true`
- `textgrid_parse_success=true`
- `context_mode=context_0_10`
- `location_reliability=forced_alignment`

## PGMQ Archive Verification SQL

Verify the queue state around the processed message:

```sql
select *
from pgmq.q_practice_jobs
order by msg_id desc
limit 5;
```

After a successful once-run, verify that the processed message is no longer active in the queue and appears in the archive/history view exposed by your PGMQ setup.

## Troubleshooting

No queue message:

- confirm a real message exists in `practice_jobs`
- rerun after enqueueing a test job

Checkpoint missing:

- set `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH`
- confirm the path points to a real local checkpoint
- do not commit checkpoint files

MFA unavailable:

- confirm `MFA_CONDA_ENV=mfa`
- confirm MFA dictionary and acoustic model are available locally
- confirm queued audio can be prepared into temporary WAV input

Signed audio URL expired:

- re-enqueue the job with a fresh signed URL
- the worker relies on the scorer path to download queued audio from `audio_url`

Backend refused connection:

- confirm the backend is running
- confirm `NODE_WEBHOOK_URL` points to `POST /practice/webhook/ai-result`

Archive failed:

- confirm the webhook returned success first
- confirm the project exposes `archive_practice_job(...)` or `pgmq_archive`
- the worker should leave the message unarchived when the POST fails

Fallback alignment used:

- confirm MFA dictionary, acoustic model, and environment are available locally
- inspect sanitized alignment metadata in `feedback.ai_result.metadata`
- fallback metadata should clearly mark approximate and limited location reliability
