# MFA PGMQ Once Validation

## Purpose

This document defines a safe once-only PGMQ validation workflow for:

- `SCORER_MODE=cnn_attention_context`
- `ALIGNMENT_MODE=mfa`
- `MFA_CONDA_ENV=mfa`
- `MFA_DICTIONARY_PATH=english_us_mfa`
- `MFA_ACOUSTIC_MODEL_PATH=english_mfa`

The workflow reads one message from queue `practice_jobs`, downloads the queued audio to a temporary file, runs MFA-aligned context inference, validates the AI result and backend payload, POSTs only when `--post` is passed, and archives only when `--archive` is passed after a successful POST.

Default behavior is safe:

- no POST by default
- no archive by default
- no infinite loop

Alignment timing is not pronunciation correctness. Classifier confidence is not pronunciation correctness. Heuristic score is not real GOP.

## Script

```text
ai-worker/scripts/demo_mfa_pgmq_once.py
```

## Prerequisites

- backend is running locally if `--post` is used
- Supabase env is configured locally
- queue `practice_jobs` has at least one message if you want a real once-run
- `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` is configured locally or `--checkpoint-path` is passed
- MFA environment `mfa` is available locally
- MFA dictionary `english_us_mfa` is available locally
- MFA acoustic model `english_mfa` is available locally

Do not commit audio files, TextGrid files, temporary folders, checkpoints, secrets, signed URLs, or `.env`.

## Environment Notes

The script sets:

- `SCORER_MODE=cnn_attention_context`
- `ALIGNMENT_MODE=mfa`
- `ALLOW_ALIGNMENT_FALLBACK=true` unless overridden by `--allow-fallback false`
- `MFA_CONDA_ENV`
- `MFA_DICTIONARY_PATH`
- `MFA_ACOUSTIC_MODEL_PATH`
- `CNN_ATTENTION_CONTEXT_MODE=context_0_10`
- `CNN_ATTENTION_CONTEXT_LEFT_SECONDS=0.10`
- `CNN_ATTENTION_CONTEXT_RIGHT_SECONDS=0.10`

Real validation requires passing `--checkpoint-path` or setting `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` to a real local checkpoint. Checkpoints are local artifacts and must not be committed.

## Dry-Run Command Without POST Or Archive

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_mfa_pgmq_once.py
```

Explicit checkpoint example:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_mfa_pgmq_once.py --checkpoint-path "$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH"
```

## POST Command

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_mfa_pgmq_once.py --post --webhook-url http://localhost:8000/practice/webhook/ai-result --secret <local-ai-webhook-secret> --checkpoint-path "$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH"
```

## POST And Archive Command

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_mfa_pgmq_once.py --post --archive --webhook-url http://localhost:8000/practice/webhook/ai-result --secret <local-ai-webhook-secret> --checkpoint-path "$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH"
```

## Expected Output

The script prints:

- `CONFIG`
- `QUEUE_MESSAGE`
- `JOB_PAYLOAD`
- `DOWNLOAD_RESULT`
- `AI_RESULT_SUMMARY`
- `WEBHOOK_PAYLOAD_SUMMARY`
- `VALIDATION`
- `METADATA_SAFETY_CHECK`
- `POST_RESULT`
- `ARCHIVE_RESULT`
- `FINAL_SUMMARY`

Expected safe behavior:

- if no queue message exists, prints `no_queue_message=true` and exits safely
- redacts `audio_url`
- never prints the webhook secret
- removes temporary downloaded audio after the run
- never archives unless `--archive` is passed after a successful POST

Expected successful validation fields:

- `alignment_method=mfa`
- `is_forced_alignment=true` when MFA succeeds
- `mfa_used=true` when MFA succeeds
- `context_mode=context_0_10`
- `ai_result_valid=true`
- `payload_valid=true`
- `metadata_safety_check passed=true`
- `post_attempted=false` for the default dry run
- `archive_attempted=false` for the default dry run

If MFA fails and fallback is allowed, the script should preserve fallback metadata while still avoiding local path and signed URL leakage in the payload.

## Supabase Queue And Archive Verification SQL

If your project exposes queue/archive state through SQL inspection, verify the message state around the once-run:

```sql
select *
from pgmq.q_practice_jobs
order by msg_id desc
limit 5;
```

After a successful `--post --archive`, verify the message no longer remains active in the queue and appears in the archive/history view exposed by your PGMQ setup.

If your project exposes archived messages through a separate table or view, adapt the queue name accordingly and verify the same `msg_id`.

## practice_history Verification SQL

After a successful POST, verify the backend update:

```sql
select
  id,
  status,
  score,
  problem_phonemes,
  updated_at,
  feedback->'ai_result'->'scorer'->>'name' as scorer_name,
  feedback->'ai_result'->'metadata'->>'alignment_method' as alignment_method,
  feedback->'ai_result'->'metadata'->>'is_forced_alignment' as is_forced_alignment,
  feedback->'ai_result'->'metadata'->>'mfa_used' as mfa_used,
  feedback->'ai_result'->'metadata'->>'context_mode' as context_mode,
  feedback->'ai_result'->'metadata'->>'location_reliability' as location_reliability
from public.practice_history
where id = '<JOB_ID>';
```

Expected values after successful MFA POST:

- `status=completed`
- `scorer_name=cnn_attention_context`
- `alignment_method=mfa`
- `is_forced_alignment=true`
- `mfa_used=true`
- `context_mode=context_0_10`

## Troubleshooting

No queue message:

- confirm a real message exists in `practice_jobs`
- rerun after enqueueing a test job

Checkpoint missing:

- pass `--checkpoint-path`
- or set `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH`
- do not commit checkpoint files

MFA unavailable:

- confirm `MFA_CONDA_ENV=mfa`
- confirm MFA dictionary and acoustic model are available locally
- if fallback is disabled, the run should fail instead of silently approximating alignment

Signed audio URL expired:

- re-enqueue the job with a fresh signed URL
- the script downloads queue audio from `audio_url` and will fail if the URL is no longer valid

Backend connection refused:

- confirm the backend is running
- confirm `--webhook-url` points to `POST /practice/webhook/ai-result`

Archive not attempted:

- confirm `--archive` was passed
- confirm `--post` was also passed
- confirm POST succeeded before archive was attempted
