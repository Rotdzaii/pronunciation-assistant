# Context CNN Attention Worker End-To-End Validation

## Purpose

This document describes the queue-like end-to-end validation demo for `SCORER_MODE=cnn_attention_context`.

The demo simulates the core local worker flow:

1. Build a queue-like job payload.
2. Run the worker scorer dispatch path for `cnn_attention_context`.
3. Build the normalized AI result.
4. Validate the AI result.
5. Build the backend webhook payload.
6. Validate the webhook payload.
7. Optionally POST to the backend when `--post` is explicitly provided.

The default run is dry-run only.

## Difference From Backend POST Verification

`demo_context_backend_payload.py` validates the context scorer payload path directly.

`demo_context_worker_end_to_end.py` validates a queue-like worker flow by building a job payload compatible with `worker.py` expectations and calling the worker scorer dispatch path locally. It does not connect to real PGMQ unless the full worker loop is run separately.

## Queue-Like Payload Shape

The demo payload includes:

```json
{
  "job_id": "demo-job-id",
  "student_id": "demo-student",
  "target_word": "example",
  "target_text": "example",
  "prompt_text": "example",
  "audio_url": "<temp-or-local-audio>",
  "audio_path": "<temp-or-local-audio>",
  "canonical_phones": ["EH", "G", "Z", "AE", "M", "P", "AH", "L"]
}
```

## Environment Variables

The script sets context variables internally:

```powershell
$env:SCORER_MODE="cnn_attention_context"
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="C:\path\to\context-checkpoint.pt"
$env:CNN_ATTENTION_CONTEXT_MODE="context_0_10"
$env:CNN_ATTENTION_CONTEXT_LEFT_SECONDS="0.10"
$env:CNN_ATTENTION_CONTEXT_RIGHT_SECONDS="0.10"
```

For real POST:

```powershell
$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="<local-ai-webhook-secret>"
```

Do not commit secrets, `.env`, checkpoints, or audio artifacts.

## Dry-Run Command

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_worker_end_to_end.py --job-id demo-job-id
```

Expected output sections:

- `CONFIG`
- `JOB PAYLOAD`
- `SCORER RESULT SUMMARY`
- `VALIDATION`
- `WEBHOOK PAYLOAD SUMMARY`
- `POST RESULT`
- `NEXT VERIFY STEPS`

Expected dry-run result:

- inference runs if torch and checkpoint are available
- `ai_result_valid=True`
- `payload_valid=True`
- `post_attempted=False`
- context metadata is present under `feedback.ai_result.metadata`

## Optional Real POST

Use a real existing `practice_history` UUID:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_worker_end_to_end.py `
  --job-id <existing-practice-history-uuid> `
  --post `
  --webhook-url http://localhost:8000/practice/webhook/ai-result `
  --secret <local-ai-webhook-secret>
```

POST is opt-in only. The script validates the AI result and webhook payload before POSTing.

## Supabase Verification After POST

After a successful POST, verify the target `practice_history` row:

- `status=completed`
- `score` is set
- `problem_phonemes` is set
- `feedback.ai_result` exists
- `feedback.ai_result.scorer.name=cnn_attention_context`
- `feedback.ai_result.metadata.context_used=true`
- `feedback.ai_result.metadata.context_mode=context_0_10`
- `feedback.ai_result.metadata.crop_start_time` is present
- `feedback.ai_result.metadata.crop_end_time` is present

## Limitations

- This is a local queue-like simulation unless connected to real PGMQ through `worker.py`.
- The checkpoint is local only and must not be committed.
- Generated audio is only for plumbing validation.
- Fallback alignment is approximate and not real forced alignment.
- Heuristic score is not real GOP/CaGOP.
- Classifier confidence is not pronunciation correctness.
- The Phase 2 context model is a research candidate, not a fully final production pronunciation model.
