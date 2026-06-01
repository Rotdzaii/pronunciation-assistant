# Context CNN Attention Backend POST Verification

## Purpose

This document describes how to verify the `cnn_attention_context` AI Worker payload against the local FastAPI backend webhook with a real POST.

The script validates the AI result and webhook payload before POSTing. POST is opt-in only.

## Prerequisites

- FastAPI backend is running locally.
- Backend webhook route is available:
  - `POST /practice/webhook/ai-result`
- `AI_WEBHOOK_SECRET` matches the backend webhook secret.
- A real existing `practice_history` UUID is available for `--job-id`.
- AI Worker venv dependencies are installed:
  - `torch`
  - `librosa`
  - `soundfile`
  - `numpy`
  - `requests`
- A local context CNN Attention checkpoint exists.

Checkpoint files are local artifacts and must not be committed.

## Environment Variables

```powershell
$env:SCORER_MODE="cnn_attention_context"
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="C:\path\to\context-checkpoint.pt"
$env:CNN_ATTENTION_CONTEXT_MODE="context_0_10"
$env:CNN_ATTENTION_CONTEXT_LEFT_SECONDS="0.10"
$env:CNN_ATTENTION_CONTEXT_RIGHT_SECONDS="0.10"
$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="<local-ai-webhook-secret>"
```

Do not commit `.env` files or webhook secrets.

## Dry-Run Command

Run this first:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_backend_payload.py `
  --job-id demo-job-id
```

Expected dry-run result:

- Inference runs if torch and checkpoint are available.
- `ai_result_valid=True`
- `payload_valid=True`
- `post_attempted=False`
- `feedback.ai_result.metadata.context_used=true`
- `feedback.ai_result.metadata.context_mode=context_0_10`

The dry run may warn that `demo-job-id` is not UUID-shaped. That is expected for dry-run use.

## Real POST Command

Use a real existing `practice_history` UUID:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_backend_payload.py `
  --job-id <existing-practice-history-uuid> `
  --post `
  --webhook-url http://localhost:8000/practice/webhook/ai-result `
  --secret <local-ai-webhook-secret>
```

Expected script output:

- `ai_result_valid=True`
- `payload_valid=True`
- `expected_status=2xx`
- `post_success=True`
- `post_result=status_code=200 ...` or another 2xx response

The script does not print the secret.

## Supabase Verification

After a successful POST, verify the target `practice_history` row:

- `status=completed`
- `score` is set
- `problem_phonemes` is set
- `feedback.ai_result` exists
- `feedback.ai_result.metadata.context_used=true`
- `feedback.ai_result.metadata.context_mode=context_0_10`
- `feedback.ai_result.metadata.crop_start_time` is present
- `feedback.ai_result.metadata.crop_end_time` is present

Also confirm the output still marks:

- `model_output_is_scoring=false`
- `scoring_is_heuristic=true`
- `score_note` mentions heuristic or demo scoring
- confidence note says classifier confidence is not pronunciation correctness

## Context Metadata Location

The backend payload stores the rich AI result under:

```text
feedback.ai_result
```

Context metadata should be available at:

```text
feedback.ai_result.metadata.context_mode
feedback.ai_result.metadata.context_used
feedback.ai_result.metadata.context_left_seconds
feedback.ai_result.metadata.context_right_seconds
feedback.ai_result.metadata.segment_start_time
feedback.ai_result.metadata.segment_end_time
feedback.ai_result.metadata.crop_start_time
feedback.ai_result.metadata.crop_end_time
```

## Troubleshooting

Missing torch:

- Reinstall or repair the AI Worker inference dependencies in `ai-worker/.venv`.
- The validation environment previously used `torch 2.12.0+cpu`.

Missing checkpoint:

- Set `--checkpoint-path` or `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH`.
- Do not commit the checkpoint.

Secret mismatch:

- Confirm backend and worker use the same `AI_WEBHOOK_SECRET`.
- Do not print or commit the secret.

Job id not found:

- Use an existing `practice_history.id`.
- Dry-run can use a fake id; real POST should use a real UUID.

Backend not running:

- Start FastAPI locally before POSTing.
- Confirm the webhook URL and port.

Payload invalid:

- Do not POST invalid payloads.
- Re-run the dry-run command and inspect `payload_issues`.

Fallback alignment limitations:

- Fallback alignment is approximate and not real forced alignment.
- Context crop quality depends on segment boundary quality.

Scoring limitations:

- Heuristic score is not real GOP/CaGOP.
- Classifier confidence is not pronunciation correctness.
