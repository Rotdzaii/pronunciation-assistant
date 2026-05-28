# Backend POST Verification

## Purpose

Use this guide to safely verify the local path:

```text
AI Worker sample result -> FastAPI webhook -> practice_history update
```

The verification uses `ai-worker/scripts/demo_backend_integration.py`. Dry-run mode never sends a network request. Real POST mode is opt-in with `--post`.

## Prerequisites

- FastAPI backend is running locally.
- Backend `.env` has `AI_WEBHOOK_SECRET` configured.
- The worker POST uses the same secret through `AI_WEBHOOK_SECRET` or `--secret`.
- A `practice_history` row already exists for the test `job_id`.
- The test job is safe to update, preferably with `status = processing`.
- The real POST `job_id` is a UUID, because the FastAPI request model validates it as `UUID`.

Do not commit `.env` files or print secret values in logs, screenshots, or docs.

## Find Or Create A Test Job

Option 1: create a normal practice job through the backend:

```powershell
curl.exe -X POST `
  -H "Authorization: Bearer <student-supabase-access-token>" `
  -H "Content-Type: application/json" `
  -d "{\"target_word\":\"Architecture\",\"audio_url\":\"https://example.test/audio.wav\"}" `
  http://localhost:8000/practice/create-job
```

Use the returned `job_id`.

Option 2: use Supabase local/project data and pick a row from `public.practice_history` with `status = processing`.

## Dry Run

Dry-run validates and prints the sample webhook payload. It does not require the backend to be running.

```powershell
python ai-worker/scripts/demo_backend_integration.py --job-id demo-job-id --dry-run
```

For a backend-compatible dry-run, use a UUID-shaped job id:

```powershell
python ai-worker/scripts/demo_backend_integration.py --job-id 11111111-1111-1111-1111-111111111111 --dry-run
```

## Real POST

Prefer environment variables so the secret is not exposed in command history:

```powershell
$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="<local-ai-webhook-secret>"
python ai-worker/scripts/demo_backend_integration.py --job-id <existing-practice-job-uuid> --post
```

With an optional backend health probe:

```powershell
python ai-worker/scripts/demo_backend_integration.py `
  --job-id <existing-practice-job-uuid> `
  --post `
  --check-backend-health http://localhost:8000/health
```

Equivalent explicit URL command:

```powershell
python ai-worker/scripts/demo_backend_integration.py `
  --job-id <existing-practice-job-uuid> `
  --post `
  --webhook-url http://localhost:8000/practice/webhook/ai-result
```

The script reads the webhook URL from `NODE_WEBHOOK_URL` first, then `AI_WEBHOOK_URL`.

## Expected Successful Response

Default expected POST status is `200`.

```json
{
  "job_id": "<existing-practice-job-uuid>",
  "status": "completed",
  "message": "Practice job result updated"
}
```

Use `--expected-status-code` only if the backend contract intentionally changes.

## Verify The Result

Use the backend API with a valid Supabase access token:

```powershell
curl.exe -H "Authorization: Bearer <supabase-access-token>" `
  http://localhost:8000/practice/<existing-practice-job-uuid>
```

Or inspect `public.practice_history` in Supabase.

Expected row changes:

- `status = completed`
- `score` is populated
- `problem_phonemes` is a JSON array
- `feedback.summary` and `feedback.tips` exist
- `feedback.ai_result` stores the richer normalized AI result
- `updated_at` changed

## Troubleshooting

`401` or `403`: secret mismatch. Confirm backend `AI_WEBHOOK_SECRET` and the worker `AI_WEBHOOK_SECRET` refer to the same local value.

`404`: wrong route or unknown job id. Confirm `/practice/webhook/ai-result` and an existing `practice_history.id`.

`422` or `400`: payload mismatch. Run dry-run and confirm `payload_valid=True`. For real POST, make sure `job_id` is a UUID.

Connection refused or timeout: backend is not running, the port is wrong, or the health URL/webhook URL points to the wrong service.

No database update: check backend logs, Supabase service-role configuration, and whether the row exists in `public.practice_history`.

## Warnings

- Do not commit `.env` files, secrets, checkpoints, audio, raw datasets, archives, or generated WAV files.
- `feedback.ai_result` stores rich AI output for debugging and research.
- The demo score is heuristic. It is not production GOP.
- Fallback alignment is approximate and must not be presented as precise forced alignment.
- Classifier confidence must not be used as the pronunciation score.
