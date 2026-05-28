# Backend Integration Test

## Purpose

This document describes how to safely test:

```text
AI Worker final output -> backend webhook -> practice_history update
```

The test uses `ai-worker/scripts/demo_backend_integration.py`. It is dry-run by default and sends no network request unless `--post` is explicitly supplied.

## Backend Webhook Details

FastAPI route:

```text
POST /practice/webhook/ai-result
```

Required secret header:

```text
x-ai-webhook-secret
```

Current accepted payload fields:

- `job_id`
- `status`
- `score`
- `problem_phonemes`
- `feedback`

The backend updates `practice_history.status`, `score`, `problem_phonemes`, `feedback`, and `updated_at`.

`feedback` is JSONB, so `feedback.ai_result` from the worker payload can preserve the full normalized AI result without a backend schema change.

## Prerequisites For POST Mode

- FastAPI backend is running.
- `AI_WEBHOOK_SECRET` matches the backend configuration.
- A `practice_history` row exists for the supplied `job_id`.
- The row is safe to update, usually with `status = processing`.

The demo does not create a practice job. Use an existing test job id from local development.

## Dry Run

```powershell
python ai-worker/scripts/demo_backend_integration.py --job-id demo-job-id --dry-run
```

Dry-run validates and prints the payload but does not POST.

## Actual POST

Using env vars:

```powershell
$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="replace-with-local-secret"
python ai-worker/scripts/demo_backend_integration.py --job-id <existing-job-id> --post
```

Using CLI args:

```powershell
python ai-worker/scripts/demo_backend_integration.py --job-id <existing-job-id> --post --webhook-url http://localhost:8000/practice/webhook/ai-result --secret <local-secret>
```

The script never prints the secret value.

## Expected Backend Result

For an existing job id, the backend should return a 2xx response similar to:

```json
{
  "job_id": "<existing-job-id>",
  "status": "completed",
  "message": "Practice job result updated"
}
```

Verify through the database or API:

- `practice_history.status = completed`
- `practice_history.score` is populated
- `practice_history.problem_phonemes` is JSON array
- `practice_history.feedback.summary` and `feedback.tips` exist
- `practice_history.feedback.ai_result` preserves the full normalized result

## Limitations

- The demo uses a sample payload unless connected to a real worker job.
- Real CNN Attention inference requires local `torch` dependencies and checkpoint.
- `heuristic_gop` is not real GOP.
- Fallback alignment is approximate and not real forced alignment.
- MFA execution is scaffolded only.

## Troubleshooting

`401` or `403`: webhook secret mismatch. Check `AI_WEBHOOK_SECRET` and `x-ai-webhook-secret`.

`404`: route mismatch or `job_id` not found. Confirm `/practice/webhook/ai-result` and an existing `practice_history.id`.

`422`: payload validation issue. Run dry-run and check `payload_valid=True`.

Connection refused: backend is not running or URL/port is wrong.

No database update: confirm the backend service-role Supabase configuration and check backend logs.
