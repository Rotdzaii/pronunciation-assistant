# Phoenix v2 Deploy Config

## Purpose

This document standardizes the AI Worker deploy/runtime configuration for Phoenix v2 Stable.

Phoenix v2 is Deep Learning-first and production-safe. Deployment must use the selected CNN Attention context scorer with MFA alignment. This document does not train models, download datasets, modify checkpoints, modify frontend code, or modify the database.

## Phoenix v2 Stable Profile

Recommended deployment profile:

```dotenv
WORKER_MODE=loop
SCORER_MODE=cnn_attention_context
ALIGNMENT_MODE=mfa
MODEL_VERSION=phoenix_v2_stable
```

Recommended validation profile:

```dotenv
WORKER_MODE=once
SCORER_MODE=cnn_attention_context
ALIGNMENT_MODE=mfa
MODEL_VERSION=phoenix_v2_stable
```

Use `WORKER_MODE=loop` for deployed workers. Use `WORKER_MODE=once` for manual validation and controlled queue tests.

## Required Environment

Core worker configuration:

| Variable | Required | Phoenix v2 value | Notes |
|---|---:|---|---|
| `WORKER_MODE` | Yes | `loop` or `once` | `loop` for deployment, `once` for validation. |
| `SCORER_MODE` | Yes | `cnn_attention_context` | Required Deep Learning scorer for Phoenix v2 Stable. |
| `ALIGNMENT_MODE` | Yes | `mfa` | MFA provides forced alignment timing for localization. |
| `MODEL_VERSION` | Yes | `phoenix_v2_stable` | Added to worker metadata and webhook feedback. |
| `MODEL_CONFIDENCE_THRESHOLD` | No | `0.65` | Must be a number from `0` to `1`. |
| `WORKER_POLL_INTERVAL_SECONDS` | No | `1` | Loop-mode polling interval. |
| `WORKER_IDLE_BACKOFF_MAX_SECONDS` | No | `10` | Loop-mode idle backoff ceiling. |

Supabase and queue configuration:

| Variable | Required | Example | Notes |
|---|---:|---|---|
| `SUPABASE_URL` | Yes | `https://your-project.supabase.co` | Project URL. |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | `<service-role-key>` | Server-only secret. Never commit or expose it. |
| `QUEUE_NAME` | No | `practice_jobs` | Worker defaults to `practice_jobs` when unset. |

Storage bucket configuration:

| Variable | Required by worker | Example | Notes |
|---|---:|---|---|
| `PRACTICE_AUDIO_BUCKET` | No | `practice-audios` | Configured in the backend, not read directly by the current worker. Queue messages carry `audio_url`; treat it as sensitive when it is signed. |

Webhook configuration:

| Variable | Required | Example | Notes |
|---|---:|---|---|
| `NODE_WEBHOOK_URL` | Yes | `http://localhost:8000/practice/webhook/ai-result` | Current worker-required webhook URL. In this project it points to the FastAPI practice webhook despite the historical variable name. |
| `AI_WEBHOOK_SECRET` | Yes | `<webhook-secret>` | Sent as `x-ai-webhook-secret`. Never commit it. |

Model configuration:

| Variable | Required | Example | Notes |
|---|---:|---|---|
| `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` | Yes | `/models/l2_arctic_cnn_attention_context_0_10.pt` | Must point to a compatible local or mounted checkpoint. The worker fails safely when this is missing or invalid. |
| `CNN_ATTENTION_CONTEXT_MODE` | No | `context_0_10` | Context scorer mode. |
| `CNN_ATTENTION_CONTEXT_LEFT_SECONDS` | No | `0.10` | Left context window. |
| `CNN_ATTENTION_CONTEXT_RIGHT_SECONDS` | No | `0.10` | Right context window. |

Checkpoint files are local-only deploy artifacts. Do not commit `.pt`, `.pth`, or other model checkpoint files.

MFA configuration:

| Variable | Required for `ALIGNMENT_MODE=mfa` | Example | Notes |
|---|---:|---|---|
| `MFA_COMMAND` | No | `mfa` | Defaults to `mfa`. Can be an executable name or local path. |
| `MFA_CONDA_ENV` | No | `mfa` | When set, the worker runs MFA through `conda run -n <env>`. |
| `MFA_DICTIONARY_PATH` | Yes | `english_us_mfa` | MFA dictionary model name or local path. |
| `MFA_ACOUSTIC_MODEL_PATH` | Yes | `english_mfa` | MFA acoustic model name or local path. |
| `MFA_TEMP_DIR` | No | `/tmp/ai-worker-mfa` | Directory for temporary MFA corpus/output work. Defaults to the system temp directory. |
| `ALLOW_ALIGNMENT_FALLBACK` | No | `true` | Allows fallback alignment if MFA fails. Set to `false` when validating strict MFA behavior. |

MFA is used for alignment only. It supplies timing and localization metadata; it is not the scoring core and must not replace the CNN Attention context scorer.

## Example `.env`

Use `ai-worker/.env.example` as the committed placeholder template. Copy it to `ai-worker/.env` locally and replace placeholders with environment-specific values.

```dotenv
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key
NODE_WEBHOOK_URL=http://localhost:8000/practice/webhook/ai-result
AI_WEBHOOK_SECRET=replace-with-ai-webhook-secret
QUEUE_NAME=practice_jobs
WORKER_MODE=loop
WORKER_POLL_INTERVAL_SECONDS=1
WORKER_IDLE_BACKOFF_MAX_SECONDS=10
SCORER_MODE=cnn_attention_context
ALIGNMENT_MODE=mfa
MODEL_VERSION=phoenix_v2_stable
MODEL_CONFIDENCE_THRESHOLD=0.65
CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH=/absolute/local/path/to/l2_arctic_cnn_attention_context_0_10.pt
CNN_ATTENTION_CONTEXT_MODE=context_0_10
CNN_ATTENTION_CONTEXT_LEFT_SECONDS=0.10
CNN_ATTENTION_CONTEXT_RIGHT_SECONDS=0.10
MFA_COMMAND=mfa
MFA_CONDA_ENV=mfa
MFA_DICTIONARY_PATH=english_us_mfa
MFA_ACOUSTIC_MODEL_PATH=english_mfa
MFA_TEMP_DIR=
ALLOW_ALIGNMENT_FALLBACK=true
```

## Safety Notes

- Never commit `ai-worker/.env` or any runtime `.env` file.
- Never commit `SUPABASE_SERVICE_ROLE_KEY`.
- Never commit `AI_WEBHOOK_SECRET`.
- Never commit signed audio URLs or queue payloads containing signed URLs.
- Never commit checkpoint `.pt` or `.pth` files.
- Never commit local audio files, TextGrid files, MFA temporary directories, or deploy-specific mount paths.
- Keep `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` local to the machine or mounted in the deploy environment.
- Logs and docs must not include service-role keys, webhook secrets, signed URLs, checkpoint paths, local audio paths, or MFA temp output paths from real environments.

## Manual Validation Checklist

Before deployment:

- Compile worker source without walking `ai-worker/.venv`:

```powershell
python -m compileall ai-worker/worker.py ai-worker/app ai-worker/audio ai-worker/scorers ai-worker/scripts
```

- Set the Phoenix v2 validation profile:

```powershell
$env:WORKER_MODE="once"
$env:SCORER_MODE="cnn_attention_context"
$env:ALIGNMENT_MODE="mfa"
$env:MODEL_VERSION="phoenix_v2_stable"
```

- Run the worker once with one safe test queue message in `practice_jobs`.
- Verify the queue message contains `job_id`, `student_id`, `target_word`, and `audio_url`.
- Verify the worker posts to `NODE_WEBHOOK_URL` and the webhook returns 2xx.
- Verify the queue message is archived only after webhook success.
- Verify a missing or invalid `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` returns a failed result with `score=null`, `problem_phonemes=[]`, and a checkpoint error type.
- Verify webhook failure or non-2xx response leaves the queue message unarchived.
- Verify logs do not leak secrets, signed URLs, local checkpoint paths, local audio paths, TextGrid paths, or MFA temporary paths.

## Related Docs

- [Phoenix v2 Worker Hardening](PHOENIX_V2_WORKER_HARDENING.md)
- [Phoenix v2 Runtime Validation](PHOENIX_V2_RUNTIME_VALIDATION.md)
- [MFA Worker Once Validation](MFA_WORKER_ONCE_VALIDATION.md)
- [Backend Webhook Contract](BACKEND_WEBHOOK_CONTRACT.md)
