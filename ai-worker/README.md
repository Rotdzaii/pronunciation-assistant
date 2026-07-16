# AI Worker

The worker polls Supabase PGMQ for pronunciation jobs, scores them, posts a result
to the FastAPI webhook, and archives a message only after a successful webhook
response.

## Canonical local runtime

Use the virtual environment inside `ai-worker/`; do not use the backend venv or
the root `requirements.txt`. The worker command is `python worker.py`, and it
loads `ai-worker/.env`.

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant\ai-worker
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python.exe worker.py
```

Set only local values in `.env`; never commit or print service-role keys,
webhook secrets, signed URLs, checkpoints, audio, or other secrets.

The required connection values are `SUPABASE_URL`,
`SUPABASE_SERVICE_ROLE_KEY`, `NODE_WEBHOOK_URL`, and `AI_WEBHOOK_SECRET`.
The standard local queue is `QUEUE_NAME=practice_jobs`; `WORKER_MODE=loop`
keeps the worker polling.

## MFA on Windows

The current documented MFA workflow is Windows Conda, not WSL. Keep these
settings aligned with `.env.example` when forced alignment is enabled:

```dotenv
MFA_RUNTIME=conda
MFA_COMMAND=mfa
MFA_CONDA_ENV=aligner
MFA_DICTIONARY_PATH=english_us_mfa
MFA_ACOUSTIC_MODEL_PATH=english_mfa
```

If Smart App Control blocks an unsigned `_kalpy.pyd`, the Conda MFA runtime
cannot run. Resolve the Windows trust/policy issue before starting the worker;
do not switch to WSL as a workaround in this local workflow.

## Scorer dependencies

`ai-worker/requirements.txt` is the dependency file used by the local worker
venv and launcher. It does not currently declare PyTorch, while
`SCORER_MODE=cnn_attention_context` needs the local inference dependencies at
job-processing time. The launcher now stops with a clear message if a configured
CNN scorer cannot import PyTorch; it does not install undeclared dependencies or
change package files. A reproducible local inference dependency source is not
confirmed by the tracked worker requirements.

## Run modes

The default is `WORKER_MODE=loop`. For an intentional one-message check, set
the process environment only for that PowerShell session:

**PowerShell**

```powershell
$env:WORKER_MODE = "once"
.\.venv\Scripts\python.exe worker.py
```

The worker reads a message from `practice_jobs`, invokes the selected
`SCORER_MODE`, posts to `POST /practice/webhook/ai-result`, and archives only
after a 2xx response. It does not log secrets.

## Documentation

- [AI Worker Pipeline Summary](docs/AI_WORKER_PIPELINE_SUMMARY.md)
- [AI Worker Integration Status](docs/AI_WORKER_INTEGRATION_STATUS.md)
- [Final AI Output Contract](docs/FINAL_AI_OUTPUT_CONTRACT.md)
- [Backend Webhook Contract](docs/BACKEND_WEBHOOK_CONTRACT.md)
- [MFA alignment setup](docs/MFA_ALIGNMENT_SETUP.md)
- [End-to-End Worker Demo](docs/END_TO_END_WORKER_DEMO.md)
