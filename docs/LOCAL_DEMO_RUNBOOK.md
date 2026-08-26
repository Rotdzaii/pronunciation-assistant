# Local Demo Runbook

This repository has three local services. The repository-root `package.json` is
not an Expo application and has no startup scripts. Do not run `npm install`,
`npm install expo`, or `npx expo start` at the root.

| Service | Working directory | Verified command | Port / queue |
| --- | --- | --- | --- |
| FastAPI backend | `fastapi-backend/` | `python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` | 8000 |
| Expo frontend | `frontend/` | `npx expo start -c` | 8081; Expo may fall back to 8082/8083 |
| AI worker | `ai-worker/` | `python worker.py` | `practice_jobs` PGMQ queue |

## Before starting

Create local environment files from their examples and supply only local,
non-committed values. Never paste a real token, service-role key, webhook secret,
or signed URL into documentation.

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant
copy fastapi-backend\.env.example fastapi-backend\.env
copy frontend\.env.example frontend\.env
copy ai-worker\.env.example ai-worker\.env
```

The backend reads `fastapi-backend/.env`; the worker reads `ai-worker/.env`.
`frontend/.env` may contain only `EXPO_PUBLIC_` values. Configure the worker's
`NODE_WEBHOOK_URL` as `http://localhost:8000/practice/webhook/ai-result` and
leave `QUEUE_NAME=practice_jobs` unless the backend queue configuration is
intentionally changed.

## Standard startup order

### Terminal 1 — Backend

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant\fastapi-backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

This uses the project's backend environment and
`fastapi-backend/requirements.txt`, not the root `requirements.txt`.

### Terminal 2 — Expo frontend

Expo must run inside `frontend/`, never at the repository root. The root package
file is not the Expo package file. Do not run `npm install expo` at root, and do
not run `npm audit fix --force` in normal setup.

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant\frontend
npm install
npx expo start -c
```

For a phone demo when the local network is unreliable, use Expo's tunnel. It
needs `@expo/ngrok` in `frontend/` if Expo prompts for it.

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant\frontend
npx expo start --tunnel -c
```

The Expo Go QR code is only for development/demo. Expo Go must be compatible
with the frontend's current Expo SDK (`~51.0.28`). Do not upgrade Expo SDK in
this setup workflow.

### Terminal 3 — AI worker

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant\ai-worker
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe worker.py
```

The current Windows MFA workflow is Conda-based: `MFA_RUNTIME=conda` and
`MFA_CONDA_ENV=aligner` in `ai-worker/.env`. Do not use WSL for the documented
local runtime. If Windows Smart App Control blocks an unsigned `_kalpy.pyd`,
MFA cannot start; resolve that Windows policy/trust issue before retrying.

After the three services start, run the health check.

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant
check_demo_health.bat
```

## Optional launcher

`run_server.bat` performs the same three starts in separate CMD windows. It
changes into each service directory before installing dependencies or running a
service; its Expo command is therefore never evaluated at repository root.

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant
run_server.bat
```

For a port conflict, inspect the listed process before stopping it. The reset
script stops listeners on 8000, 8081, 8082, and 8083.

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant
reset_demo_ports.bat
run_server.bat
```

## Optional public Cloudflare demo

`run_deploy.bat` launches the local services, uses the Cloudflare configuration
at `%USERPROFILE%\.cloudflared\config.yml`, and starts the `phoenix-demo`
tunnel. It supplies the public backend URL to its Expo child process; it does
not print secrets.

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant
run_deploy.bat
```

The tunnel expects `app.myphoenix.me` to route to frontend port 8081 and
`api.myphoenix.me` to route to backend port 8000. Use only `run_deploy.bat` for
this public-tunnel workflow; legacy launcher names and flags are unsupported.

## Troubleshooting

- Backend has a listener but `/health` fails: wait for Uvicorn reload to finish,
  then inspect the Terminal 1 traceback and `fastapi-backend/.env`.
- Expo cannot start: confirm the Terminal 2 working directory is `frontend/`;
  install dependencies there, not at root.
- A phone cannot open Expo: use the tunnel command above and verify Expo Go SDK
  compatibility.
- Worker cannot read jobs: verify the backend created a `practice_jobs` message,
  the worker uses the same Supabase project, and the exposed PGMQ read/archive
  RPCs exist.
- MFA fails on Windows: verify Conda can run the configured `aligner`
  environment, then check for a Smart App Control block of `_kalpy.pyd`.
