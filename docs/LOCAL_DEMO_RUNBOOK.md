# Local Demo Runbook

Quick reference for running the Pronunciation Assistant demo on a local machine.

---

## Services and ports

| Service | Command | Port |
|---------|---------|------|
| FastAPI Backend | `uvicorn app.main:app --reload` | 8000 |
| Expo Frontend (web) | `expo start --web` | 8081 (fallback 8082, 8083) |
| AI Worker | `python worker.py` | — (polls DB queue) |
| Cloudflare Tunnel | `cloudflared tunnel run phoenix-demo` | — |

Public domain mapping (Cloudflare):
- `app.myphoenix.me` → `http://localhost:8081`
- `api.myphoenix.me` → `http://localhost:8000`

---

## Required .env files

Create these files before running the demo. Never commit them.

### `fastapi-backend/.env`

```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
AI_WEBHOOK_SECRET=<secret>
APP_NAME=PronunciationAssistant
CORS_ORIGINS=["http://localhost:8081","https://app.myphoenix.me"]
```

### `frontend/.env`

For local demo:
```
EXPO_PUBLIC_API_BASE_URL=http://localhost:8000
```

For public Cloudflare demo:
```
EXPO_PUBLIC_API_BASE_URL=https://api.myphoenix.me
```

> Changing `EXPO_PUBLIC_API_BASE_URL` requires restarting the Expo dev server
> (or rebuilding the Docker image).

### `ai-worker/.env`

```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
NODE_WEBHOOK_URL=http://localhost:8000/practice/webhook/ai-result
AI_WEBHOOK_SECRET=<secret>
SCORER_MODE=cnn_attention_context
CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH=./checkpoints/l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_42_HQTV.pt
ALIGNMENT_MODE=fallback
```

---

## Running the demo

### Option 1 — Standard local demo

```bat
.\run_server.bat
```

Opens three terminal windows: Backend, Frontend, AI Worker.

### Option 2 — With Cloudflare tunnel

```bat
.\run_demo_with_tunnel.bat
```

Equivalent to `run_server.bat --with-tunnel`. Requires `cloudflared` in PATH and
`~/.cloudflared/config.yml` configured for the `phoenix-demo` tunnel.

### Option 3 — Start individual service

```bat
.\run_server.bat backend
.\run_server.bat frontend
.\run_server.bat ai-worker
```

Useful for debugging a single service without opening all windows.

---

## Health check

```bat
.\check_demo_health.bat
```

Output example:
```
[OK]   Backend running on http://localhost:8000  (PID 1234 / uvicorn.exe)
       GET /health -> 200 OK
[OK]   Frontend running on http://localhost:8081  (PID 5678 / node.exe)
[OK]   AI Worker process found (PID 9012)
[WARN] Cloudflare tunnel not running.
```

---

## Killing stuck ports

If a service fails to start because the port is already in use:

```bat
.\reset_demo_ports.bat
```

This kills processes on ports 8000, 8081, 8082, 8083 and shows PID + process
name before killing. Then rerun `run_server.bat`.

Alternatively:
```bat
.\run_server.bat --force-kill
```

---

## Debugging individual services

### Backend not starting

1. Run `.\run_server.bat backend` to see the full error output.
2. Check `fastapi-backend/.env` exists and has valid Supabase credentials.
3. Verify Python is installed: `python --version`
4. If the venv activation fails, delete `fastapi-backend/.venv` and rerun — the
   launcher recreates it automatically.
5. Check for import errors in the uvicorn startup log. Common cause: missing
   package → the launcher now always runs `pip install` to catch this.

### Frontend not starting

1. Run `.\run_server.bat frontend` to see the full error output.
2. Check `frontend/.env` exists with `EXPO_PUBLIC_API_BASE_URL`.
3. Verify Node.js is installed: `node --version` (requires ≥ 18).
4. If Expo binary is missing: `cd frontend && npm install`
5. If Metro bundler hangs, press `w` in the Expo terminal to open the web build.
6. Port conflict: run `reset_demo_ports.bat` then restart.

### AI Worker not starting

1. Run `.\run_server.bat ai-worker` to see the full error output.
2. Check `ai-worker/.env` exists.
3. Confirm the checkpoint file exists:
   `ai-worker/checkpoints/l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_42_HQTV.pt`
4. If torch is missing, the launcher installs it automatically. The first install
   takes ~3 min (torch CPU wheel is ~185 MB).
5. If `SCORER_MODE=cnn_attention_context`, the worker loads the CNN model on
   startup. Allow ~10s for model load before the first job is processed.

### Cloudflare tunnel not starting

1. Confirm `cloudflared` is in PATH: `cloudflared --version`
2. Check the config: `type %USERPROFILE%\.cloudflared\config.yml`
3. Re-authenticate if the cert is expired: `cloudflared tunnel login`
4. Confirm the tunnel exists: `cloudflared tunnel list`
5. The tunnel name used is `phoenix-demo`.

---

## Demo quick-start (before thesis defense)

```bat
cd C:\Users\Admin\Documents\KLTN\pronunciation-assistant

rem 1. Start all services
.\run_server.bat

rem 2. Verify everything is up
.\check_demo_health.bat

rem 3. (Optional) Expose publicly
.\run_demo_with_tunnel.bat

rem 4. Open demo in browser
start http://localhost:8081
```

Expected latency for AI scoring (steady-state, CPU-only Docker):
- ~4–9 s (warm), ~12–14 s (first few requests after restart)

---

## File inventory

| File | Purpose |
|------|---------|
| `run_server.bat` | Main launcher — opens Backend, Frontend, AI Worker |
| `run_sever.bat` | Typo-safe wrapper — delegates to `run_server.bat` |
| `run_demo_with_tunnel.bat` | Launcher + Cloudflare tunnel |
| `reset_demo_ports.bat` | Kill processes on demo ports |
| `check_demo_health.bat` | Health check for all services |
| `docs/LOCAL_DEMO_RUNBOOK.md` | This file |
