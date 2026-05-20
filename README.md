# Pronunciation Assistant

Local demo setup for the FastAPI backend, Expo frontend web app, Supabase Auth/Storage/Postgres, and the practice flow.

## Active project structure

- `frontend/` = active Expo frontend.
- `fastapi-backend/` = active FastAPI backend.
- `ai-worker/` = lightweight queue worker with mock and Wav2Vec2 baseline scorers that calls the FastAPI webhook.
- `archive/legacy/` = old reference code only.

## Prerequisites

- Git
- Node.js LTS
- npm
- Python 3.11 or 3.12
- Supabase project access
- FFmpeg for decoding browser/mobile audio formats such as `webm` and `m4a` in Wav2Vec2 mode

On Windows, install FFmpeg with:

```powershell
winget install Gyan.FFmpeg
# or
choco install ffmpeg
```

## Clone and checkout

```powershell
git clone <REPO_URL>
cd pronunciation-assistant
git checkout develop
```

## Start the full local demo

Create the backend, frontend, and worker env files first:

```powershell
copy fastapi-backend\.env.example fastapi-backend\.env
copy frontend\.env.example frontend\.env
copy ai-worker\.env.example ai-worker\.env
```

Fill the Supabase backend secrets in `fastapi-backend/.env`, the frontend-safe `EXPO_PUBLIC_` values in `frontend/.env`, and the AI worker service-role/webhook settings in `ai-worker/.env`. Do not print or commit secrets.

Then double-click `run_server.bat` from the repo root, or run:

```powershell
.\run_server.bat
```

The script can resume a partially running demo. On each run it checks the backend, frontend, and AI worker separately. If a service already appears to be running, it skips that service and only starts missing services instead of opening duplicate windows.

The script opens service terminal windows as needed:

- `FastAPI Backend`: creates `fastapi-backend/.venv` if needed, installs backend requirements on first setup, and starts `http://localhost:8000`.
- `Expo Frontend`: installs frontend dependencies if needed, sets `EXPO_PUBLIC_API_BASE_URL=http://localhost:8000` for that terminal session, and starts Expo Web.
- `AI Worker`: creates `ai-worker/.venv` if needed, installs worker requirements on first setup, activates the venv, and runs `python worker.py`.

If `ai-worker/.env` is missing, the script prints a warning and starts only the backend and frontend. The worker should normally use `WORKER_MODE=loop` for the local demo.

Detection rules:

- Backend is considered running when TCP port `8000` is listening.
- Frontend is considered running when one of Expo web ports `8081`, `8082`, or `8083` is listening.
- AI worker is considered running when a Python process command line includes `ai-worker/worker.py`.

To manually stop old services, close the terminal window for that service or press `Ctrl+C` in it. If a service was started in another terminal, find the process using `netstat -ano` for ports or Task Manager for Python/Node processes, then stop only the old process you no longer need.

## Backend setup

```powershell
cd fastapi-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
```

Fill backend-only secrets in `fastapi-backend/.env`:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_JWT_SECRET`
- `AI_WEBHOOK_SECRET`
- `PRACTICE_AUDIO_BUCKET=practice-audios`

Start the API:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Frontend setup

Open a new terminal:

```powershell
cd frontend
npm install
copy .env.example .env
```

Fill only frontend-safe Expo variables in `frontend/.env`:

```dotenv
EXPO_PUBLIC_API_BASE_URL=http://localhost:8000
EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
```

Run Expo web:

```powershell
npm run web
```

Open `http://localhost:8081`.

## Supabase setup checklist

- Authentication users exist.
- Registration sends the selected role as Supabase `user_metadata.app_role`.
- Apply the database migrations so the auth trigger creates `public.profiles` automatically with matching `id`, `email`, and `app_role`.
- `public.practice_history` exists.
- `practice-audios` storage bucket exists.
- PGMQ `practice_jobs` queue/RPC exists if using the queue path.

## Demo flow

1. Start the backend.
2. Start the frontend.
3. Login with a student user.
4. Record audio.
5. Replay audio.
6. Submit to AI scoring.
7. Confirm backend calls in the network/API logs:
   - `POST /practice/upload-audio`
   - `POST /practice/create-job`
   - `GET /practice/{job_id}`
8. The AI worker window processes the queued `practice_jobs` message automatically. You can also set `WORKER_MODE=once` and run `python worker.py` manually, or use Swagger `POST /practice/webhook/ai-result` to simulate AI completion manually.
9. Open History to see the result.

## AI worker modes

- `WORKER_MODE=once`: read one queue job and exit. This preserves the manual `python worker.py` workflow.
- `WORKER_MODE=loop`: continuously poll the queue, process available jobs immediately, and stay idle between uploads.

Loop mode keeps latency low by polling every `WORKER_POLL_INTERVAL_SECONDS` while active. When the queue is empty, it backs off gradually up to `WORKER_IDLE_BACKOFF_MAX_SECONDS` so it does not call Supabase many times per second while idle. `WORKER_MAX_JOBS_PER_RUN=0` means unlimited jobs until Ctrl+C.

With `SCORER_MODE=wav2vec2`, the first worker run may be slow because the model is downloaded and cached. CPU mode is acceptable for the baseline, but slower than GPU. Wav2Vec2 is a pretrained ASR baseline: it compares the recognized transcript with the target word for demo scoring, and it is not the final pronunciation diagnosis model. Its score is calibrated with `WAV2VEC2_BASELINE_MAX_SCORE=92` by default because a manual reference check showed Azure Pronunciation Assessment around `88%` while uncapped Wav2Vec2 text-match scoring could produce `100`.

## Current API endpoints

- `GET /health`
- `GET /auth/me`
- `POST /practice/upload-audio`
- `POST /practice/create-job`
- `GET /practice/{job_id}`
- `POST /practice/webhook/ai-result`
- `GET /practice/history`

## Common troubleshooting

- Frontend calls port 3000 instead of 8000: check `EXPO_PUBLIC_API_BASE_URL`, then restart Expo with a clear cache.
- Invalid login credentials: check the Supabase Auth user, password, and email confirmation state.
- `Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.`: clear browser `localStorage` and login again.
- `python-multipart` missing: run `python -m pip install -r requirements.txt`.
- `uvicorn` not recognized: use `python -m uvicorn`.
- VS Code auto activates the wrong `.venv`: set `"python.terminal.activateEnvironment": false`.
- Expo says `expo` is not installed: run `npm install` inside `frontend`.

To clear Expo cache:

```powershell
npx expo start --web --clear
```

## Git workflow

- Create feature branches from `develop`.
- Do not commit `.env`, `.venv`, `node_modules`, or generated caches.
- Commit convention examples:
  - `feat(backend): add practice job flow`
  - `fix(frontend): correct API base URL`
  - `chore(docs): update demo setup guide`

## Verification

Backend:

```powershell
cd fastapi-backend
python -m compileall app
```

Frontend:

```powershell
cd frontend
npm run lint
```
