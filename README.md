# Pronunciation Assistant

Local demo setup for the FastAPI backend, Expo frontend web app, Supabase Auth/Storage/Postgres, and the practice flow.

## Prerequisites

- Git
- Node.js LTS
- npm
- Python 3.11 or 3.12
- Supabase project access

## Clone and checkout

```powershell
git clone <REPO_URL>
cd pronunciation-assistant
git checkout develop
```

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
- `public.profiles` has matching `id`, `email`, and `app_role`.
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
8. Use Swagger `POST /practice/webhook/ai-result` to simulate AI completion.
9. Open History to see the result.

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
