# Pronunciation Assistant

Local demo setup for the FastAPI backend, Expo frontend web app, Supabase Auth/Storage/Postgres, and the practice flow.

## Active project structure

- `frontend/` = active Expo frontend.
- `fastapi-backend/` = active FastAPI backend.
- `ai-worker/` = lightweight queue worker that scores queued practice jobs and calls the FastAPI webhook.
- `archive/legacy/` = old reference code only.

`package.json` at the repository root is not the Expo app and has no project
startup scripts. Do not run `npm install`, `npm install expo`, `npx expo start`,
or `npm audit fix --force` at the repository root as part of normal setup.
The Expo app is only `frontend/`; do not upgrade its Expo SDK in this setup
workflow.

## Prerequisites

- Git
- Node.js LTS
- npm
- Python 3.11 or 3.12
- Supabase project access

## Clone and checkout

**PowerShell**

```powershell
git clone <REPO_URL>
cd pronunciation-assistant
git checkout develop
```

## Start the local demo

Create the backend and frontend env files first:

**Windows CMD**

```cmd
copy fastapi-backend\.env.example fastapi-backend\.env
copy frontend\.env.example frontend\.env
copy ai-worker\.env.example ai-worker\.env
```

Fill the Supabase backend secrets in `fastapi-backend/.env` and the frontend-safe `EXPO_PUBLIC_` values in `frontend/.env`.

Then double-click `run_server.bat` from the repo root, or run:

**Windows CMD**

```cmd
.\run_server.bat
```

The launcher opens three terminal windows (the worker is skipped only when
`ai-worker/.env` is absent):

- `FastAPI Backend`: creates `fastapi-backend/.venv` if needed, installs backend requirements on first setup, and starts `http://localhost:8000`.
- `Expo Frontend`: changes into `frontend/`, installs that app's dependencies if needed, sets `EXPO_PUBLIC_API_BASE_URL=http://localhost:8000` for that terminal session, and starts Expo Web.
- `AI Worker`: creates and uses `ai-worker/.venv`, installs `ai-worker/requirements.txt`, and runs `python worker.py`.

If port `8000` is busy, close the old backend process or restart your machine, then run the script again.

## Chạy bằng Docker

Yêu cầu: Docker Desktop (hoặc Docker Engine + Compose plugin).

### Build và khởi động

**Bash/Linux**

```bash
docker compose up --build
```

Lệnh này build một Docker image duy nhất chứa cả `fastapi-backend` lẫn `ai-worker`,
rồi khởi chạy 2 container riêng biệt.

`requirements.txt` ở root là dependency chung chỉ cho Docker image này. Khi
chạy thủ công, vẫn dùng `fastapi-backend/requirements.txt` cho backend và
`ai-worker/requirements.txt` cho worker.

### Biến môi trường

Tạo file `.env` ở thư mục gốc (cùng cấp với `docker-compose.yml`) với các giá trị sau:

```dotenv
# Supabase — bắt buộc cho cả backend và ai-worker
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=

# Webhook auth secret — phải khớp giữa backend và ai-worker
AI_WEBHOOK_SECRET=

# ai-worker — NODE_WEBHOOK_URL được docker-compose tự hardcode thành
# http://backend:8000/practice/webhook/ai-result khi chạy trong container,
# KHÔNG cần điền trong .env

# ai-worker tuỳ chọn (có default)
# docker-compose.yml đặt SCORER_MODE=cnn_attention_context cho worker container.
# Không cần đặt SCORER_MODE ở đây trừ khi file compose được thay đổi có chủ đích.
WORKER_MODE=loop
QUEUE_NAME=practice_jobs
MODEL_CONFIDENCE_THRESHOLD=0.65
WORKER_POLL_INTERVAL_SECONDS=1.0
WORKER_IDLE_BACKOFF_MAX_SECONDS=10.0
QUEUE_VISIBILITY_TIMEOUT_SECONDS=60
ALIGNMENT_MODE=fallback
MODEL_VERSION=phoenix_v2_stable

# fastapi-backend tuỳ chọn (có default)
APP_ENV=production
PRACTICE_AUDIO_BUCKET=practice-audios
BACKEND_CORS_ORIGINS=http://localhost:8081
```

### Build frontend với URL backend khác (tunnel / production)

`EXPO_PUBLIC_API_BASE_URL` được **bake vào bundle lúc build** (không đọc lại lúc runtime),
nên phải truyền vào trước khi build image:

**Windows CMD**
```cmd
set EXPO_PUBLIC_API_BASE_URL=https://<url-tunnel-backend>
docker compose build frontend
```

**Bash/Linux**
```bash
EXPO_PUBLIC_API_BASE_URL=https://<url-tunnel-backend> docker compose build frontend
```

Nếu chạy local bình thường (backend tại `localhost:8000`) thì **không cần set**,
default `http://localhost:8000` sẽ được dùng tự động.

### Ghi chú về supabase version

`ai-worker/requirements.txt` ban đầu ghim `supabase==2.10.0` trong khi `fastapi-backend`
yêu cầu `supabase>=2.11.0`. Đã nâng ai-worker lên `supabase>=2.11.0,<3.0.0` để gộp được
một image. Đã verify an toàn bằng cách: (1) đọc source thực của cả hai phiên bản từ venv
riêng biệt, (2) xác nhận `_sync/client.py` và `execute()` của postgrest 0.18→0.19 không
thay đổi interface mà worker sử dụng (`client.rpc().execute()`, không có `.table()`,
không truy cập `response.error`).

## Manual startup order

Use the launcher above for the usual local demo. The following manual commands
are the canonical debugging path; run each terminal from the listed directory.

### Terminal 1 — Backend

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant\fastapi-backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`requirements.txt` in this command is
`fastapi-backend/requirements.txt`, not the root `requirements.txt`.

### Terminal 2 — Frontend Expo

Expo must be started from `frontend/`, never from the repository root.

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant\frontend
npm install
npx expo start -c
```

If testing with Expo Go over the local network is unreliable, use a tunnel.
Install `@expo/ngrok` in `frontend/` only if Expo asks for it; the tunnel uses
that package.

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant\frontend
npx expo start --tunnel -c
```

The Expo Go QR code is for development/demo only. The installed Expo Go client
must support this project's Expo SDK (`~51.0.28`); do not change the SDK for
this setup. Do not run `npm install expo` at the repository root and do not run
`npm audit fix --force` in the normal setup flow.

### Terminal 3 — AI Worker

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant\ai-worker
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe worker.py
```

The worker loads `ai-worker/.env`, polls `QUEUE_NAME=practice_jobs` by default,
and posts to `NODE_WEBHOOK_URL`. For the current MFA workflow, use the local
Windows Conda runtime configured as `MFA_RUNTIME=conda` and
`MFA_CONDA_ENV=aligner`; this runbook does not prescribe WSL.

After all three terminals are running, use `check_demo_health.bat`; start the
Expo tunnel only when it is needed.

## Backend setup

**PowerShell**

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

Use the Terminal 2 Windows CMD commands above. They are intentionally absolute
so Expo cannot be run from the repository root.

Fill only frontend-safe Expo variables in `frontend/.env`:

```dotenv
EXPO_PUBLIC_API_BASE_URL=http://localhost:8000
EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
```

For web-only development, `npm run web` is a valid `frontend/package.json`
script when executed inside `frontend/`; it is not a root command.

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
8. Run `python worker.py` from `ai-worker/` to process one queued `practice_jobs` message, or use Swagger `POST /practice/webhook/ai-result` to simulate AI completion manually.
9. Open History to see the result.

For the final context CNN Attention flow, use [Demo Context AI Flow Checklist](docs/DEMO_CONTEXT_AI_FLOW_CHECKLIST.md) and [Demo Context AI Flow Validation Result](docs/DEMO_CONTEXT_AI_FLOW_VALIDATION_RESULT.md).

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
- `python-multipart` missing: in `fastapi-backend/`, run `python -m pip install -r requirements.txt`.
- `uvicorn` not recognized: use `python -m uvicorn`.
- VS Code auto activates the wrong `.venv`: set `"python.terminal.activateEnvironment": false`.
- Expo says `expo` is not installed: run `npm install` inside `frontend`.

To clear the Expo cache, use the Terminal 2 command `npx expo start -c` from
`frontend/`.

## Git workflow

- Create feature branches from `develop`.
- Do not commit `.env`, `.venv`, `node_modules`, or generated caches.
- Commit convention examples:
  - `feat(backend): add practice job flow`
  - `fix(frontend): correct API base URL`
  - `chore(docs): update demo setup guide`

## Verification

Backend:

**PowerShell**

```powershell
cd fastapi-backend
python -m compileall app
```

Frontend:

**PowerShell**

```powershell
cd frontend
npm run lint
```

## Supabase email confirmation and unconfirmed-account cleanup

Configure Supabase Dashboard → Authentication → URL Configuration with these
Redirect URLs:

```text
http://localhost:8081/callback
https://app.myphoenix.me/callback
```

For native builds, also allow `pronunciation-assistant://callback`. In
Authentication → Email Templates → Confirm signup, use `{{ .ConfirmationURL }}`
rather than linking only to `{{ .SiteURL }}`. Set **Email OTP Expiration** to
**3600 seconds** in the Authentication email settings.

The app allows a confirmation-email resend after 60 seconds. Unconfirmed users
are retained for 24 hours. The cleanup function will only consider users whose
`email_confirmed_at` and `last_sign_in_at` are both null and whose `created_at`
is more than 24 hours old. Confirmed users, users who have signed in, and admins
are excluded.

### Deploy and preview cleanup

1. Deploy the private Edge Function:

   ```bash
   supabase functions deploy cleanup-unconfirmed-users
   ```

2. In Supabase SQL Editor, store the service-role key in Vault. Do not place it
   in the frontend or in a migration:

   ```sql
   select vault.create_secret(
     'SERVICE_ROLE_KEY_VALUE',
     'cleanup_unconfirmed_users_service_role_key'
   );
   ```

3. Apply `fastapi-backend/db/migrations/015_schedule_unconfirmed_user_cleanup.sql`.
   It verifies that `public.profiles.id` cascades on deletion and schedules an
   hourly **dry run** at minute zero. The function is protected by the
   service-role authorization header; no browser-facing cleanup endpoint is
   available.

Preview the exact candidate set before enabling any real deletion:

```sql
select
  id,
  email,
  created_at,
  email_confirmed_at,
  last_sign_in_at
from auth.users
where email_confirmed_at is null
  and last_sign_in_at is null
  and created_at < now() - interval '24 hours'
order by created_at;
```

Check Edge Function logs and the Cron job result while it is in dry-run mode.
Only after reviewing this preview, enable deletion with:

```sql
select cron.alter_job(
  job_id := (select jobid from cron.job where jobname = 'cleanup-unconfirmed-users-hourly'),
  command := 'select private.invoke_cleanup_unconfirmed_users(false);'
);
```

### Rollback

Stop scheduled invocations immediately with:

```sql
select cron.unschedule(jobid)
from cron.job
where jobname = 'cleanup-unconfirmed-users-hourly';
```

Hard-deleted Auth users cannot be restored by this function. Restore an
accidentally deleted user only from an approved Supabase backup/PITR procedure;
then keep the Cron job in dry-run mode until the preview is reviewed again.
