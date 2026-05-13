# Pronunciation Assistant FastAPI Backend

FastAPI backend for the Expo pronunciation practice demo. It verifies Supabase access tokens, loads roles from `public.profiles`, uploads practice audio to Supabase Storage, creates practice jobs, and receives simulated AI results through a shared-secret webhook.

## Setup

```powershell
cd fastapi-backend

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
copy .env.example .env
```

Fill in these Supabase values in `.env`:

```dotenv
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_ANON_KEY="your-supabase-anon-key"
SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-key"
AI_WEBHOOK_SECRET="replace-with-ai-webhook-secret"
PRACTICE_AUDIO_BUCKET="practice-audios"
```

Start the API:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Current API endpoints

- `GET /health`
- `GET /auth/me`
- `POST /practice/upload-audio`
- `POST /practice/create-job`
- `GET /practice/{job_id}`
- `POST /practice/webhook/ai-result`
- `GET /practice/history`

## Auth

`GET /auth/me` expects a Supabase access token:

```powershell
curl.exe -H "Authorization: Bearer <supabase-access-token>" http://localhost:8000/auth/me
```

Successful response:

```json
{
  "id": "user-id",
  "email": "user@example.com",
  "auth_role": "authenticated",
  "app_role": "student"
}
```

The endpoint verifies the Supabase access token with Supabase Auth's `/auth/v1/user` endpoint and loads `app_role` from the `profiles` table using the service role key.

## Audio upload

`POST /practice/upload-audio` expects a Supabase access token for a user whose profile has `app_role = "student"`.

The Supabase Storage bucket must exist:

```text
practice-audios
```

Upload a local audio file:

```powershell
curl.exe -X POST `
  -H "Authorization: Bearer <supabase-access-token>" `
  -F "file=@C:\path\to\audio.wav;type=audio/wav" `
  http://localhost:8000/practice/upload-audio
```

Allowed MIME types:

- `audio/wav`
- `audio/mpeg`
- `audio/mp4`
- `audio/x-m4a`

Successful response:

```json
{
  "message": "uploaded",
  "storage_path": "student-id/uuid-audio.wav",
  "audio_url": "https://...",
  "mime_type": "audio/wav",
  "size": 12345
}
```

## Practice jobs

`POST /practice/create-job` expects a Supabase access token for a user whose profile has `app_role = "student"`.

Create and enqueue a practice job after uploading audio:

```powershell
curl.exe -X POST `
  -H "Authorization: Bearer <student-supabase-access-token>" `
  -H "Content-Type: application/json" `
  -d "{\"target_word\":\"Architecture\",\"audio_url\":\"https://...\"}" `
  http://localhost:8000/practice/create-job
```

Successful response:

```json
{
  "job_id": "practice-job-id",
  "status": "processing",
  "message": "Practice job created and queued"
}
```

The API inserts into `public.practice_history` with `problem_phonemes = []` and `feedback = {}`, then calls `public.enqueue_practice_job(...)`.

Fetch a practice job:

```powershell
curl.exe -H "Authorization: Bearer <supabase-access-token>" `
  http://localhost:8000/practice/<practice-job-id>
```

Students can fetch only their own jobs. Teachers can fetch any job when their profile has `app_role = "teacher"`.

List practice history as a student:

```powershell
curl.exe -H "Authorization: Bearer <student-supabase-access-token>" `
  "http://localhost:8000/practice/history?limit=20&offset=0"
```

List completed practice history as a teacher for one student:

```powershell
curl.exe -H "Authorization: Bearer <teacher-supabase-access-token>" `
  "http://localhost:8000/practice/history?student_id=<student-id>&status=completed&limit=20&offset=0"
```

Successful history response:

```json
{
  "items": [
    {
      "id": "practice-job-id",
      "student_id": "student-id",
      "target_word": "Architecture",
      "audio_url": "https://...",
      "status": "completed",
      "score": 86.5,
      "problem_phonemes": [],
      "feedback": {},
      "created_at": "2026-05-12T00:00:00Z",
      "updated_at": "2026-05-12T00:01:00Z"
    }
  ],
  "limit": 20,
  "offset": 0
}
```

## AI result webhook

`POST /practice/webhook/ai-result` is for the AI worker. It does not use a user JWT. It requires the shared secret header `x-ai-webhook-secret`.

Mark a job completed:

```powershell
curl.exe -X POST `
  -H "x-ai-webhook-secret: <ai-webhook-secret>" `
  -H "Content-Type: application/json" `
  -d "{\"job_id\":\"<practice-job-id>\",\"status\":\"completed\",\"score\":82.5,\"problem_phonemes\":[],\"feedback\":{}}" `
  http://localhost:8000/practice/webhook/ai-result
```

Mark a job failed:

```powershell
curl.exe -X POST `
  -H "x-ai-webhook-secret: <ai-webhook-secret>" `
  -H "Content-Type: application/json" `
  -d "{\"job_id\":\"<practice-job-id>\",\"status\":\"failed\",\"problem_phonemes\":[],\"feedback\":{}}" `
  http://localhost:8000/practice/webhook/ai-result
```

Successful response:

```json
{
  "job_id": "practice-job-id",
  "status": "completed",
  "message": "Practice job result updated"
}
```
