# Pronunciation Assistant FastAPI Backend

FastAPI backend scaffold for the AI-powered English pronunciation diagnosis system.

## Current scope

This scaffold includes:

- FastAPI app setup
- CORS for Expo frontend
- Health check endpoint
- Swagger docs
- Supabase JWT authentication foundation
- `GET /auth/me`

Not included yet:

- Audio upload
- Practice job flow
- AI inference
- Teacher analytics

## Setup

```powershell
cd fastapi-backend

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
copy .env.example .env
```

Fill in these Supabase values in `.env`:

```dotenv
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_ANON_KEY="your-supabase-anon-key"
SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-key"
SUPABASE_JWT_SECRET="your-supabase-jwt-secret"
```

Start the API:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger docs are available at:

```text
http://localhost:8000/docs
```

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

The endpoint verifies the Supabase access token and loads `app_role` from the `profiles` table using the service role key.
