# Frontend

Expo Router web frontend for the pronunciation practice demo.

## Setup

```powershell
cd frontend
npm install
copy .env.example .env
```

`frontend/.env` must contain only frontend-safe variables:

```dotenv
EXPO_PUBLIC_API_BASE_URL=http://localhost:8000
EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
```

Do not put backend secrets in frontend env files.

## Scripts

- `npm install`: install dependencies.
- `npm run web`: start Expo web, usually at `http://localhost:8081`.
- `npm run lint`: run TypeScript checks with `tsc --noEmit`.

## Local demo

Start the FastAPI backend first, then run:

```powershell
npm run web
```

Login with a Supabase student user, record audio, replay it, submit it for scoring, and open History after simulating completion through the backend Swagger webhook.
