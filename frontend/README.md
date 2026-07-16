# Frontend

Expo Router web frontend for the pronunciation practice demo.

## Setup and start

Expo belongs to this directory only. Do not run `npm install`, `npm install expo`,
or `npx expo start` at the repository root. Do not use `npm audit fix --force`
as a normal setup step.

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant\frontend
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

- `npm install`: install dependencies in `frontend/` only.
- `npm run web`: start Expo web, usually at `http://localhost:8081`; run it in `frontend/` only.
- `npm run lint`: run TypeScript checks with `tsc --noEmit`.

## Local demo

Start the FastAPI backend first, then start Expo with a clear cache:

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant\frontend
npx expo start -c
```

When a phone cannot reach Metro over the local network, use the development/demo
tunnel instead. It requires `@expo/ngrok` in `frontend/` if Expo prompts for it.

**Windows CMD**

```cmd
cd /d C:\Users\Admin\Documents\KLTN\pronunciation-assistant\frontend
npx expo start --tunnel -c
```

Expo Go QR codes are only for development/demo. Expo Go must be compatible with
the project's Expo SDK (`~51.0.28`). Do not upgrade the SDK in this workflow.
