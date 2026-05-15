@echo off
setlocal

rem Use UTF-8 output and keep all paths anchored at the repository root.
chcp 65001 >nul
title Pronunciation Assistant - Local Demo Launcher

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%fastapi-backend"
set "FRONTEND_DIR=%ROOT%frontend"
set "API_BASE_URL=http://localhost:8000"

rem Child window entry points. Keeping setup here avoids fragile nested quoting.
if /I "%~1"=="backend" goto backend
if /I "%~1"=="frontend" goto frontend

echo ------------------------------------------------------------
echo Pronunciation Assistant local demo
echo ------------------------------------------------------------

rem Check required backend environment file without printing secrets.
if not exist "%BACKEND_DIR%\.env" (
    echo Missing fastapi-backend/.env. Copy .env.example to .env and fill Supabase secrets.
    pause
    exit /b 1
)

rem Check required frontend environment file without printing secrets.
if not exist "%FRONTEND_DIR%\.env" (
    echo Missing frontend/.env. Copy .env.example to .env and fill EXPO_PUBLIC variables.
    pause
    exit /b 1
)

rem Port 8000 is fixed for the demo; do not silently switch ports.
netstat -ano -p tcp | findstr /R /C:":8000 .*LISTENING" >nul
if not errorlevel 1 (
    echo WARNING: Port 8000 is already in use.
    echo Close the old backend process or restart your machine, then run this script again.
    pause
    exit /b 1
)

rem Start FastAPI and Expo Web in separate terminal windows.
start "FastAPI Backend" cmd /k call "%~f0" backend
start "Expo Frontend" cmd /k call "%~f0" frontend

echo Started local demo windows:
echo - FastAPI Backend
echo - Expo Frontend
echo.
echo Keep both windows open while using the demo.
pause
exit /b 0

:backend
title FastAPI Backend
echo ------------------------------------------------------------
echo FastAPI backend
echo ------------------------------------------------------------

rem Change to the backend project before creating the venv or running uvicorn.
cd /d "%BACKEND_DIR%" || exit /b 1

set "VENV_CREATED=0"
if not exist ".venv\" (
    echo Creating backend virtual environment...
    python -m venv .venv || exit /b 1
    set "VENV_CREATED=1"
)

rem Activate the backend venv for this terminal session.
call ".venv\Scripts\activate.bat" || exit /b 1

rem Install backend dependencies only when the venv was newly created.
if "%VENV_CREATED%"=="1" (
    echo Installing backend dependencies...
    python -m pip install -r requirements.txt || exit /b 1
)

echo Starting FastAPI backend at %API_BASE_URL% ...
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
exit /b %errorlevel%

:frontend
title Expo Frontend
echo ------------------------------------------------------------
echo Expo frontend
echo ------------------------------------------------------------

rem Change to the frontend project before installing dependencies or starting Expo.
cd /d "%FRONTEND_DIR%" || exit /b 1

rem Install frontend dependencies when node_modules is missing.
if not exist "node_modules\" (
    echo Installing frontend dependencies...
    npm install || exit /b 1
)

rem Set the API base URL only for this terminal session.
set "EXPO_PUBLIC_API_BASE_URL=%API_BASE_URL%"

echo Starting Expo Web with EXPO_PUBLIC_API_BASE_URL=%API_BASE_URL% ...
npm run web -- --clear
exit /b %errorlevel%
