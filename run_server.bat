@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Use UTF-8 output and keep all paths anchored at the repository root.
chcp 65001 >nul
title Pronunciation Assistant - Local Demo Launcher

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%fastapi-backend"
set "FRONTEND_DIR=%ROOT%frontend"
set "AI_WORKER_DIR=%ROOT%ai-worker"
set "API_BASE_URL=http://localhost:8000"

rem Child window entry points. Keeping setup here avoids fragile nested quoting.
if /I "%~1"=="backend" goto backend
if /I "%~1"=="frontend" goto frontend
if /I "%~1"=="ai-worker" goto ai_worker

echo ------------------------------------------------------------
echo Pronunciation Assistant local demo
echo ------------------------------------------------------------

set "BACKEND_STATUS=skipped"
set "FRONTEND_STATUS=skipped"
set "AI_WORKER_STATUS=skipped"

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

call :is_port_listening 8000
if "%PORT_LISTENING%"=="1" (
    echo FastAPI backend already appears to be running on port 8000. Skipping backend start.
    set "BACKEND_STATUS=running"
) else (
    start "FastAPI Backend" cmd /k call "%~f0" backend
    set "BACKEND_STATUS=started"
)

set "FRONTEND_RUNNING=0"
for %%P in (8081 8082 8083) do (
    call :is_port_listening %%P
    if "!PORT_LISTENING!"=="1" set "FRONTEND_RUNNING=1"
)

if "%FRONTEND_RUNNING%"=="1" (
    echo Expo frontend already appears to be running. Skipping frontend start.
    set "FRONTEND_STATUS=running"
) else (
    start "Expo Frontend" cmd /k call "%~f0" frontend
    set "FRONTEND_STATUS=started"
)

if not exist "%AI_WORKER_DIR%\.env" (
    echo WARNING: Missing ai-worker/.env. Copy .env.example to .env and fill worker settings.
    echo WARNING: AI Worker window will not be started.
    set "AI_WORKER_STATUS=skipped because missing env"
    goto after_worker_check
)

call :is_ai_worker_running
if "%AI_WORKER_RUNNING%"=="1" (
    echo AI worker already appears to be running. Skipping worker start.
    set "AI_WORKER_STATUS=running"
    goto after_worker_check
)

start "AI Worker" cmd /k call "%~f0" ai-worker
set "AI_WORKER_STATUS=started"

:after_worker_check
echo.
echo Local demo summary:
echo - Backend: %BACKEND_STATUS%
echo - Frontend: %FRONTEND_STATUS%
echo - AI Worker: %AI_WORKER_STATUS%
echo.
echo Keep any demo windows open while using the demo.
pause
exit /b 0

:is_port_listening
set "PORT_LISTENING=0"
netstat -ano -p tcp | findstr /R /C:":%~1 .*LISTENING" >nul
if not errorlevel 1 set "PORT_LISTENING=1"
exit /b 0

:is_ai_worker_running
set "AI_WORKER_RUNNING=0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$items = Get-CimInstance Win32_Process; foreach ($item in $items) { if (($item.Name -eq 'python.exe' -or $item.Name -eq 'pythonw.exe') -and $item.CommandLine -and ($item.CommandLine -match 'ai-worker[\\/]+worker\.py' -or $item.CommandLine -match '(^|\s)worker\.py($|\s)')) { exit 0 } }; exit 1" >nul 2>nul
if not errorlevel 1 set "AI_WORKER_RUNNING=1"
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
npm run web
exit /b %errorlevel%

:ai_worker
title AI Worker
echo ------------------------------------------------------------
echo AI worker
echo ------------------------------------------------------------

rem Change to the worker project before creating the venv or running the worker.
cd /d "%AI_WORKER_DIR%" || exit /b 1

if not exist ".env" (
    echo Missing ai-worker/.env. Copy .env.example to .env and fill worker settings.
    echo Worker was not started.
    pause
    exit /b 1
)

set "VENV_CREATED=0"
if not exist ".venv\" (
    echo Creating AI worker virtual environment...
    python -m venv .venv || exit /b 1
    set "VENV_CREATED=1"
)

rem Activate the AI worker venv for this terminal session.
call ".venv\Scripts\activate.bat" || exit /b 1

rem Install worker dependencies only when the venv was newly created.
if "%VENV_CREATED%"=="1" (
    echo Installing AI worker dependencies...
    python -m pip install -r requirements.txt || exit /b 1
)

echo Starting AI worker...
if not defined WORKER_MODE (
    for /f "tokens=1,* delims==" %%A in ('findstr /R /B "WORKER_MODE=" ".env" 2^>nul') do set "WORKER_MODE=%%B"
)
if not defined WORKER_MODE set "WORKER_MODE=loop"
python worker.py
exit /b %errorlevel%
