@echo off
setlocal EnableExtensions EnableDelayedExpansion

chcp 65001 >nul 2>nul
title Pronunciation Assistant - Local Demo Launcher

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%fastapi-backend"
set "FRONTEND_DIR=%ROOT%frontend"
set "AI_WORKER_DIR=%ROOT%ai-worker"
if not defined API_BASE_URL set "API_BASE_URL=http://localhost:8000"

rem Dispatch to child window entry points and special flags.
if /I "%~1"=="backend"       goto backend
if /I "%~1"=="frontend"      goto frontend
if /I "%~1"=="ai-worker"     goto ai_worker
if /I "%~1"=="--force-kill"  goto force_kill

goto main

rem ---------------------------------------------------------------------------
rem MAIN - opens three child cmd windows (backend / frontend / ai-worker)
rem ---------------------------------------------------------------------------
:main
echo ------------------------------------------------------------
echo  Pronunciation Assistant  -  Local Demo
echo ------------------------------------------------------------
echo  ROOT: %ROOT%
echo.

rem Pre-flight: check required tools.
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python not found in PATH. Install Python 3.10+ and retry.
    pause & exit /b 1
)
where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] node not found in PATH. Install Node.js 18+ and retry.
    pause & exit /b 1
)
where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm not found in PATH. Install Node.js 18+ and retry.
    pause & exit /b 1
)

rem Pre-flight: check .env files.
if not exist "%BACKEND_DIR%\.env" (
    echo [ERROR] Missing fastapi-backend/.env
    echo         Copy fastapi-backend/.env.example to fastapi-backend/.env and fill in Supabase secrets.
    pause & exit /b 1
)
if not exist "%FRONTEND_DIR%\.env" (
    echo [ERROR] Missing frontend/.env
    echo         Copy frontend/.env.example to frontend/.env and set EXPO_PUBLIC_API_BASE_URL.
    pause & exit /b 1
)

set "SKIP_WORKER="
if not exist "%AI_WORKER_DIR%\.env" (
    echo [WARN] Missing ai-worker/.env - AI Worker will be skipped.
    echo        Copy ai-worker/.env.example to ai-worker/.env to enable it.
    set "SKIP_WORKER=1"
)

rem Backend: check if port 8000 already occupied.
set "BACKEND_STATUS=skipped"
call :check_port_pid 8000
if defined PORT_PID (
    echo [INFO] Port 8000 in use by PID !PORT_PID! ^(!PORT_PROC!^). Skipping backend start.
    echo        Run this script with --force-kill to free the port.
    set "BACKEND_STATUS=already running - port 8000 PID !PORT_PID!"
) else (
    start "FastAPI Backend" cmd /k call "%~f0" backend
    set "BACKEND_STATUS=started"
)

rem Frontend: check if any Expo port (8081-8083) is occupied.
set "FRONTEND_STATUS=skipped"
set "FE_PORT_USED="
for %%P in (8081 8082 8083) do (
    call :check_port_pid %%P
    if not defined FE_PORT_USED (
        if defined PORT_PID set "FE_PORT_USED=%%P (PID !PORT_PID! !PORT_PROC!)"
    )
)
if defined FE_PORT_USED (
    echo [INFO] Frontend port already in use: !FE_PORT_USED!
    echo        Run this script with --force-kill to free the port.
    set "FRONTEND_STATUS=already running - !FE_PORT_USED!"
) else (
    start "Expo Frontend" cmd /k call "%~f0" frontend
    set "FRONTEND_STATUS=started"
)

rem AI Worker: check if worker.py process exists.
set "AI_WORKER_STATUS=skipped"
if not defined SKIP_WORKER (
    call :is_ai_worker_running
    if "!AI_WORKER_RUNNING!"=="1" (
        echo [INFO] AI Worker already running. Skipping worker start.
        set "AI_WORKER_STATUS=already running"
    ) else (
        start "AI Worker" cmd /k call "%~f0" ai-worker
        set "AI_WORKER_STATUS=started"
    )
)

echo.
echo ------------------------------------------------------------
echo  Summary
echo ------------------------------------------------------------
echo  Backend   : !BACKEND_STATUS!
echo  Frontend  : !FRONTEND_STATUS!
echo  AI Worker : !AI_WORKER_STATUS!
echo.
echo  Keep all service windows open during development.
echo  Run this script with --force-kill to kill stuck processes on demo ports.
echo  Need public access for a real demo? Run run_deploy.bat instead.
echo ------------------------------------------------------------
pause
exit /b 0

rem ---------------------------------------------------------------------------
rem --force-kill: kill processes on demo ports
rem ---------------------------------------------------------------------------
:force_kill
echo Killing processes on demo ports 8000 8081 8082 8083...
echo.
for %%P in (8000 8081 8082 8083) do (
    call :check_port_pid %%P
    if defined PORT_PID (
        echo [KILL] Port %%P  PID !PORT_PID!  !PORT_PROC!
        taskkill /F /PID !PORT_PID! >nul 2>nul
        if not errorlevel 1 (
            echo       Killed.
        ) else (
            echo       Failed to kill - may need elevated permissions.
        )
    ) else (
        echo [FREE] Port %%P not in use.
    )
)
echo.
echo Done. Run run_server.bat to restart services.
pause
exit /b 0

rem ---------------------------------------------------------------------------
rem BACKEND child window
rem ---------------------------------------------------------------------------
:backend
title FastAPI Backend
echo ------------------------------------------------------------
echo  FastAPI Backend  -  http://localhost:8000
echo ------------------------------------------------------------

cd /d "%BACKEND_DIR%"
if errorlevel 1 (
    echo [ERROR] Cannot cd to %BACKEND_DIR%
    pause & exit /b 1
)

if not exist ".venv\" (
    echo Creating backend virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] venv creation failed. Check Python installation.
        pause & exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Cannot activate .venv - try deleting fastapi-backend/.venv and rerunning.
    pause & exit /b 1
)

echo Installing/verifying backend dependencies...
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check requirements.txt and internet connection.
    pause & exit /b 1
)

echo.
echo [OK] Dependencies verified.
echo      Docs: http://localhost:8000/docs
echo.
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo [EXIT] FastAPI backend exited with code %errorlevel%.
pause
exit /b %errorlevel%

rem ---------------------------------------------------------------------------
rem FRONTEND child window
rem ---------------------------------------------------------------------------
:frontend
title Expo Frontend
echo ------------------------------------------------------------
echo  Expo Frontend  -  http://localhost:8081
echo ------------------------------------------------------------

cd /d "%FRONTEND_DIR%"
if errorlevel 1 (
    echo [ERROR] Cannot cd to %FRONTEND_DIR%
    pause & exit /b 1
)

if not exist "node_modules\" (
    echo node_modules not found. Running npm install...
    npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        pause & exit /b 1
    )
) else (
    if not exist "node_modules\.bin\expo" (
        echo [WARN] node_modules exists but expo binary missing. Running npm install...
        npm install
        if errorlevel 1 (
            echo [ERROR] npm install failed.
            pause & exit /b 1
        )
    )
)

set "EXPO_PUBLIC_API_BASE_URL=%API_BASE_URL%"

echo.
echo [OK] Starting Expo Web...
echo      EXPO_PUBLIC_API_BASE_URL=%EXPO_PUBLIC_API_BASE_URL%
echo      URL: http://localhost:8081 (fallback 8082/8083)
echo.
npm run web
echo.
echo [EXIT] Expo frontend exited with code %errorlevel%.
pause
exit /b %errorlevel%

rem ---------------------------------------------------------------------------
rem AI WORKER child window
rem ---------------------------------------------------------------------------
:ai_worker
title AI Worker
echo ------------------------------------------------------------
echo  AI Worker
echo ------------------------------------------------------------

cd /d "%AI_WORKER_DIR%"
if errorlevel 1 (
    echo [ERROR] Cannot cd to %AI_WORKER_DIR%
    pause & exit /b 1
)

if not exist ".env" (
    echo [ERROR] Missing ai-worker/.env
    echo         Copy .env.example to .env and fill in worker settings.
    pause & exit /b 1
)

if not exist ".venv\" (
    echo Creating AI worker virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] venv creation failed. Check Python installation.
        pause & exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Cannot activate .venv - try deleting ai-worker/.venv and rerunning.
    pause & exit /b 1
)

echo Installing/verifying AI worker dependencies...
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check ai-worker/requirements.txt.
    pause & exit /b 1
)

findstr /I /B /C:"SCORER_MODE=cnn_attention" ".env" >nul 2>nul
if not errorlevel 1 (
    python -c "import torch" >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] The configured CNN worker scorer requires PyTorch, but it is not available in ai-worker/.venv.
        echo         ai-worker/requirements.txt does not declare this inference dependency. Provision the approved local inference environment, then retry.
        pause & exit /b 1
    )
)

echo.
echo [OK] Dependencies verified. Starting AI worker...
echo.
python worker.py
echo.
echo [EXIT] AI worker exited with code %errorlevel%.
pause
exit /b %errorlevel%

rem ---------------------------------------------------------------------------
rem SUBROUTINES
rem ---------------------------------------------------------------------------

rem :check_port_pid <port>
rem  Sets PORT_PID (process ID) and PORT_PROC (process name), both empty if free.
:check_port_pid
set "PORT_PID="
set "PORT_PROC="
for /f "tokens=5" %%A in ('netstat -ano -p tcp 2^>nul ^| findstr /R /C:":%~1 .*LISTENING"') do (
    if not defined PORT_PID set "PORT_PID=%%A"
)
if defined PORT_PID (
    for /f "skip=1 tokens=2 delims=," %%A in ('wmic process where "ProcessId=%PORT_PID%" get Name /format:csv 2^>nul') do (
        if not defined PORT_PROC set "PORT_PROC=%%A"
    )
)
exit /b 0

rem :is_ai_worker_running
rem  Sets AI_WORKER_RUNNING=1 when a python process running worker.py is found.
:is_ai_worker_running
set "AI_WORKER_RUNNING=0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=Get-CimInstance Win32_Process|Where-Object{($_.Name -eq 'python.exe'-or $_.Name -eq 'pythonw.exe')-and $_.CommandLine -match 'worker\.py'};if($p){exit 0}else{exit 1}" >nul 2>nul
if not errorlevel 1 set "AI_WORKER_RUNNING=1"
exit /b 0
