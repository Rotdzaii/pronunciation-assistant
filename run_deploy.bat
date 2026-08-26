@echo off
setlocal EnableExtensions EnableDelayedExpansion

chcp 65001 >nul 2>nul
title Pronunciation Assistant - Public Deploy Launcher

set "ROOT=%~dp0"
set "DEMO_SCRIPT=%ROOT%run_server.bat"
set "API_BASE_URL=https://api.myphoenix.me"

echo ------------------------------------------------------------
echo  Pronunciation Assistant  -  Public Deploy (Cloudflare Tunnel)
echo ------------------------------------------------------------
echo.

if not exist "%DEMO_SCRIPT%" (
    echo [ERROR] run_server.bat not found next to this script at:
    echo         %DEMO_SCRIPT%
    echo         This script depends on run_server.bat to start local services.
    pause & exit /b 1
)

where cloudflared >nul 2>nul
if errorlevel 1 (
    echo [ERROR] cloudflared not found in PATH.
    echo         Install from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/
    pause & exit /b 1
)

set "CF_CONFIG=%USERPROFILE%\.cloudflared\config.yml"
if not exist "%CF_CONFIG%" (
    echo [ERROR] Cloudflare config not found at %CF_CONFIG%
    echo         Run: cloudflared tunnel login
    echo         Run: cloudflared tunnel create phoenix-demo
    echo         Then configure %CF_CONFIG%
    pause & exit /b 1
)

echo Step 1/2 - Starting local services (backend / frontend / AI worker)...
echo           A summary window will open and pause - close it or press any
echo           key there once the 3 service windows are confirmed running,
echo           then this window will continue automatically.
echo.
call "%DEMO_SCRIPT%"

echo.
echo Step 2/2 - Starting Cloudflare Tunnel (phoenix-demo)...
echo   app.myphoenix.me  -^>  http://localhost:8081 (frontend)
echo   api.myphoenix.me  -^>  http://localhost:8000 (backend)
start "Cloudflare Tunnel" cmd /k cloudflared tunnel --config "%CF_CONFIG%" run phoenix-demo
echo [OK] Tunnel window opened.
echo.
echo ------------------------------------------------------------
echo  Public URLs
echo ------------------------------------------------------------
echo   Frontend : https://app.myphoenix.me
echo   Backend  : https://api.myphoenix.me
echo.
echo   NOTE: this launcher sets EXPO_PUBLIC_API_BASE_URL=https://api.myphoenix.me
echo         for its Expo process so publicly-served frontend clients reach
echo         the public API. Do not put backend secrets in frontend/.env.
echo.
echo   Keep the Backend / Frontend / AI Worker / Tunnel windows open
echo   for the duration of the demo.
echo ------------------------------------------------------------
pause
exit /b 0
