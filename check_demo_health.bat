@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>nul
title Demo Health Check

echo ------------------------------------------------------------
echo  Pronunciation Assistant  -  Demo Health Check
echo ------------------------------------------------------------
echo.

set "PASS=0"
set "FAIL=0"
set "WARN=0"

rem --- 1. Backend -------------------------------------------------------------
echo [Backend]
call :check_port_pid 8000
if not defined PORT_PID (
    echo   [FAIL] Port 8000 not listening - backend is NOT running.
    set /a FAIL+=1
) else (
    set "HEALTH_RESP="
    for /f "delims=" %%R in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "try{(Invoke-WebRequest -Uri http://localhost:8000/health -TimeoutSec 4 -UseBasicParsing).StatusCode}catch{'ERR'}" 2^>nul') do set "HEALTH_RESP=%%R"
    if "!HEALTH_RESP!"=="200" (
        echo   [OK]   Backend running on http://localhost:8000 ^(PID %PORT_PID% %PORT_PROC%^)
        echo          GET /health -^> 200 OK
        set /a PASS+=1
    ) else if "!HEALTH_RESP!"=="ERR" (
        echo   [WARN] Port 8000 has a listener ^(PID %PORT_PID% %PORT_PROC%^) but /health returned error.
        echo          Backend may still be starting. Wait a few seconds and rerun.
        set /a WARN+=1
    ) else (
        echo   [WARN] Port 8000 listener ^(PID %PORT_PID% %PORT_PROC%^) - /health returned: !HEALTH_RESP!
        set /a WARN+=1
    )
)

rem --- 2. Frontend ------------------------------------------------------------
echo.
echo [Frontend]
set "FE_PORT="
set "FE_PID="
set "FE_PROC="
for %%P in (8081 8082 8083) do (
    if not defined FE_PORT (
        call :check_port_pid %%P
        if defined PORT_PID (
            set "FE_PORT=%%P"
            set "FE_PID=%PORT_PID%"
            set "FE_PROC=%PORT_PROC%"
        )
    )
)
if defined FE_PORT (
    echo   [OK]   Frontend running on http://localhost:%FE_PORT% ^(PID %FE_PID% %FE_PROC%^)
    set /a PASS+=1
) else (
    echo   [FAIL] No listener on ports 8081/8082/8083 - frontend is NOT running.
    set /a FAIL+=1
)

rem --- 3. AI Worker -----------------------------------------------------------
echo.
echo [AI Worker]
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=Get-CimInstance Win32_Process|Where-Object{($_.Name -eq 'python.exe'-or $_.Name -eq 'pythonw.exe')-and $_.CommandLine -match 'worker\.py'};if($p){Write-Host ('   [OK]   AI Worker process found (PID '+($p|Select-Object -First 1).ProcessId+')')}else{exit 1}" 2>nul
if errorlevel 1 (
    echo   [FAIL] AI Worker process not found.
    set /a FAIL+=1
) else (
    set /a PASS+=1
)

rem --- 4. Cloudflare Tunnel ---------------------------------------------------
echo.
echo [Cloudflare Tunnel]
powershell -NoProfile -ExecutionPolicy Bypass -Command "if(Get-Process cloudflared -EA SilentlyContinue){exit 0}else{exit 1}" >nul 2>nul
if not errorlevel 1 (
    echo   [OK]   cloudflared process running.
    echo          app.myphoenix.me -^> http://localhost:8081
    echo          api.myphoenix.me -^> http://localhost:8000
    set /a PASS+=1
) else (
    echo   [WARN] Cloudflare tunnel not running.
    echo          Run run_deploy.bat to start it.
    set /a WARN+=1
)

rem --- Summary ----------------------------------------------------------------
echo.
echo ------------------------------------------------------------
echo  Results:  %PASS% OK   %WARN% WARN   %FAIL% FAIL
echo ------------------------------------------------------------
if %FAIL% GTR 0 (
    echo.
    echo  Some services are NOT running.
    echo  To restart all:        run_server.bat
    echo  To free stuck ports:   reset_demo_ports.bat
)
echo.
pause
exit /b %FAIL%

rem :check_port_pid <port>
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
