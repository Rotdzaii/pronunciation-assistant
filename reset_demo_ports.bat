@echo off
rem Kills processes occupying the demo ports (8000 8081 8082 8083).
rem Run this when services fail to start because a port is already in use.
chcp 65001 >nul
call "%~dp0run_server.bat" --force-kill
