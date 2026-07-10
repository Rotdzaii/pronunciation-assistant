@echo off
rem Starts all services AND the Cloudflare tunnel (phoenix-demo).
rem Domain mapping (from ~/.cloudflared/config.yml):
rem   app.myphoenix.me  ->  http://localhost:8081  (Expo frontend)
rem   api.myphoenix.me  ->  http://localhost:8000  (FastAPI backend)
chcp 65001 >nul
call "%~dp0run_server.bat" --with-tunnel
