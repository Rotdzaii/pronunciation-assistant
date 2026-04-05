@echo off
chcp 65001
title Voice AI Project - Backend Server
echo ----------------------------------------------------------
echo [STEP 1] Kiem tra moi truong ao .venv...
if not exist .venv (
    echo LOI: Khong tim thay thu muc .venv. Hay chay setup_project.bat truoc!
    pause
    exit
)

echo [STEP 2] Kiem tra card do hoa RTX 3050...
.\.venv\Scripts\python.exe -c "import torch; print('--- CUDA Status:', torch.cuda.is_available())"

echo [STEP 3] Dang khoi dong server FastAPI tai http://127.0.0.1:8000
echo  Cho 5-10 giay de AI Model load vao GPU ...
echo ----------------------------------------------------------

.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
pause