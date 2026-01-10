from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from core.phoneme_engine import analyze_pronunciation
from core.database import save_attempt_to_cloud
from core.prosody_engine import analyze_prosody # Import engine mới
import os
import nltk
import torch

# Tối ưu cấu hình cho RTX 3050
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("\n--- 🔍 ĐANG KIỂM TRA CẤU HÌNH HỆ THỐNG ---")
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    os.makedirs("data/samples", exist_ok=True)
    if not os.path.exists(".env"):
        print("--- ⚠️ CẢNH BÁO: Thiếu file .env ---")
    print("--- 🚀 HỆ THỐNG ĐÃ SẴN SÀNG ---\n")

@app.get("/")
async def read_index():
    # Phục vụ file giao diện chính
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Cậu chưa để file index.html vào thư mục 'static' rồi."}

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...), target_word: str = Form(...)):
    file_path = os.path.join("data/samples", file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # 1. Gọi AI xử lý âm vị (WhisperX + RTX 3050)
    ai_result = analyze_pronunciation(file_path, target_word)
    
    # 2. Phân tích âm học (Parselmouth - Pitch & Intensity)
    acoustic_data = analyze_prosody(file_path)
    
    # 3. Lưu lên Supabase Cloud
    save_attempt_to_cloud(target_word, ai_result["overall_score"], ai_result["phoneme_details"])
    
    # 4. Trả về định dạng chuẩn tích hợp 5 bước AI Thinking
    return {
        "overall_score": ai_result["overall_score"],
        "phoneme_details": ai_result["phoneme_details"],
        "prosody_analysis": acoustic_data,
        "ai_thinking": {
            "step_1_vad": "✅ Đã nhận diện giọng nói",
            "step_2_stt": ai_result["transcribed_text"],
            "step_3_alignment": "✅ Đã khớp mốc thời gian âm vị",
            "step_4_prosody": f"✅ Pitch: {acoustic_data['pitch']['mean']}Hz | Intensity: {acoustic_data['intensity']['mean']}dB",
            "step_5_db": "✅ Đã đồng bộ lên Supabase Cloud"
        }
    }