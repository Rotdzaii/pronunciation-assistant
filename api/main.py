from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from core.phoneme_engine import analyze_pronunciation
from core.database import save_attempt_to_cloud
from core.prosody_engine import analyze_prosody
import os
import io
import nltk
import torch
import sys
# Tối ưu cấu hình cho RTX 3050
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("\n--- ĐANG KIỂM TRA CẤU HÌNH HỆ THỐNG ---")
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    os.makedirs("data/samples", exist_ok=True)
    if not os.path.exists(".env"):
        print("---  CẢNH BÁO: Thiếu file .env ---")
    print("---  HỆ THỐNG ĐÃ SẴN SÀNG ---\n")

@app.get("/")
async def read_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Cậu chưa để file index.html vào thư mục 'static' rồi."}

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    # Bỏ target_word vì AI sẽ tự nhận dạng từ giọng nói
    file_path = os.path.join("data/samples", file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # 1. AI tự nhận dạng và bóc tách âm vị (WhisperX + G2P + Vocabulary Boosting)
    ai_result = analyze_pronunciation(file_path)
    if "error" in ai_result: 
        return ai_result

    # 2. Phân tích âm học (Parselmouth + Librosa)
    acoustic_data = analyze_prosody(file_path)
    
    # 3. Lưu kết quả lên Cloud
    save_attempt_to_cloud(
        ai_result["transcribed_text"], 
        ai_result["overall_score"], 
        ai_result["phoneme_details"]
    )
    
    return {
        "overall_score": ai_result["overall_score"],
        "phoneme_details": ai_result["phoneme_details"],
        "ai_thinking": {
            "step_1_vad": "✅ Đã nhận diện giọng nói",
            "step_2_stt": ai_result["transcribed_text"],
            "step_3_alignment": "✅ Đã khớp mốc thời gian âm vị",
            "step_4_prosody": f"✅ Pitch: {acoustic_data['pitch']['mean']}Hz | Intensity: {acoustic_data['intensity']['mean']}dB",
            "step_5_db": "✅ Đã đồng bộ lên Supabase Cloud"
        }
    }