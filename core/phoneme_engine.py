import whisperx
import torch
from g2p_en import G2p
import os

# --- CẤU HÌNH HỆ THỐNG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Khởi tạo g2p global
g2p = G2p()

_model_cache = {
    "asr": None,
    "align": None,
    "metadata": None
}

def get_models():
    global _model_cache
    if _model_cache["asr"] is None:
        print(f"--- 🚀 Đang tải WhisperX (Base) lên {DEVICE.upper()}... ---")
        # Ép ngôn ngữ tiếng Anh ngay từ lúc load model
        _model_cache["asr"] = whisperx.load_model("base", DEVICE, compute_type=COMPUTE_TYPE, language="en")
        
        print(f"--- 🎯 Đang tải Alignment Model (English)... ---")
        model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
        _model_cache["align"] = model_a
        _model_cache["metadata"] = metadata
        
    return _model_cache["asr"], _model_cache["align"], _model_cache["metadata"]

def analyze_pronunciation(audio_path, target_word):
    model, model_a, metadata = get_models()
    
    # 1. Chuyển từ mục tiêu sang âm vị
    target_phonemes = g2p(target_word)
    
    # 2. WhisperX xử lý âm thanh (Ép ngôn ngữ English)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=1, language="en")
    
    # 3. Lấy text mà AI thực sự nghe được để hiển thị Dev Mode
    heard_text = " ".join([seg["text"] for seg in result["segments"]]).strip()
    print(f"--- 🎙️ AI Heard: '{heard_text}' ---")

    # 4. Căn chỉnh thời gian (Alignment)
    result_aligned = whisperx.align(
        result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=True
    )

    # 5. Logic chấm điểm nâng cấp (Tránh lỗi luôn bị 0%)
    analysis = []
    confidence = 0
    # Kiểm tra xem AI có nghe thấy từ nào không
    is_detected = len(result_aligned['segments']) > 0
    
    if is_detected:
        confidence = result_aligned['segments'][0].get('confidence', 0)
        # So sánh không phân biệt hoa thường để tránh lỗi 0%
        is_match = target_word.lower() in heard_text.lower()
    else:
        is_match = False

    for p in target_phonemes:
        p_clean = p.strip()
        if p_clean:
            # Nếu AI nghe đúng từ, điểm sẽ dựa trên độ tự tin (min 50%), ngược lại là 0%
            score = round(max(confidence, 0.5) * 100, 2) if is_match else 0
            status = "Correct" if score >= 70 else "Incorrect"
            
            analysis.append({
                "phoneme": p_clean,
                "status": status,
                "score": score
            })
            
    # Tính điểm trung bình
    avg_score = sum([p["score"] for p in analysis]) / len(analysis) if analysis else 0

    return {
        "overall_score": round(avg_score, 2),
        "phoneme_details": analysis,
        "transcribed_text": heard_text if heard_text else "(No speech detected)"
    }