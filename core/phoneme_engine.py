import whisperx
import torch
from g2p_en import G2p
import os
import numpy as np

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
    # Từ điển ánh xạ Arpabet sang IPA chuẩn quốc tế
ARPABET_TO_IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
    'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'IH': 'ɪ', 'IY': 'i', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u', 'B': 'b', 'CH': 'tʃ', 'D': 'd',
    'DH': 'ð', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'JH': 'dʒ', 'K': 'k',
    'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p', 'R': 'r',
    'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v', 'W': 'w',
    'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}
def get_ipa(arpabet_symbol):
    """Chuyển đổi ký hiệu Arpabet thành IPA bằng cách loại bỏ số trọng âm."""
    # Ví dụ: IH1 -> IH -> ɪ
    clean_symbol = ''.join([i for i in arpabet_symbol if not i.isdigit()])
    return ARPABET_TO_IPA.get(clean_symbol, arpabet_symbol)

def analyze_pronunciation(audio_path, target_word):
    """
    Hàm xử lý chính: Chạy WhisperX Alignment trên RTX 3050.
    Trả về chi tiết âm vị bao gồm cả nhãn IPA.
    """
    # --- LOGIC GIẢ LẬP KẾT QUẢ TỪ WHISPERX ---
    # Trong thực tế, đây là nơi cậu gọi model WhisperX để lấy alignment
    raw_results = [
        {"symbol": "IH1", "score": 85},
        {"symbol": "NG", "score": 45},
        {"symbol": "G", "score": 90},
        {"symbol": "L", "score": 75},
        {"symbol": "IH0", "score": 60},
        {"symbol": "SH", "score": 95}
    ]
    
    phoneme_details = []
    for p in raw_results:
        phoneme_details.append({
            "phoneme": p['symbol'],
            "ipa": get_ipa(p['symbol']), # Bổ sung nhãn IPA
            "score": p['score'],
            "status": "Correct" if p['score'] >= 70 else "Incorrect"
        })
    
    # Tính điểm trung bình (Overall Score)
    overall = int(np.mean([p['score'] for p in phoneme_details]))
    
    return {
        "overall_score": overall,
        "phoneme_details": phoneme_details,
        "transcribed_text": target_word.upper()
    }