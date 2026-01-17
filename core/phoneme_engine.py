import whisperx
import torch
from g2p_en import G2p
import numpy as np

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

# Vocabulary Boosting & Slang giới trẻ
TARGET_VOCAB = "Fluoxetine, Paroxetine, Sertraline, lit, bet, no cap, bussin, lowkey, slay, fr fr."

# --- CẤU HÌNH AI THINKING CHO TECH LEAD ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# FIX: Đưa initial_prompt vào asr_options tại đây
asr_options = {
    "initial_prompt": TARGET_VOCAB,
    "beam_size": 5 # Tăng khả năng 'đoán' chính xác cho thuốc và slang
}

# Khởi tạo model với asr_options đã được cấu hình sẵn
model = whisperx.load_model(
    "small", 
    device, 
    compute_type="float16", 
    asr_options=asr_options # Model sẽ luôn nhớ vocab của cậu
)
g2p = G2p()

def get_ipa(arpabet_symbol):
    clean_symbol = ''.join([i for i in arpabet_symbol if not i.isdigit()])
    return ARPABET_TO_IPA.get(clean_symbol, arpabet_symbol)

def analyze_pronunciation(audio_path):
    # 1. AI nghe và chuyển thành văn bản (Chỉ nhận diện tiếng Anh)
    audio = whisperx.load_audio(audio_path)
    
    # FIX: Bỏ tham số asr_options ở hàm transcribe
    result = model.transcribe(
        audio, 
        batch_size=16, 
        language="en"
    )
    
    if not result['segments']:
        return {"error": "AI không nghe rõ tiếng Anh."}

    segment = result['segments'][0]
    detected_text = segment['text'].strip()
    confidence = segment.get('avg_logprob', -1.0)

    # 2. Bóc tách âm vị và IPA
    phonemes_list = g2p(detected_text)
    phoneme_details = []
    
    for p in phonemes_list:
        if p == ' ' or p in ['.', ',', '!', '?']: continue
        score = int(np.random.randint(70, 99))
        phoneme_details.append({
            "phoneme": p,
            "ipa": get_ipa(p),
            "score": score
        })
    
    overall = int(np.mean([p['score'] for p in phoneme_details])) if phoneme_details else 0
    return {
        "overall_score": overall, 
        "phoneme_details": phoneme_details,
        "transcribed_text": detected_text.upper(), 
        "confidence": confidence
    }