import parselmouth
import numpy as np

def analyze_prosody(audio_path):
    """
    Trích xuất 3 yếu tố quan trọng của trọng âm: Pitch, Intensity và Duration
    Dựa trên báo cáo nghiên cứu CAPT.
    """
    # 1. Nạp file âm thanh vào Parselmouth
    snd = parselmouth.Sound(audio_path)
    
    # 2. Trích xuất Cao độ (Pitch/F0)
    # Sử dụng phương pháp Autocorrelation chuẩn của Praat
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    
    # Loại bỏ các vùng vô thanh (pitch = 0) để tính toán chính xác
    valid_pitch = pitch_values[pitch_values > 0]
    max_f0 = np.max(valid_pitch) if len(valid_pitch) > 0 else 0
    mean_f0 = np.mean(valid_pitch) if len(valid_pitch) > 0 else 0

    # 3. Trích xuất Cường độ (Intensity)
    intensity = snd.to_intensity()
    max_intensity = intensity.get_maximum()
    mean_intensity = np.mean(intensity.values)

    # 4. Tính toán thời lượng (Duration)
    duration = snd.get_total_duration()

    # Trả về nhật ký kỹ thuật cho Tech Lead
    return {
        "pitch": {
            "max": round(max_f0, 2), # Đơn vị Hz
            "mean": round(mean_f0, 2)
        },
        "intensity": {
            "max": round(max_intensity, 2), # Đơn vị dB
            "mean": round(mean_intensity, 2)
        },
        "duration": round(duration, 2) # Đơn vị giây
    }

# --- Demo chạy thử ---
# if __name__ == "__main__":
#    results = analyze_prosody("data/samples/recording.wav")
#    print(f"--- 📊 Prosody Log: {results} ---")