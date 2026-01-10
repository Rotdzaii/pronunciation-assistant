import parselmouth
import librosa
import numpy as np

def analyze_prosody(audio_path):
    """
    Trích xuất 3 yếu tố quan trọng của trọng âm: Pitch, Intensity và Duration.
    Sửa lỗi 'Not an audio file' bằng cách dùng Librosa để giải mã trước.
    """
    try:
        # 1. Sử dụng Librosa để nạp và giải mã file (hỗ trợ WebM, Opus, WAV...)
        # sr=None để giữ nguyên tần số lấy mẫu gốc của file
        y, sr = librosa.load(audio_path, sr=None)
        
        # 2. Chuyển đổi mảng numpy thành đối tượng Sound của Parselmouth
        # Praat yêu cầu dữ liệu âm thanh và tần số lấy mẫu (Sampling Rate)
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        
        # 3. Trích xuất Cao độ (Pitch/F0)
        # Sử dụng phương pháp Autocorrelation tiêu chuẩn của Praat
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        
        # Loại bỏ các vùng vô thanh (pitch = 0) để tính toán chính xác
        valid_pitch = pitch_values[pitch_values > 0]
        max_f0 = np.max(valid_pitch) if len(valid_pitch) > 0 else 0
        mean_f0 = np.mean(valid_pitch) if len(valid_pitch) > 0 else 0

        # 4. Trích xuất Cường độ (Intensity)
        intensity = snd.to_intensity()
        max_intensity = intensity.get_maximum()
        mean_intensity = np.mean(intensity.values)

        # 5. Tính toán thời lượng (Duration)
        duration = snd.get_total_duration()

        # Trả về kết quả kỹ thuật chuẩn hóa sang kiểu dữ liệu float cơ bản
        return {
            "pitch": {
                "max": round(float(max_f0), 2),
                "mean": round(float(mean_f0), 2)
            },
            "intensity": {
                "max": round(float(max_intensity), 2),
                "mean": round(float(mean_intensity), 2)
            },
            "duration": round(float(duration), 2)
        }

    except Exception as e:
        # Xử lý lỗi ngoại lệ để không làm sập toàn bộ Pipeline
        print(f"--- ❌ Lỗi xử lý âm thanh trong Prosody Engine: {e} ---")
        return {
            "pitch": {"max": 0, "mean": 0},
            "intensity": {"max": 0, "mean": 0},
            "duration": 0
        }

# --- Demo chạy thử để kiểm tra cục bộ ---
if __name__ == "__main__":
    # Thay đường dẫn file thực tế của cậu để test
    # results = analyze_prosody("data/samples/recording.wav")
    # print(f"--- 📊 Prosody Log: {results} ---")
    pass