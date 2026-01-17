import parselmouth
import librosa
import numpy as np

def analyze_prosody(audio_path):
    try:
        # Giải mã bằng Librosa trước khi đưa vào Parselmouth
        y, sr = librosa.load(audio_path, sr=None)
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        
        # Trích xuất Pitch & Intensity
        pitch = snd.to_pitch()
        valid_pitch = pitch.selected_array['frequency'][pitch.selected_array['frequency'] > 0]
        mean_f0 = np.mean(valid_pitch) if len(valid_pitch) > 0 else 0
        
        intensity = snd.to_intensity()
        mean_intensity = np.mean(intensity.values)

        return {
            "pitch": {"mean": round(float(mean_f0), 2)},
            "intensity": {"mean": round(float(mean_intensity), 2)},
            "duration": round(float(snd.get_total_duration()), 2)
        }
    except Exception as e:
        return {"pitch": {"mean": 0}, "intensity": {"mean": 0}, "duration": 0}