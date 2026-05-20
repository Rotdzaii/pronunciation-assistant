from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUIET_RMS_THRESHOLD = 0.005
QUIET_PEAK_THRESHOLD = 0.02
NORMALIZE_TARGET_PEAK = 0.95
NOISE_WARNING = "Bản ghi có thể có nhiễu nền, kết quả AI có thể kém chính xác."


class AudioPreprocessingError(RuntimeError):
    """Raised when audio cannot be decoded or preprocessed."""


class AudioDecodeError(AudioPreprocessingError):
    """Raised when ffmpeg or WAV decoding cannot read the submitted audio."""


@dataclass(frozen=True)
class AudioPreprocessingConfig:
    target_sample_rate: int
    min_duration_seconds: float
    max_duration_seconds: float
    trim_silence: bool
    normalize: bool
    denoise: bool
    denoise_strength: str
    noise_profile_seconds: float

    @classmethod
    def from_env(cls) -> "AudioPreprocessingConfig":
        return cls(
            target_sample_rate=_env_int("AUDIO_TARGET_SAMPLE_RATE", 16000),
            min_duration_seconds=_env_float("AUDIO_MIN_DURATION_SECONDS", 0.5),
            max_duration_seconds=_env_float("AUDIO_MAX_DURATION_SECONDS", 30.0),
            trim_silence=_env_bool("AUDIO_TRIM_SILENCE", True),
            normalize=_env_bool("AUDIO_NORMALIZE", True),
            denoise=_env_bool("AUDIO_DENOISE", False),
            denoise_strength=os.getenv("AUDIO_DENOISE_STRENGTH", "light").strip().lower(),
            noise_profile_seconds=_env_float("AUDIO_NOISE_PROFILE_SECONDS", 0.3),
        )


@dataclass(frozen=True)
class AudioPreprocessingResult:
    waveform: Any
    sample_rate: int
    metadata: dict[str, Any]


def preprocess_audio(
    audio_path: str | Path,
    config: AudioPreprocessingConfig | None = None,
    content_type: str | None = None,
) -> AudioPreprocessingResult:
    import librosa
    import numpy as np
    import soundfile as sf

    config = config or AudioPreprocessingConfig.from_env()
    input_path = Path(audio_path)
    original_path = str(input_path)
    original_suffix = input_path.suffix.lower() or None
    warnings: list[str] = []
    converted_path: Path | None = None
    ffmpeg_converted = False

    try:
        converted_path = _convert_to_wav(input_path, config.target_sample_rate)
        ffmpeg_converted = True
        print(
            "audio_ffmpeg_conversion=succeeded,"
            f"input_suffix:{original_suffix or 'none'},"
            f"content_type:{content_type or 'unknown'},"
            "converted_format:wav,"
            f"target_sample_rate:{config.target_sample_rate},"
            "mono:true"
        )
        waveform, source_sample_rate = sf.read(
            str(converted_path),
            dtype="float32",
            always_2d=False,
        )
    except AudioDecodeError:
        print(
            "audio_ffmpeg_conversion=failed,"
            f"input_suffix:{original_suffix or 'none'},"
            f"content_type:{content_type or 'unknown'}"
        )
        raise
    except Exception as exc:
        print(
            "audio_ffmpeg_conversion=failed,"
            f"input_suffix:{original_suffix or 'none'},"
            f"content_type:{content_type or 'unknown'}"
        )
        raise AudioDecodeError(
            "Audio could not be decoded with ffmpeg. Browser webm/m4a/mp3 input must be convertible to WAV."
        ) from exc
    finally:
        if converted_path is not None:
            try:
                converted_path.unlink(missing_ok=True)
            except OSError:
                pass

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        warnings.append("Audio has no samples after decode.")
        return _result(
            waveform,
            config.target_sample_rate,
            config,
            warnings,
            original_path=original_path,
            ffmpeg_converted=ffmpeg_converted,
            original_suffix=original_suffix,
            content_type=content_type,
        )

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1, dtype=np.float32)

    if source_sample_rate != config.target_sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=source_sample_rate,
            target_sr=config.target_sample_rate,
        ).astype(np.float32)

    if config.trim_silence and waveform.size:
        trimmed, index = librosa.effects.trim(waveform, top_db=30)
        if trimmed.size and trimmed.size < waveform.size:
            waveform = trimmed.astype(np.float32)
            warnings.append(
                f"Trimmed leading/trailing silence from sample {int(index[0])} to {int(index[1])}."
            )

    noise_warning = _detect_noise_warning(waveform)
    if noise_warning and not config.denoise:
        warnings.append(NOISE_WARNING)

    if config.denoise and waveform.size:
        try:
            waveform = _apply_light_denoise(waveform, config, np).astype(np.float32)
            warnings.append(
                "Applied light experimental denoise; conservative settings preserve final consonants and fricatives."
            )
        except Exception:
            warnings.append(
                "Noise reduction failed; continuing with original preprocessed audio."
            )

    max_samples = int(config.max_duration_seconds * config.target_sample_rate)
    is_too_long = max_samples > 0 and waveform.size > max_samples
    if is_too_long:
        waveform = waveform[:max_samples].astype(np.float32)
        warnings.append(
            f"Audio exceeded {config.max_duration_seconds:g}s and was truncated for scoring."
        )

    peak = _peak_amplitude(waveform)
    source_is_too_quiet = peak < QUIET_PEAK_THRESHOLD or _rms_energy(waveform) < QUIET_RMS_THRESHOLD
    if config.normalize and peak > 0:
        waveform = np.clip(
            waveform * (NORMALIZE_TARGET_PEAK / peak),
            -1.0,
            1.0,
        ).astype(np.float32)

    return _result(
        waveform,
        config.target_sample_rate,
        config,
        warnings,
        original_path=original_path,
        is_too_quiet_override=source_is_too_quiet,
        is_too_long=is_too_long,
        ffmpeg_converted=ffmpeg_converted,
        original_suffix=original_suffix,
        content_type=content_type,
    )


def _result(
    waveform: Any,
    sample_rate: int,
    config: AudioPreprocessingConfig,
    warnings: list[str],
    original_path: str,
    is_too_quiet_override: bool | None = None,
    is_too_long: bool = False,
    ffmpeg_converted: bool = False,
    original_suffix: str | None = None,
    content_type: str | None = None,
) -> AudioPreprocessingResult:
    duration_seconds = _duration_seconds(waveform, sample_rate)
    peak = _peak_amplitude(waveform)
    rms = _rms_energy(waveform)
    is_too_short = duration_seconds < config.min_duration_seconds
    is_too_quiet = (
        is_too_quiet_override
        if is_too_quiet_override is not None
        else peak < QUIET_PEAK_THRESHOLD or rms < QUIET_RMS_THRESHOLD
    )

    if is_too_short:
        warnings.append(
            f"Audio duration {duration_seconds:.2f}s is below minimum {config.min_duration_seconds:g}s."
        )
    if is_too_quiet:
        warnings.append("Audio level is very quiet and may reduce recognition quality.")

    print(
        "audio_preprocessing_info="
        f"input_suffix:{original_suffix or 'none'},"
        f"content_type:{content_type or 'unknown'},"
        f"duration_seconds:{duration_seconds:.2f},"
        f"sample_rate:{sample_rate}"
    )

    return AudioPreprocessingResult(
        waveform=waveform,
        sample_rate=sample_rate,
        metadata={
            "original_path": original_path,
            "duration_seconds": round(duration_seconds, 2),
            "sample_rate": sample_rate,
            "target_sample_rate": config.target_sample_rate,
            "original_suffix": original_suffix,
            "content_type": content_type,
            "is_too_short": is_too_short,
            "is_too_long": is_too_long,
            "is_too_quiet": is_too_quiet,
            "peak_amplitude": round(peak, 4),
            "rms_energy": round(rms, 4),
            "preprocessing": {
                "target_sample_rate": config.target_sample_rate,
                "mono": True,
                "normalized": config.normalize,
                "trimmed_silence": config.trim_silence,
                "ffmpeg_converted": ffmpeg_converted,
                "converted_format": "wav" if ffmpeg_converted else None,
                "denoise_enabled": config.denoise,
                "denoise_strength": config.denoise_strength,
                "noise_profile_seconds": config.noise_profile_seconds,
            },
            "denoise_enabled": config.denoise,
            "denoise_strength": config.denoise_strength,
            "noise_warning": NOISE_WARNING if NOISE_WARNING in warnings else None,
            "warnings": warnings,
        },
    )


def _convert_to_wav(audio_path: str | Path, target_sample_rate: int) -> Path:
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    converted_path = Path(temp_file.name)
    temp_file.close()

    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        str(converted_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        converted_path.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()
        print(f"audio_ffmpeg_conversion=failed,error:{detail[:500]}")
        raise AudioDecodeError(f"ffmpeg audio conversion failed: {detail}")

    return converted_path


def _duration_seconds(waveform: Any, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(getattr(waveform, "size", 0)) / sample_rate


def _peak_amplitude(waveform: Any) -> float:
    if getattr(waveform, "size", 0) == 0:
        return 0.0
    import numpy as np

    return float(np.max(np.abs(waveform)))


def _rms_energy(waveform: Any) -> float:
    if getattr(waveform, "size", 0) == 0:
        return 0.0
    import numpy as np

    return float(np.sqrt(np.mean(np.square(waveform))))


def _detect_noise_warning(waveform: Any) -> bool:
    if getattr(waveform, "size", 0) < 2048:
        return False
    import numpy as np

    frame_size = min(2048, waveform.size)
    start = np.asarray(waveform[:frame_size])
    full_rms = _rms_energy(waveform)
    start_rms = _rms_energy(start)
    peak = _peak_amplitude(waveform)
    if full_rms <= 0 or peak <= 0:
        return False
    return start_rms > max(0.01, full_rms * 0.6) and peak / full_rms < 18


def _apply_light_denoise(
    waveform: Any,
    config: AudioPreprocessingConfig,
    np: Any,
) -> Any:
    if config.denoise_strength not in {"light", "conservative"}:
        raise AudioPreprocessingError("AUDIO_DENOISE_STRENGTH currently supports only light")

    profile_samples = int(config.noise_profile_seconds * config.target_sample_rate)
    profile_samples = max(512, min(profile_samples, waveform.size))
    noise_profile = waveform[:profile_samples]
    if noise_profile.size < 512:
        return waveform

    frame_length = 512
    hop_length = 128
    window = np.hanning(frame_length).astype(np.float32)
    padded = np.pad(waveform, (0, frame_length), mode="constant")
    frames = []
    for start in range(0, max(1, padded.size - frame_length), hop_length):
        frames.append(padded[start : start + frame_length] * window)
    spectrum = np.fft.rfft(np.asarray(frames), axis=1)
    magnitude = np.abs(spectrum)
    phase = np.exp(1j * np.angle(spectrum))

    profile_padded = np.pad(noise_profile, (0, frame_length), mode="constant")
    profile_frames = []
    for start in range(0, max(1, profile_padded.size - frame_length), hop_length):
        profile_frames.append(profile_padded[start : start + frame_length] * window)
    noise_spectrum = np.fft.rfft(np.asarray(profile_frames), axis=1)
    noise_floor = np.percentile(np.abs(noise_spectrum), 60, axis=0)

    # Conservative spectral subtraction. A floor of 0.65 keeps fricatives and final
    # consonant bursts from being erased by stationary-noise estimation mistakes.
    reduced = np.maximum(magnitude - (noise_floor * 0.6), magnitude * 0.65)
    reconstructed = np.fft.irfft(reduced * phase, axis=1)

    output = np.zeros(padded.size, dtype=np.float32)
    weight = np.zeros(padded.size, dtype=np.float32)
    for index, frame in enumerate(reconstructed):
        start = index * hop_length
        output[start : start + frame_length] += frame[:frame_length] * window
        weight[start : start + frame_length] += window * window

    valid = weight > 1e-6
    output[valid] /= weight[valid]
    return output[: waveform.size]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise AudioPreprocessingError(f"{name} must be a number") from exc
    if parsed < 0:
        raise AudioPreprocessingError(f"{name} must be greater than or equal to 0")
    return parsed


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise AudioPreprocessingError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise AudioPreprocessingError(f"{name} must be greater than 0")
    return parsed
