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
NOISE_WARNING = "Ban ghi co the co nhieu nen, ket qua AI co the kem chinh xac."


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
            warnings.append("Noise reduction failed; continuing with original preprocessed audio.")

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
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
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
        sanitized_detail = _sanitize_ffmpeg_detail(detail)
        print(f"audio_ffmpeg_conversion=failed,error:{sanitized_detail[:500]}")
        raise AudioDecodeError(f"ffmpeg audio conversion failed: {sanitized_detail}")

    return converted_path


def _duration_seconds(waveform: Any, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(getattr(waveform, "size", 0)) / sample_rate


def _sanitize_ffmpeg_detail(detail: str) -> str:
    import re

    sanitized = re.sub(r"[A-Za-z]:\\[^:\r\n]+", "[redacted-local-path]", detail)
    sanitized = re.sub(r"/tmp/[^\s:]+", "[redacted-local-path]", sanitized, flags=re.IGNORECASE)
    return sanitized


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


SNR_FLOOR_PERCENTILE = 10
SNR_MIN_DB_DEFAULT = 15.0
MIN_VOICED_DURATION_SECONDS_DEFAULT = 0.3
VOICED_FLOOR_MULTIPLIER = 3.0
_SNR_HOP_LENGTH = 512


# ---------------------------------------------------------------------------
# Hybrid VAD
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HybridVadConfig:
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    fmin: float = 80.0
    fmax: float = 400.0
    min_voiced_prob: float = 0.70
    energy_margin_db: float = 6.0
    hangover_ms: float = 120.0
    unvoiced_only_min_ratio: float = 0.10
    unvoiced_only_min_run_seconds: float = 0.30
    unvoiced_only_max_gap_seconds: float = 0.12

    @classmethod
    def from_env(cls) -> "HybridVadConfig":
        """Load provisional pitch-degraded fallback thresholds from the environment."""
        return cls(
            unvoiced_only_min_ratio=_env_float("VAD_UNVOICED_ONLY_MIN_RATIO", 0.10),
            unvoiced_only_min_run_seconds=_env_float("VAD_UNVOICED_ONLY_MIN_RUN_SECONDS", 0.30),
            unvoiced_only_max_gap_seconds=_env_float("VAD_UNVOICED_ONLY_MAX_GAP_SECONDS", 0.12),
        )


def _frames_to_segments(mask: Any, times: Any) -> list[tuple[float, float]]:
    """Convert a frame-level boolean mask to (start, end) time segments."""
    segments: list[tuple[float, float]] = []
    if not len(mask):
        return segments
    start_idx = None
    for i, is_on in enumerate(mask):
        if is_on and start_idx is None:
            start_idx = i
        elif not is_on and start_idx is not None:
            segments.append((float(times[start_idx]), float(times[i])))
            start_idx = None
    if start_idx is not None:
        segments.append((float(times[start_idx]), float(times[-1]) if len(times) else 0.0))
    return segments


def _close_short_gaps(mask: Any, max_gap_frames: int, np: Any) -> Any:
    """Fill false gaps bounded by speech frames without extending edge noise."""
    closed = np.asarray(mask, dtype=bool).copy()
    if max_gap_frames <= 0 or closed.size < 3:
        return closed

    index = 0
    while index < closed.size:
        if closed[index]:
            index += 1
            continue
        start = index
        while index < closed.size and not closed[index]:
            index += 1
        end = index
        if start > 0 and end < closed.size and (end - start) <= max_gap_frames:
            closed[start:end] = True
    return closed


def _largest_run_seconds(mask: Any, sample_rate: int, hop_length: int, np: Any) -> float:
    if sample_rate <= 0 or hop_length <= 0:
        return 0.0
    values = np.asarray(mask, dtype=bool)
    longest = 0
    current = 0
    for is_active in values:
        if is_active:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest * hop_length) / sample_rate


def _resolve_speech_mask(
    voiced_candidate: Any,
    unvoiced_candidate: Any,
    *,
    sample_rate: int,
    hop_length: int,
    cfg: HybridVadConfig,
    np: Any,
) -> dict[str, Any]:
    """Apply normal voiced-anchor or provisional pitch-degraded VAD policy."""
    voiced_candidate_count = int(np.sum(voiced_candidate))
    unvoiced_candidate_count = int(np.sum(unvoiced_candidate))
    frame_count = len(unvoiced_candidate)
    unvoiced_frames_ratio = float(unvoiced_candidate_count) / frame_count if frame_count else 0.0

    if voiced_candidate_count:
        # Existing voiced-anchor behavior remains unchanged.
        hangover_frames = max(1, int(cfg.hangover_ms / cfg.hop_ms))
        kernel = np.ones(2 * hangover_frames + 1, dtype=np.int32)
        voiced_neighborhood = np.convolve(voiced_candidate.astype(np.int32), kernel, mode="same") > 0
        speech_mask = voiced_candidate | (unvoiced_candidate & voiced_neighborhood)
        speech_mask = np.convolve(speech_mask.astype(np.int32), kernel, mode="same") > 0
        return {
            "speech_mask": speech_mask,
            "speech_detection_mode": "voiced_anchor",
            "audio_quality_status": "ok",
            "warning": None,
            "voiced_candidate_count": voiced_candidate_count,
            "unvoiced_candidate_count": unvoiced_candidate_count,
            "unvoiced_frames_ratio": unvoiced_frames_ratio,
            "largest_unvoiced_run_seconds": _largest_run_seconds(unvoiced_candidate, sample_rate, hop_length, np),
        }

    max_gap_frames = max(0, int(round(cfg.unvoiced_only_max_gap_seconds * sample_rate / hop_length)))
    sustained_unvoiced = _close_short_gaps(unvoiced_candidate, max_gap_frames, np)
    largest_unvoiced_run_seconds = _largest_run_seconds(sustained_unvoiced, sample_rate, hop_length, np)
    has_sustained_unvoiced_activity = (
        unvoiced_frames_ratio >= cfg.unvoiced_only_min_ratio
        and largest_unvoiced_run_seconds >= cfg.unvoiced_only_min_run_seconds
    )
    if has_sustained_unvoiced_activity:
        return {
            "speech_mask": sustained_unvoiced,
            "speech_detection_mode": "pitch_degraded_unvoiced_only",
            "audio_quality_status": "warning",
            "warning": "pitch_degraded_unvoiced_only",
            "voiced_candidate_count": 0,
            "unvoiced_candidate_count": unvoiced_candidate_count,
            "unvoiced_frames_ratio": unvoiced_frames_ratio,
            "largest_unvoiced_run_seconds": largest_unvoiced_run_seconds,
        }
    return {
        "speech_mask": np.zeros(frame_count, dtype=bool),
        "speech_detection_mode": "no_voiced_anchor",
        "audio_quality_status": "invalid",
        "warning": None,
        "voiced_candidate_count": 0,
        "unvoiced_candidate_count": unvoiced_candidate_count,
        "unvoiced_frames_ratio": unvoiced_frames_ratio,
        "largest_unvoiced_run_seconds": largest_unvoiced_run_seconds,
    }


def _detect_human_speech_waveform(
    y: Any,
    sr: int,
    cfg: HybridVadConfig,
) -> dict[str, Any]:
    """Run VAD on an already-decoded mono waveform to avoid repeat decoding."""
    import librosa
    import numpy as np

    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError("Audio file appears to be empty.")
    if sr <= 0:
        raise ValueError("Audio sample rate must be positive.")
    y = y - np.mean(y)

    frame_length = max(256, int(sr * cfg.frame_ms / 1000.0))
    hop_length = max(64, int(sr * cfg.hop_ms / 1000.0))

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=np.max)
    noise_floor_db = float(np.percentile(rms_db, 20))
    energetic = rms_db >= (noise_floor_db + cfg.energy_margin_db)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        sr=sr,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )
    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    n = min(len(rms_db), len(voiced_flag), len(voiced_prob), len(zcr), len(f0))
    rms_db = rms_db[:n]
    energetic = energetic[:n]
    f0 = f0[:n]
    voiced_flag = np.asarray(voiced_flag[:n], dtype=bool)
    voiced_prob = np.asarray(voiced_prob[:n], dtype=np.float32)
    zcr = zcr[:n]

    finite_f0 = np.isfinite(f0)
    human_f0 = finite_f0 & (f0 >= cfg.fmin) & (f0 <= cfg.fmax)
    voiced_candidate = energetic & voiced_flag & human_f0 & (voiced_prob >= cfg.min_voiced_prob)
    zcr_thr = float(np.percentile(zcr[energetic], 60)) if np.any(energetic) else float(np.percentile(zcr, 60))
    unvoiced_candidate = energetic & (zcr >= zcr_thr)

    mask_result = _resolve_speech_mask(
        voiced_candidate,
        unvoiced_candidate,
        sample_rate=sr,
        hop_length=hop_length,
        cfg=cfg,
        np=np,
    )
    speech_mask = mask_result["speech_mask"]

    times = librosa.times_like(rms_db, sr=sr, hop_length=hop_length)
    segments = _frames_to_segments(speech_mask, times)
    finite_values = f0[finite_f0]
    voiced_anchor_duration_seconds = float(mask_result["voiced_candidate_count"] * hop_length) / sr

    return {
        "sr": sr,
        "hop_length": hop_length,
        "times": times,
        "speech_mask": speech_mask,
        "segments": segments,
        "voiced_candidate": voiced_candidate,
        "unvoiced_candidate": unvoiced_candidate,
        "rms_db": rms_db,
        "zcr": zcr,
        "f0": f0,
        "voiced_flag": voiced_flag,
        "voiced_prob": voiced_prob,
        "noise_floor_db": noise_floor_db,
        "zcr_threshold": zcr_thr,
        "finite_f0_frames": int(np.sum(finite_f0)),
        "finite_f0_ratio": float(np.sum(finite_f0)) / n if n else 0.0,
        "f0_min_hz": float(np.min(finite_values)) if finite_values.size else None,
        "f0_max_hz": float(np.max(finite_values)) if finite_values.size else None,
        "pyin_voiced_flag_ratio": float(np.sum(voiced_flag)) / n if n else 0.0,
        "voiced_candidate_count": mask_result["voiced_candidate_count"],
        "unvoiced_candidate_count": mask_result["unvoiced_candidate_count"],
        "unvoiced_frames_ratio": mask_result["unvoiced_frames_ratio"],
        "largest_unvoiced_run_seconds": mask_result["largest_unvoiced_run_seconds"],
        "voiced_anchor_duration_seconds": voiced_anchor_duration_seconds,
        "speech_detection_mode": mask_result["speech_detection_mode"],
        "audio_quality_status": mask_result["audio_quality_status"],
        "warning": mask_result["warning"],
    }


def detect_human_speech(
    path: str,
    cfg: HybridVadConfig = HybridVadConfig(),
) -> dict[str, Any]:
    """Heuristic 3-state VAD (voiced / unvoiced / non-speech) using librosa + numpy.

    Logic:
      1) Energy gate: reject frames too close to the estimated noise floor.
      2) Voiced gate: energetic frames with human-range F0 and high voiced_prob.
      3) Unvoiced gate: high-ZCR energetic frames adjacent to voiced frames.
      4) Hangover smoothing: fill short gaps between speech frames.

    Returns a dict with speech segments, per-frame masks, and diagnostic arrays.
    """
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise AudioPreprocessingError("detect_human_speech requires librosa and numpy.") from exc

    if not isinstance(path, str) or not path:
        raise ValueError("`path` must be a non-empty string.")

    y, sr = librosa.load(path, sr=None, mono=True)
    return _detect_human_speech_waveform(y, sr, cfg)


# ---------------------------------------------------------------------------
# SNR + quality estimation (now VAD-backed)
# ---------------------------------------------------------------------------

def estimate_snr(audio_path: str | Path) -> dict[str, Any]:
    """Estimate audio quality metrics for the quality gate.

    voiced_duration_seconds is now derived from hybrid VAD segment durations
    (voiced + adjacent unvoiced frames with hangover) rather than energy-only
    frame counting, so flat ambient noise no longer accumulates speech time.

    snr_db is retained as a secondary diagnostic field; it is no longer the
    primary rejection criterion.

    Additional diagnostic fields:
      voiced_frames_ratio   — fraction of frames that are voiced candidates
      unvoiced_frames_ratio — fraction of frames that are unvoiced-only candidates
      mean_voiced_prob      — mean pyin voiced probability over energetic frames
    """
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise AudioPreprocessingError("estimate_snr requires librosa and numpy.") from exc

    # Prepared MFA WAVs are PCM mono 16 kHz. Read them directly so the VAD does
    # not trigger librosa/audioread decoding after preparation.
    path = Path(audio_path)
    if path.suffix.lower() == ".wav":
        import wave

        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise AudioPreprocessingError("VAD WAV input must be mono signed 16-bit PCM.")
            sr = wav_file.getframerate()
            audio = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2").astype(np.float32)
            audio /= 32768.0
    else:
        audio, sr = librosa.load(str(path), sr=16000, mono=True)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=_SNR_HOP_LENGTH)[0]

    _empty = {
        "snr_db": None,
        "file_duration_seconds": 0.0,
        "voiced_anchor_duration_seconds": 0.0,
        "detected_speech_duration_seconds": 0.0,
        "voiced_duration_seconds": 0.0,
        "voiced_frames_ratio": 0.0,
        "unvoiced_frames_ratio": 0.0,
        "mean_voiced_prob": 0.0,
        "finite_f0_frames": 0,
        "finite_f0_ratio": 0.0,
        "f0_min_hz": None,
        "f0_max_hz": None,
        "pyin_voiced_flag_ratio": 0.0,
        "voiced_candidate_count": 0,
        "unvoiced_candidate_count": 0,
        "largest_unvoiced_run_seconds": 0.0,
        "speech_detection_mode": "no_voiced_anchor",
        "audio_quality_status": "invalid",
        "warning": None,
    }
    if rms.size == 0 or float(np.max(rms)) < 1e-9:
        return _empty

    epsilon = 1e-8
    noise_floor_lin = float(np.percentile(rms, SNR_FLOOR_PERCENTILE))
    median_rms = float(np.median(rms))
    signal_frames = rms[rms > median_rms]
    signal_level = float(np.mean(signal_frames)) if signal_frames.size > 0 else float(np.max(rms))
    snr_db = round(20.0 * float(np.log10((signal_level + epsilon) / (noise_floor_lin + epsilon))), 2)

    # Hybrid VAD pass (slower due to pyin, provides accurate speech duration)
    try:
        vad = _detect_human_speech_waveform(audio, sr, HybridVadConfig.from_env())
    except Exception:
        # If VAD fails, fall back to energy-only voiced duration
        voiced_frame_count = int(np.sum(rms > noise_floor_lin * VOICED_FLOOR_MULTIPLIER))
        voiced_duration = round(float(voiced_frame_count * _SNR_HOP_LENGTH) / sr, 3) if sr > 0 else 0.0
        return {
            **_empty,
            "snr_db": snr_db,
            "file_duration_seconds": round(float(audio.size) / sr, 3) if sr > 0 else 0.0,
            "voiced_anchor_duration_seconds": voiced_duration,
            "detected_speech_duration_seconds": voiced_duration,
            "voiced_duration_seconds": voiced_duration,
            "speech_detection_mode": "vad_unavailable_energy_fallback",
            "audio_quality_status": "warning",
            "warning": "vad_unavailable_energy_fallback",
        }

    speech_duration = round(sum(end - start for start, end in vad["segments"]), 3)

    n = len(vad["voiced_candidate"])
    voiced_frames_ratio = round(float(np.sum(vad["voiced_candidate"])) / n, 3) if n > 0 else 0.0
    unvoiced_only = vad["unvoiced_candidate"] & ~vad["voiced_candidate"]
    unvoiced_frames_ratio = round(float(np.sum(unvoiced_only)) / n, 3) if n > 0 else 0.0

    # mean_voiced_prob over energetic frames (align lengths — pyin may differ by 1 frame)
    energetic_vad = vad["rms_db"] >= (vad["noise_floor_db"] + HybridVadConfig().energy_margin_db)
    vp = vad["voiced_prob"]
    min_n = min(len(energetic_vad), len(vp))
    energetic_vad = energetic_vad[:min_n]
    vp = vp[:min_n]
    mean_voiced_prob = round(float(np.mean(vp[energetic_vad])), 3) if np.any(energetic_vad) else 0.0

    return {
        "snr_db": snr_db,
        "file_duration_seconds": round(float(audio.size) / sr, 3) if sr > 0 else 0.0,
        "voiced_anchor_duration_seconds": round(float(vad["voiced_anchor_duration_seconds"]), 3),
        "detected_speech_duration_seconds": speech_duration,
        "voiced_duration_seconds": speech_duration,
        "voiced_frames_ratio": voiced_frames_ratio,
        "unvoiced_frames_ratio": unvoiced_frames_ratio,
        "mean_voiced_prob": mean_voiced_prob,
        "finite_f0_frames": int(vad["finite_f0_frames"]),
        "finite_f0_ratio": round(float(vad["finite_f0_ratio"]), 3),
        "f0_min_hz": round(float(vad["f0_min_hz"]), 2) if vad["f0_min_hz"] is not None else None,
        "f0_max_hz": round(float(vad["f0_max_hz"]), 2) if vad["f0_max_hz"] is not None else None,
        "pyin_voiced_flag_ratio": round(float(vad["pyin_voiced_flag_ratio"]), 3),
        "voiced_candidate_count": int(vad["voiced_candidate_count"]),
        "unvoiced_candidate_count": int(vad["unvoiced_candidate_count"]),
        "largest_unvoiced_run_seconds": round(float(vad["largest_unvoiced_run_seconds"]), 3),
        "speech_detection_mode": vad["speech_detection_mode"],
        "audio_quality_status": vad["audio_quality_status"],
        "warning": vad["warning"],
    }


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
