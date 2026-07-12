from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio.preprocessing import AudioPreprocessingConfig, AudioPreprocessingError, preprocess_audio
from app.contracts.alignment_contract import AlignmentError


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True)
class PreparedMfaAudio:
    path: Path
    duration_seconds: float
    sample_rate: int
    samples: int
    peak_amplitude: float
    rms_energy: float


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in {None, ""}:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise AlignmentError(f"{name} must be a number.", code="audio_invalid") from exc
    if value < 0:
        raise AlignmentError(f"{name} must be non-negative.", code="audio_invalid")
    return value


def prepare_audio_for_mfa(source_audio: str | Path, output_path: str | Path) -> PreparedMfaAudio:
    """Decode audio through the shared preprocessor and write a validated MFA WAV.

    Silence trimming and truncation stay disabled. MFA timestamps must retain the
    same origin and duration as the audio later cropped by the CNN scorer.
    """

    import numpy as np

    source = Path(source_audio).expanduser()
    destination = Path(output_path).expanduser()
    if not source.is_file() or source.stat().st_size == 0:
        raise AlignmentError("Audio file is missing or empty.", code="audio_empty")
    try:
        if source.resolve() == destination.resolve():
            raise AlignmentError("MFA audio preparation must not overwrite the source audio.", code="audio_invalid")
    except OSError:
        pass

    config = AudioPreprocessingConfig(
        target_sample_rate=TARGET_SAMPLE_RATE,
        min_duration_seconds=0.0,
        max_duration_seconds=max(_env_float("MFA_MAX_AUDIO_DURATION_SECONDS", 60.0), 1.0),
        trim_silence=False,
        normalize=False,
        denoise=False,
        denoise_strength="light",
        noise_profile_seconds=0.3,
    )
    try:
        result = preprocess_audio(source, config=config)
    except AudioPreprocessingError as exc:
        raise AlignmentError("Audio could not be decoded for MFA.", code="audio_invalid") from exc

    if bool(result.metadata.get("is_too_long")):
        raise AlignmentError(
            "Audio exceeds MFA_MAX_AUDIO_DURATION_SECONDS and was not aligned to avoid changing timestamps.",
            code="audio_too_long",
        )

    samples = np.asarray(result.waveform, dtype=np.float32)
    if samples.size == 0:
        raise AlignmentError("Audio has no decoded samples.", code="audio_empty")
    if not np.isfinite(samples).all():
        raise AlignmentError("Audio contains non-finite samples.", code="audio_invalid")

    duration = samples.size / TARGET_SAMPLE_RATE
    if duration < _env_float("MFA_MIN_AUDIO_DURATION_SECONDS", 0.1):
        raise AlignmentError("Audio is too short for MFA alignment.", code="audio_too_short")
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if peak < 1e-5 or rms < 1e-6:
        raise AlignmentError("Audio is silent and cannot be force aligned.", code="audio_silent")

    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2", copy=False)
    try:
        with wave.open(str(destination), "wb") as wav_file:
            wav_file.setnchannels(TARGET_CHANNELS)
            wav_file.setsampwidth(TARGET_SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(TARGET_SAMPLE_RATE)
            wav_file.writeframes(pcm16.tobytes())
        with wave.open(str(destination), "rb") as wav_file:
            if (wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth()) != (
                TARGET_SAMPLE_RATE,
                TARGET_CHANNELS,
                TARGET_SAMPLE_WIDTH_BYTES,
            ):
                raise AlignmentError("Prepared MFA WAV has an unexpected format.", code="audio_invalid")
            if wav_file.getnframes() <= 0:
                raise AlignmentError("Prepared MFA WAV has no frames.", code="audio_empty")
    except (OSError, wave.Error) as exc:
        raise AlignmentError("Unable to write or verify MFA-ready WAV audio.", code="audio_invalid") from exc

    return PreparedMfaAudio(destination, duration, TARGET_SAMPLE_RATE, int(samples.size), peak, rms)


def validate_prepared_mfa_wav(audio_path: str | Path) -> None:
    """Verify that an existing file is safe to reuse for MFA and CNN crops."""

    path = Path(audio_path).expanduser()
    if path.suffix.lower() != ".wav":
        raise AlignmentError("Prepared audio must be a WAV file.", code="audio_invalid")
    if not path.is_file() or path.stat().st_size == 0:
        raise AlignmentError("Prepared WAV audio is missing or empty.", code="audio_missing")
    try:
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise AlignmentError("Prepared WAV must use uncompressed PCM.", code="audio_invalid")
            if (wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth()) != (
                TARGET_SAMPLE_RATE,
                TARGET_CHANNELS,
                TARGET_SAMPLE_WIDTH_BYTES,
            ):
                raise AlignmentError("Prepared WAV has an unexpected format.", code="audio_invalid")
            if wav_file.getnframes() <= 0:
                raise AlignmentError("Prepared WAV has no frames.", code="audio_empty")
    except (OSError, wave.Error) as exc:
        raise AlignmentError("Prepared WAV could not be verified.", code="audio_invalid") from exc
