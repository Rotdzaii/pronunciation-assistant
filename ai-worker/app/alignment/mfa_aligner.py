from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

from app.alignment.textgrid_parser import parse_textgrid
from app.contracts.alignment_contract import AlignmentError


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH_BYTES = 2


def _configured_model(value: str | None, name: str) -> str:
    if not value:
        raise AlignmentError(f"{name} is required for MFA alignment.")
    path = Path(value).expanduser()
    if not path.exists():
        return str(value)
    return str(path)


def _mfa_command(command: str) -> str:
    resolved = shutil.which(command)
    if resolved:
        return resolved
    command_path = Path(command).expanduser()
    if command_path.exists():
        return str(command_path)
    raise AlignmentError(
        f"MFA command not found: {command}. Install/configure MFA locally or use ALIGNMENT_MODE=fallback."
    )


def _mfa_command_prefix(command: str, conda_env: str | None) -> list[str]:
    if conda_env:
        return ["conda", "run", "-n", conda_env, command]
    return [_mfa_command(command)]


def _write_lab_file(path: Path, prompt_text: str) -> None:
    text = " ".join(str(prompt_text or "").split())
    if not text:
        raise AlignmentError("prompt_text is required for MFA alignment.")
    path.write_text(text + "\n", encoding="utf-8")


def _prepare_audio_for_mfa(source_audio: Path, output_dir: Path) -> Path:
    if source_audio.suffix.lower() == ".wav":
        try:
            with wave.open(str(source_audio), "rb") as audio_file:
                if (
                    audio_file.getframerate() == TARGET_SAMPLE_RATE
                    and audio_file.getnchannels() == TARGET_CHANNELS
                    and audio_file.getsampwidth() == TARGET_SAMPLE_WIDTH_BYTES
                ):
                    prepared_audio = output_dir / source_audio.name
                    shutil.copy2(source_audio, prepared_audio)
                    return prepared_audio
        except (wave.Error, OSError):
            pass

    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise AlignmentError(
            "Preparing frontend audio for MFA requires librosa and numpy so browser audio can be normalized to "
            "16 kHz mono WAV."
        ) from exc

    try:
        audio, _ = librosa.load(str(source_audio), sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise AlignmentError("Unable to decode queued audio into MFA-ready WAV input.") from exc

    prepared_audio = output_dir / f"{source_audio.stem}.wav"
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    try:
        with wave.open(str(prepared_audio), "wb") as wav_file:
            wav_file.setnchannels(TARGET_CHANNELS)
            wav_file.setsampwidth(TARGET_SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(TARGET_SAMPLE_RATE)
            wav_file.writeframes(pcm16.tobytes())
    except OSError as exc:
        raise AlignmentError("Unable to write MFA-ready temporary WAV audio.") from exc

    return prepared_audio


def run_mfa_alignment(
    audio_path: str | Path,
    prompt_text: str,
    output_dir: str | Path | None = None,
    dictionary_path: str | Path | None = None,
    acoustic_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a locally configured MFA command and parse its TextGrid output."""

    source_audio = Path(audio_path).expanduser()
    if not source_audio.exists():
        raise AlignmentError(f"Audio file not found for MFA alignment: {source_audio}")

    command_name = os.getenv("MFA_COMMAND", "mfa")
    conda_env = os.getenv("MFA_CONDA_ENV")
    command_prefix = _mfa_command_prefix(command_name, conda_env)
    dictionary = _configured_model(str(dictionary_path or os.getenv("MFA_DICTIONARY_PATH") or ""), "MFA_DICTIONARY_PATH")
    acoustic_model = _configured_model(
        str(acoustic_model_path or os.getenv("MFA_ACOUSTIC_MODEL_PATH") or ""),
        "MFA_ACOUSTIC_MODEL_PATH",
    )

    temp_root = Path(os.getenv("MFA_TEMP_DIR") or tempfile.gettempdir()).expanduser()
    temp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mfa-align-", dir=str(temp_root)) as temp_dir:
        work_dir = Path(temp_dir)
        corpus_dir = work_dir / "corpus"
        align_output_dir = Path(output_dir).expanduser() if output_dir else work_dir / "aligned"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        align_output_dir.mkdir(parents=True, exist_ok=True)

        working_audio = _prepare_audio_for_mfa(source_audio, corpus_dir)
        _write_lab_file(working_audio.with_suffix(".lab"), prompt_text)

        command_args = [
            *command_prefix,
            "align",
            str(corpus_dir),
            str(dictionary),
            str(acoustic_model),
            str(align_output_dir),
            "--clean",
            "--single_speaker",
        ]
        completed = subprocess.run(
            command_args,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise AlignmentError(f"MFA alignment failed with exit code {completed.returncode}: {stderr}")

        textgrid_candidates = sorted(align_output_dir.rglob(f"{working_audio.stem}.TextGrid"))
        if not textgrid_candidates:
            textgrid_candidates = sorted(align_output_dir.rglob("*.TextGrid"))
        if not textgrid_candidates:
            raise AlignmentError("MFA completed but no TextGrid output was found.")

        result = parse_textgrid(textgrid_candidates[0])
        result["metadata"]["mfa_exit_code"] = completed.returncode
        result["metadata"]["mfa_command"] = command_name
        result["metadata"]["mfa_conda_env"] = conda_env
        return result
