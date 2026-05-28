from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.alignment.textgrid_parser import parse_textgrid
from app.contracts.alignment_contract import AlignmentError


def _configured_path(value: str | None, name: str) -> Path:
    if not value:
        raise AlignmentError(f"{name} is required for MFA alignment.")
    path = Path(value).expanduser()
    if not path.exists():
        raise AlignmentError(f"{name} does not exist: {path}")
    return path


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


def _write_lab_file(path: Path, prompt_text: str) -> None:
    text = " ".join(str(prompt_text or "").split())
    if not text:
        raise AlignmentError("prompt_text is required for MFA alignment.")
    path.write_text(text + "\n", encoding="utf-8")


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

    command = _mfa_command(os.getenv("MFA_COMMAND", "mfa"))
    dictionary = _configured_path(str(dictionary_path or os.getenv("MFA_DICTIONARY_PATH") or ""), "MFA_DICTIONARY_PATH")
    acoustic_model = _configured_path(
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

        working_audio = corpus_dir / source_audio.name
        shutil.copy2(source_audio, working_audio)
        _write_lab_file(working_audio.with_suffix(".lab"), prompt_text)

        command_args = [
            command,
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

        textgrid_candidates = sorted(align_output_dir.rglob(f"{source_audio.stem}.TextGrid"))
        if not textgrid_candidates:
            textgrid_candidates = sorted(align_output_dir.rglob("*.TextGrid"))
        if not textgrid_candidates:
            raise AlignmentError(f"MFA completed but no TextGrid was found in {align_output_dir}")

        result = parse_textgrid(textgrid_candidates[0])
        result["metadata"]["mfa_command"] = os.getenv("MFA_COMMAND", "mfa")
        result["metadata"]["mfa_output_dir"] = str(align_output_dir)
        return result
