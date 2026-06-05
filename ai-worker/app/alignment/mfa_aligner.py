from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.alignment.textgrid_parser import parse_textgrid
from app.contracts.alignment_contract import AlignmentError


def _configured_model(value: str | None, name: str) -> str:
    if not value:
        raise AlignmentError(f"{name} is required for MFA alignment.")
    path = Path(value).expanduser()
    if path.exists():
        return str(path)
    return str(value)


def _mfa_command_prefix(command: str, conda_env: str | None = None) -> list[str]:
    if conda_env:
        return ["conda", "run", "-n", conda_env, command]

    resolved = shutil.which(command)
    if resolved:
        return [resolved]
    command_path = Path(command).expanduser()
    if command_path.exists():
        return [str(command_path)]
    raise AlignmentError(
        f"MFA command not found: {command}. Install/configure MFA locally or use ALIGNMENT_MODE=fallback."
    )


def _write_lab_file(path: Path, prompt_text: str) -> None:
    text = " ".join(str(prompt_text or "").split())
    if not text:
        raise AlignmentError("prompt_text is required for MFA alignment.")
    path.write_text(text + "\n", encoding="utf-8")


def _safe_mfa_metadata(result: dict[str, Any], *, exit_code: int, textgrid_parse_success: bool) -> dict[str, Any]:
    words = result.get("words") or []
    phones = result.get("phones") or []
    return {
        "alignment_method": "mfa",
        "alignment_status": result.get("alignment_status") or result.get("status"),
        "is_forced_alignment": True,
        "requested_alignment_method": "mfa",
        "fallback_alignment": False,
        "mfa_used": True,
        "mfa_exit_code": exit_code,
        "textgrid_parse_success": textgrid_parse_success,
        "word_segments_count": len(words),
        "phone_segments_count": len(phones),
    }


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
        raise AlignmentError("Audio file not found for MFA alignment.")

    command_prefix = _mfa_command_prefix(os.getenv("MFA_COMMAND", "mfa"), os.getenv("MFA_CONDA_ENV"))
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

        working_audio = corpus_dir / source_audio.name
        shutil.copy2(source_audio, working_audio)
        _write_lab_file(working_audio.with_suffix(".lab"), prompt_text)

        command_args = [
            *command_prefix,
            "align",
            str(corpus_dir),
            dictionary,
            acoustic_model,
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
            raise AlignmentError(f"MFA alignment failed with exit code {completed.returncode}.")

        textgrid_candidates = sorted(align_output_dir.rglob(f"{source_audio.stem}.TextGrid"))
        if not textgrid_candidates:
            textgrid_candidates = sorted(align_output_dir.rglob("*.TextGrid"))
        if not textgrid_candidates:
            raise AlignmentError("MFA completed but no TextGrid was found.")

        try:
            result = parse_textgrid(textgrid_candidates[0])
        except AlignmentError as exc:
            raise AlignmentError("MFA completed, but TextGrid parsing failed.") from exc
        result["metadata"] = _safe_mfa_metadata(
            result,
            exit_code=completed.returncode,
            textgrid_parse_success=True,
        )
        return result
