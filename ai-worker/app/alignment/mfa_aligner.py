from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from app.alignment.audio_preparation import prepare_audio_for_mfa
from app.alignment.quality import validate_alignment_quality
from app.alignment.textgrid_parser import parse_textgrid
from app.alignment.transcript import normalize_and_check_transcript
from app.contracts.alignment_contract import AlignmentError


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in {None, ""}:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise AlignmentError(f"{name} must be an integer.", code="mfa_configuration_invalid") from exc
    if parsed <= 0:
        raise AlignmentError(f"{name} must be greater than zero.", code="mfa_configuration_invalid")
    return parsed


def _configured_model(value: str | Path | None, name: str, error_code: str) -> str:
    if not value:
        raise AlignmentError(f"{name} is required for MFA alignment.", code=error_code)
    raw = str(value).strip()
    path = Path(raw).expanduser()
    looks_like_path = path.is_absolute() or "/" in raw or "\\" in raw or path.suffix.lower() in {".dict", ".zip", ".yaml"}
    if looks_like_path and not path.exists():
        raise AlignmentError(f"Configured {name} path does not exist.", code=error_code)
    return str(path) if path.exists() else raw


def _mfa_command_prefix(command: str, conda_env: str | None) -> list[str]:
    if conda_env:
        conda = shutil.which("conda")
        if not conda:
            raise AlignmentError("conda is required for MFA_CONDA_ENV but was not found.", code="mfa_not_installed")
        return [conda, "run", "-n", conda_env, command]
    resolved = shutil.which(command)
    if resolved:
        return [resolved]
    candidate = Path(command).expanduser()
    if candidate.is_file():
        return [str(candidate)]
    raise AlignmentError("MFA executable was not found. Configure MFA_COMMAND or MFA_CONDA_ENV.", code="mfa_not_installed")


def _win_to_wsl_path(path: Path) -> str:
    if not path.drive:
        return path.as_posix()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix()[len(path.drive):]}"


def _to_wsl_arg(value: str) -> str:
    path = Path(value)
    return _win_to_wsl_path(path) if path.drive else value


def _write_lab_file(path: Path, transcript: str) -> None:
    path.write_text(transcript + "\n", encoding="utf-8")


def _output_summary(output: str | None) -> str:
    text = (output or "").strip().replace("\r", " ").replace("\n", " ")
    return text[:1200]


def _extract_oov_words(output: str) -> list[str]:
    lowered = output.casefold()
    if not any(marker in lowered for marker in ("out of vocabulary", "oov", "not found in the dictionary")):
        return []
    candidates = re.findall(r"(?:out of vocabulary|oov|not found in the dictionary)[^\n:]*[:\-]?\s*([^\n]+)", output, re.I)
    words: set[str] = set()
    for candidate in candidates:
        for word in re.findall(r"[\w]+(?:['-][\w]+)*", candidate, flags=re.UNICODE):
            if word.casefold() not in {"out", "of", "vocabulary", "not", "found", "in", "the", "dictionary", "oov"}:
                words.add(word)
    return sorted(words)


def _failure_from_mfa_process(returncode: int, stdout: str, stderr: str) -> AlignmentError:
    output = "\n".join(part for part in (stderr, stdout) if part)
    lowered = output.casefold()
    oov_words = _extract_oov_words(output)
    if oov_words or "out of vocabulary" in lowered or "not found in the dictionary" in lowered:
        return AlignmentError(
            "MFA could not align transcript words absent from the dictionary.",
            code="oov",
            details={"oov_words": oov_words},
        )
    if "dictionary" in lowered and any(marker in lowered for marker in ("not found", "does not exist", "no such file")):
        code = "dictionary_missing"
    elif any(marker in lowered for marker in ("acoustic model", "model path")) and any(
        marker in lowered for marker in ("not found", "does not exist", "no such file")
    ):
        code = "acoustic_model_missing"
    else:
        code = "mfa_nonzero_exit"
    return AlignmentError(
        f"MFA returned exit code {returncode}.",
        code=code,
        details={"returncode": returncode, "stderr": _output_summary(stderr), "stdout": _output_summary(stdout)},
    )


def _debug_value(value: str) -> str:
    path = Path(value)
    return path.name if path.exists() else value


def run_mfa_alignment(
    audio_path: str | Path,
    prompt_text: str,
    output_dir: str | Path | None = None,
    dictionary_path: str | Path | None = None,
    acoustic_model_path: str | Path | None = None,
    *,
    job_id: str | None = None,
    keep_debug_artifacts: bool | None = None,
) -> dict[str, Any]:
    """Run MFA safely, parse its TextGrid, and reject invalid timing output."""

    source_audio = Path(audio_path).expanduser()
    if not source_audio.is_file():
        raise AlignmentError("Audio file was not found for MFA alignment.", code="audio_missing")
    dictionary = _configured_model(dictionary_path or os.getenv("MFA_DICTIONARY_PATH"), "MFA_DICTIONARY_PATH", "dictionary_missing")
    acoustic_model = _configured_model(
        acoustic_model_path or os.getenv("MFA_ACOUSTIC_MODEL_PATH"),
        "MFA_ACOUSTIC_MODEL_PATH",
        "acoustic_model_missing",
    )
    transcript = normalize_and_check_transcript(prompt_text, dictionary)
    if transcript.oov_words:
        raise AlignmentError(
            "Transcript contains words missing from the configured MFA dictionary.",
            code="oov",
            details={"oov_words": transcript.oov_words},
        )

    wsl_distro = os.getenv("MFA_WSL_DISTRO")
    use_wsl = bool(wsl_distro)
    if use_wsl:
        wsl_user = os.getenv("MFA_WSL_USER", "phoenix")
        wsl_binary = os.getenv("MFA_WSL_BINARY", "/home/phoenix/miniforge3/envs/mfa/bin/mfa")
    else:
        command_name = os.getenv("MFA_COMMAND", "mfa")
        conda_env = os.getenv("MFA_CONDA_ENV")
        command_prefix = _mfa_command_prefix(command_name, conda_env)

    timeout_seconds = _env_int("MFA_ALIGNMENT_TIMEOUT_SECONDS", 300)
    keep_debug = _env_bool("MFA_KEEP_DEBUG_ARTIFACTS", False) if keep_debug_artifacts is None else keep_debug_artifacts
    temp_root = Path(os.getenv("MFA_TEMP_DIR") or tempfile.gettempdir()).expanduser()
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="mfa-align-", dir=str(temp_root)))
    except OSError as exc:
        raise AlignmentError("MFA temporary directory is not writable.", code="mfa_temp_unavailable") from exc
    configured_output = Path(output_dir).expanduser() if output_dir else None
    runtime_started = time.monotonic()
    succeeded = False

    try:
        corpus_dir = work_dir / "corpus"
        align_output_dir = configured_output or work_dir / "aligned"
        try:
            corpus_dir.mkdir(parents=True, exist_ok=True)
            align_output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AlignmentError("MFA output directory is not writable.", code="mfa_output_unavailable") from exc
        working_audio = prepare_audio_for_mfa(source_audio, corpus_dir / f"{source_audio.stem}.wav")
        _write_lab_file(working_audio.path.with_suffix(".lab"), transcript.text)

        if use_wsl:
            command_args = [
                "wsl", "-d", str(wsl_distro), "-u", wsl_user, "--", wsl_binary, "align",
                _win_to_wsl_path(corpus_dir), _to_wsl_arg(dictionary), _to_wsl_arg(acoustic_model),
                _win_to_wsl_path(align_output_dir), "--clean", "--single_speaker",
            ]
        else:
            command_args = [*command_prefix, "align", str(corpus_dir), dictionary, acoustic_model, str(align_output_dir), "--clean", "--single_speaker"]

        print(
            "alignment_event=mfa_start "
            f"job_id={job_id or 'unknown'} audio_duration={working_audio.duration_seconds:.3f} "
            f"transcript={transcript.text!r} dictionary={_debug_value(dictionary)!r} "
            f"acoustic_model={_debug_value(acoustic_model)!r} timeout_seconds={timeout_seconds}"
        )
        try:
            completed = subprocess.run(command_args, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AlignmentError(
                f"MFA timed out after {timeout_seconds} seconds.",
                code="mfa_timeout",
                details={"timeout_seconds": timeout_seconds, "stdout": _output_summary(exc.stdout if isinstance(exc.stdout, str) else None), "stderr": _output_summary(exc.stderr if isinstance(exc.stderr, str) else None)},
            ) from exc
        except OSError as exc:
            raise AlignmentError("MFA executable could not be started.", code="mfa_not_installed") from exc

        if completed.returncode != 0:
            raise _failure_from_mfa_process(completed.returncode, completed.stdout, completed.stderr)

        textgrid_candidates = sorted(align_output_dir.rglob(f"{working_audio.path.stem}.TextGrid")) or sorted(align_output_dir.rglob("*.TextGrid"))
        if not textgrid_candidates:
            raise AlignmentError("MFA completed but did not create a TextGrid file.", code="textgrid_missing")
        textgrid_path = textgrid_candidates[0]
        result = parse_textgrid(textgrid_path)
        quality = validate_alignment_quality(
            words=result["words"],
            phones=result["phones"],
            audio_duration=working_audio.duration_seconds,
            expected_word_count=len(transcript.words),
            oov_count=len(transcript.oov_words),
            empty_interval_count=int(result["metadata"].get("empty_interval_count", 0)),
        )
        if quality["status"] == "failed":
            raise AlignmentError("MFA created an invalid or low-quality TextGrid.", code="textgrid_invalid", details={"quality": quality})

        runtime = time.monotonic() - runtime_started
        result["status"] = "warning" if quality["status"] == "warning" else "success"
        result["alignment_status"] = result["status"]
        result["quality"] = quality
        result["metadata"].update(
            {
                "alignment_source": "mfa",
                "alignment_confidence": quality["quality_score"],
                "audio_duration_seconds": round(working_audio.duration_seconds, 3),
                "mfa_runtime_seconds": round(runtime, 3),
                "mfa_exit_code": completed.returncode,
                "mfa_dictionary": _debug_value(dictionary),
                "mfa_acoustic_model": _debug_value(acoustic_model),
                "oov_words": transcript.oov_words,
                "dictionary_vocabulary_checked": transcript.dictionary_checked,
                "textgrid_path": str(textgrid_path),
            }
        )
        print(
            "alignment_event=mfa_complete "
            f"job_id={job_id or 'unknown'} runtime_seconds={runtime:.3f} words={len(result['words'])} "
            f"phones={len(result['phones'])} oov={len(transcript.oov_words)} quality_status={quality['status']}"
        )
        succeeded = True
        return result
    except AlignmentError as exc:
        exc.details.setdefault("runtime_seconds", round(time.monotonic() - runtime_started, 3))
        if keep_debug:
            exc.details.setdefault("debug_artifact_dir", str(work_dir))
        print(f"alignment_event=mfa_failed job_id={job_id or 'unknown'} error_category={exc.code}")
        raise
    finally:
        if succeeded or not keep_debug:
            shutil.rmtree(work_dir, ignore_errors=True)
