from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from app.alignment.audio_preparation import PreparedMfaAudio, prepare_audio_for_mfa
from app.alignment.quality import validate_alignment_quality
from app.alignment.textgrid_parser import parse_textgrid
from app.alignment.transcript import normalize_and_check_transcript
from app.contracts.alignment_contract import AlignmentError


MFA_DIAGNOSTIC_TAIL_CHARS = 8000
_WINDOWS_PATH_RE = re.compile(r"(?i)\b[a-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*")
_TEMP_PATH_RE = re.compile(r"(?i)(?:/tmp|/var/tmp|/private/tmp)/[^\s:;,]+")
_WSL_MOUNT_PATH_RE = re.compile(r"(?i)/mnt/[a-z]/[^\s:;,]+")
_URL_RE = re.compile(r"https?://[^\s\"']+", flags=re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(x-amz-signature|token|sig|signature|secret|password|api[_-]?key)\b\s*([:=])\s*([^\s,;]+)"
)


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


def _mfa_runtime() -> str:
    runtime = os.getenv("MFA_RUNTIME", "conda").strip().lower() or "conda"
    if runtime not in {"conda", "wsl"}:
        raise AlignmentError("MFA_RUNTIME must be either 'conda' or 'wsl'.", code="mfa_runtime_invalid")
    return runtime


def _mfa_command_context() -> tuple[bool, list[str], dict[str, str | None]]:
    runtime = _mfa_runtime()
    if runtime == "wsl":
        required = {
            "MFA_WSL_DISTRO": os.getenv("MFA_WSL_DISTRO", "").strip(),
            "MFA_WSL_USER": os.getenv("MFA_WSL_USER", "").strip(),
            "MFA_WSL_BINARY": os.getenv("MFA_WSL_BINARY", "").strip(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise AlignmentError(
                "MFA_RUNTIME=wsl requires " + ", ".join(missing) + ".",
                code="mfa_wsl_configuration_invalid",
            )
        return True, [
            "wsl",
            "-d",
            required["MFA_WSL_DISTRO"],
            "-u",
            required["MFA_WSL_USER"],
            "--",
            required["MFA_WSL_BINARY"],
        ], {
            "mfa_runtime": "wsl",
            "command_mode": "wsl",
            "conda_env": None,
        }

    conda_env = os.getenv("MFA_CONDA_ENV", "").strip()
    if not conda_env:
        raise AlignmentError(
            "MFA_RUNTIME=conda requires MFA_CONDA_ENV.",
            code="mfa_conda_configuration_invalid",
        )
    command_name = os.getenv("MFA_COMMAND", "mfa").strip() or "mfa"
    return False, _mfa_command_prefix(command_name, conda_env), {
        "mfa_runtime": "conda",
        "command_mode": "conda_run",
        "conda_env": conda_env,
    }


def _win_to_wsl_path(path: Path) -> str:
    if not path.drive:
        return path.as_posix()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix()[len(path.drive):]}"


def _to_wsl_arg(value: str) -> str:
    path = Path(value)
    return _win_to_wsl_path(path) if path.drive else value


def _write_lab_file(path: Path, transcript: str) -> None:
    path.write_text(transcript + "\n", encoding="utf-8")


def sanitize_mfa_diagnostic(value: str | None) -> str:
    """Redact paths, URLs, and secret-like values without hiding MFA errors."""

    text = str(value or "")
    text = _URL_RE.sub("[redacted-url]", text)
    text = _SECRET_VALUE_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[redacted]", text)
    text = _WINDOWS_PATH_RE.sub("[redacted-local-path]", text)
    text = _WSL_MOUNT_PATH_RE.sub("[redacted-local-path]", text)
    return _TEMP_PATH_RE.sub("[redacted-local-path]", text)


def _diagnostic_tail(value: str | None) -> str:
    sanitized = sanitize_mfa_diagnostic(value).strip()
    return sanitized[-MFA_DIAGNOSTIC_TAIL_CHARS:]


def _short_output(value: str | None) -> str:
    return _diagnostic_tail(value).replace("\r", " ").replace("\n", " ")[:1200]


def _exception_tail(stdout: str | None, stderr: str | None) -> str | None:
    combined = "\n".join(item for item in (stderr, stdout) if item)
    lowered = combined.casefold()
    index = max(lowered.rfind("traceback"), lowered.rfind("exception"))
    return _diagnostic_tail(combined[index:]) if index >= 0 else None


def _debug_value(value: str) -> str:
    path = Path(value)
    return path.name if path.exists() else sanitize_mfa_diagnostic(value)


def _redacted_command_args(
    command_args: list[str],
    *,
    corpus_dir: Path | None = None,
    output_dir: Path | None = None,
    dictionary: str | None = None,
    acoustic_model: str | None = None,
) -> list[str]:
    redacted: list[str] = []
    for argument in command_args:
        corpus_variants = {str(corpus_dir)} if corpus_dir is not None else set()
        output_variants = {str(output_dir)} if output_dir is not None else set()
        if corpus_dir is not None and corpus_dir.drive:
            corpus_variants.add(_win_to_wsl_path(corpus_dir))
        if output_dir is not None and output_dir.drive:
            output_variants.add(_win_to_wsl_path(output_dir))
        if argument in corpus_variants:
            redacted.append("<corpus-dir>")
        elif argument in output_variants:
            redacted.append("<output-dir>")
        elif dictionary is not None and argument == dictionary and Path(dictionary).exists():
            redacted.append("<dictionary-path>")
        elif acoustic_model is not None and argument == acoustic_model and Path(acoustic_model).exists():
            redacted.append("<acoustic-model-path>")
        else:
            redacted.append(sanitize_mfa_diagnostic(argument))
    return redacted


def build_mfa_process_diagnostic(
    *,
    stage: str,
    command_args: list[str],
    command_context: dict[str, str | None],
    return_code: int | None,
    stdout: str | None,
    stderr: str | None,
    dictionary: str | None = None,
    acoustic_model: str | None = None,
    corpus_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    stderr_tail = _diagnostic_tail(stderr)
    stdout_tail = _diagnostic_tail(stdout)
    return {
        "stage": stage,
        "return_code": return_code,
        "mfa_runtime": command_context.get("mfa_runtime", "unknown"),
        "command_mode": command_context["command_mode"],
        "conda_env": command_context["conda_env"],
        "dictionary": _debug_value(dictionary) if dictionary else None,
        "acoustic_model": _debug_value(acoustic_model) if acoustic_model else None,
        "command": _redacted_command_args(
            command_args,
            corpus_dir=corpus_dir,
            output_dir=output_dir,
            dictionary=dictionary,
            acoustic_model=acoustic_model,
        ),
        "stderr_tail": stderr_tail,
        "stdout_tail": stdout_tail,
        "primary_output_tail": stderr_tail or stdout_tail,
        "exception_or_traceback_tail": _exception_tail(stdout, stderr),
    }


def _print_mfa_process_diagnostic(diagnostic: dict[str, Any]) -> None:
    print("MFA_PROCESS_DIAGNOSTIC")
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))


def _write_debug_artifacts(
    debug_dir: str | Path | None,
    *,
    job_id: str | None,
    prepared_audio: Path | None,
    transcript_lab: Path | None,
    stdout: str | None,
    stderr: str | None,
    diagnostic: dict[str, Any] | None,
) -> bool:
    if not debug_dir:
        return False
    try:
        safe_job_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", job_id or "unknown").strip(".-") or "unknown"
        artifact_dir = Path(debug_dir).expanduser() / f"mfa-{safe_job_id}-{int(time.time())}"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        if prepared_audio and prepared_audio.is_file():
            shutil.copy2(prepared_audio, artifact_dir / "prepared.wav")
        if transcript_lab and transcript_lab.is_file():
            shutil.copy2(transcript_lab, artifact_dir / "transcript.lab")
        (artifact_dir / "stdout.txt").write_text(_diagnostic_tail(stdout) + "\n", encoding="utf-8")
        (artifact_dir / "stderr.txt").write_text(_diagnostic_tail(stderr) + "\n", encoding="utf-8")
        (artifact_dir / "diagnostic.json").write_text(
            json.dumps(diagnostic or {}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (artifact_dir / "command.json").write_text(
            json.dumps((diagnostic or {}).get("command", []), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("MFA_DEBUG_ARTIFACTS")
        print(json.dumps({"saved": True, "directory": "<configured-mfa-debug-dir>"}, ensure_ascii=False))
        return True
    except OSError as exc:
        print("MFA_DEBUG_ARTIFACTS")
        print(json.dumps({"saved": False, "error": sanitize_mfa_diagnostic(str(exc))}, ensure_ascii=False))
        return False


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
    if "dictionary" in lowered and any(
        marker in lowered for marker in ("not found", "does not exist", "no such file", "could not find", "pretrainedmodelnotfounderror")
    ):
        code = "dictionary_missing"
    elif any(marker in lowered for marker in ("acoustic model", "model path")) and any(
        marker in lowered for marker in ("not found", "does not exist", "no such file", "could not find", "pretrainedmodelnotfounderror")
    ):
        code = "acoustic_model_missing"
    else:
        code = "mfa_nonzero_exit"
    return AlignmentError(
        f"MFA returned exit code {returncode}.",
        code=code,
        details={"returncode": returncode, "stderr": _short_output(stderr), "stdout": _short_output(stdout)},
    )


def run_mfa_preflight(
    dictionary: str | Path | None = None,
    acoustic_model: str | Path | None = None,
    *,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Check the configured MFA executable and selected model names without aligning audio."""

    dictionary_value = str(dictionary or os.getenv("MFA_DICTIONARY_PATH") or "")
    acoustic_value = str(acoustic_model or os.getenv("MFA_ACOUSTIC_MODEL_PATH") or "")
    _, command_prefix, command_context = _mfa_command_context()
    stages = [("version", [*command_prefix, "version"]), ("dictionary_models", [*command_prefix, "model", "list", "dictionary"]), ("acoustic_models", [*command_prefix, "model", "list", "acoustic"])]
    results: list[dict[str, Any]] = []
    command_outputs: dict[str, str] = {}
    for stage, command_args in stages:
        try:
            completed = subprocess.run(command_args, capture_output=True, text=True, timeout=timeout_seconds, check=False)
            diagnostic = build_mfa_process_diagnostic(
                stage=stage,
                command_args=command_args,
                command_context=command_context,
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                dictionary=dictionary_value,
                acoustic_model=acoustic_value,
            )
            command_outputs[stage] = "\n".join((completed.stdout or "", completed.stderr or ""))
        except subprocess.TimeoutExpired as exc:
            diagnostic = build_mfa_process_diagnostic(
                stage=stage,
                command_args=command_args,
                command_context=command_context,
                return_code=None,
                stdout=exc.stdout if isinstance(exc.stdout, str) else None,
                stderr=exc.stderr if isinstance(exc.stderr, str) else None,
                dictionary=dictionary_value,
                acoustic_model=acoustic_value,
            )
            diagnostic["error"] = "preflight_timeout"
        except OSError as exc:
            diagnostic = build_mfa_process_diagnostic(
                stage=stage,
                command_args=command_args,
                command_context=command_context,
                return_code=None,
                stdout=None,
                stderr=str(exc),
                dictionary=dictionary_value,
                acoustic_model=acoustic_value,
            )
            diagnostic["error"] = "preflight_command_start_failed"
        results.append(diagnostic)

    dictionary_listed = dictionary_value.casefold() in command_outputs.get("dictionary_models", "").casefold()
    acoustic_listed = acoustic_value.casefold() in command_outputs.get("acoustic_models", "").casefold()
    command_failed = any(item.get("return_code") != 0 for item in results)
    issues = []
    if command_failed:
        issues.append("preflight_command_failed")
    if not dictionary_listed:
        issues.append("dictionary_not_listed")
    if not acoustic_listed:
        issues.append("acoustic_model_not_listed")
    return {
        "status": "failed" if issues else "ok",
        "mfa_runtime": command_context["mfa_runtime"],
        "command_mode": command_context["command_mode"],
        "conda_env": command_context["conda_env"],
        "dictionary": _debug_value(dictionary_value),
        "acoustic_model": _debug_value(acoustic_value),
        "dictionary_listed": dictionary_listed,
        "acoustic_model_listed": acoustic_listed,
        "issues": issues,
        "commands": results,
    }


def run_mfa_alignment(
    audio_path: str | Path,
    prompt_text: str,
    output_dir: str | Path | None = None,
    dictionary_path: str | Path | None = None,
    acoustic_model_path: str | Path | None = None,
    *,
    job_id: str | None = None,
    keep_debug_artifacts: bool | None = None,
    debug: bool | None = None,
    debug_dir: str | Path | None = None,
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

    use_wsl, command_prefix, command_context = _mfa_command_context()
    timeout_seconds = _env_int("MFA_ALIGNMENT_TIMEOUT_SECONDS", 300)
    debug_enabled = bool(debug) if debug is not None else _env_bool("MFA_DEBUG", False)
    if keep_debug_artifacts:
        debug_enabled = True
    configured_debug_dir = str(debug_dir or os.getenv("MFA_DEBUG_DIR") or "").strip() or None
    temp_root = Path(os.getenv("MFA_TEMP_DIR") or tempfile.gettempdir()).expanduser()
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="mfa-align-", dir=str(temp_root)))
    except OSError as exc:
        raise AlignmentError("MFA temporary directory is not writable.", code="mfa_temp_unavailable") from exc
    configured_output = Path(output_dir).expanduser() if output_dir else None
    runtime_started = time.monotonic()
    succeeded = False
    working_audio: PreparedMfaAudio | None = None
    transcript_lab: Path | None = None
    last_stdout = ""
    last_stderr = ""
    last_diagnostic: dict[str, Any] | None = None

    try:
        corpus_dir = work_dir / "corpus"
        align_output_dir = configured_output or work_dir / "aligned"
        try:
            corpus_dir.mkdir(parents=True, exist_ok=True)
            align_output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AlignmentError("MFA output directory is not writable.", code="mfa_output_unavailable") from exc
        working_audio = prepare_audio_for_mfa(source_audio, corpus_dir / f"{source_audio.stem}.wav")
        transcript_lab = working_audio.path.with_suffix(".lab")
        _write_lab_file(transcript_lab, transcript.text)

        if use_wsl:
            command_args = [
                *command_prefix,
                "align",
                _win_to_wsl_path(corpus_dir),
                _to_wsl_arg(dictionary),
                _to_wsl_arg(acoustic_model),
                _win_to_wsl_path(align_output_dir),
                "--clean",
                "--single_speaker",
            ]
        else:
            command_args = [*command_prefix, "align", str(corpus_dir), dictionary, acoustic_model, str(align_output_dir), "--clean", "--single_speaker"]

        print(
            "alignment_event=mfa_start "
            f"job_id={job_id or 'unknown'} audio_duration={working_audio.duration_seconds:.3f} "
            f"mfa_runtime={command_context['mfa_runtime']} command_mode={command_context['command_mode']} "
            f"transcript={transcript.text!r} dictionary={_debug_value(dictionary)!r} "
            f"acoustic_model={_debug_value(acoustic_model)!r} timeout_seconds={timeout_seconds}"
        )
        try:
            completed = subprocess.run(command_args, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            last_stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            last_stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            last_diagnostic = build_mfa_process_diagnostic(
                stage="align",
                command_args=command_args,
                command_context=command_context,
                return_code=None,
                stdout=last_stdout,
                stderr=last_stderr,
                dictionary=dictionary,
                acoustic_model=acoustic_model,
                corpus_dir=corpus_dir,
                output_dir=align_output_dir,
            )
            last_diagnostic["error"] = "mfa_timeout"
            if debug_enabled:
                _print_mfa_process_diagnostic(last_diagnostic)
            raise AlignmentError(
                f"MFA timed out after {timeout_seconds} seconds.",
                code="mfa_timeout",
                details={"timeout_seconds": timeout_seconds, "stdout": _short_output(last_stdout), "stderr": _short_output(last_stderr)},
            ) from exc
        except OSError as exc:
            last_stderr = str(exc)
            last_diagnostic = build_mfa_process_diagnostic(
                stage="align",
                command_args=command_args,
                command_context=command_context,
                return_code=None,
                stdout=None,
                stderr=last_stderr,
                dictionary=dictionary,
                acoustic_model=acoustic_model,
                corpus_dir=corpus_dir,
                output_dir=align_output_dir,
            )
            last_diagnostic["error"] = "mfa_command_start_failed"
            if debug_enabled:
                _print_mfa_process_diagnostic(last_diagnostic)
            raise AlignmentError("MFA executable could not be started.", code="mfa_not_installed") from exc

        last_stdout = completed.stdout or ""
        last_stderr = completed.stderr or ""
        if completed.returncode != 0:
            last_diagnostic = build_mfa_process_diagnostic(
                stage="align",
                command_args=command_args,
                command_context=command_context,
                return_code=completed.returncode,
                stdout=last_stdout,
                stderr=last_stderr,
                dictionary=dictionary,
                acoustic_model=acoustic_model,
                corpus_dir=corpus_dir,
                output_dir=align_output_dir,
            )
            if debug_enabled:
                _print_mfa_process_diagnostic(last_diagnostic)
            raise _failure_from_mfa_process(completed.returncode, last_stdout, last_stderr)

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
            expected_words=transcript.words,
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
                "alignment_quality_status": quality["alignment_quality_status"],
                "alignment_quality_warnings": quality["alignment_quality_warnings"],
                "alignment_quality_failures": quality["alignment_quality_failures"],
                "audio_duration_seconds": round(working_audio.duration_seconds, 3),
                "mfa_runtime_seconds": round(runtime, 3),
                "mfa_exit_code": completed.returncode,
                "mfa_runtime": command_context["mfa_runtime"],
                "mfa_command_mode": command_context["command_mode"],
                "mfa_dictionary": _debug_value(dictionary),
                "mfa_acoustic_model": _debug_value(acoustic_model),
                "oov_words": transcript.oov_words,
                "dictionary_vocabulary_checked": transcript.dictionary_checked,
                "textgrid_path": str(textgrid_path),
                **{
                    key: value
                    for key, value in quality["metrics"].items()
                    if key
                    in {
                        "raw_audio_coverage_ratio",
                        "phone_span_fill_ratio",
                        "aligned_word_duration_seconds",
                        "aligned_phone_duration_seconds",
                        "aligned_span_seconds",
                        "leading_silence_seconds",
                        "trailing_silence_seconds",
                    }
                },
            }
        )
        print(
            "alignment_event=mfa_complete "
            f"job_id={job_id or 'unknown'} runtime_seconds={runtime:.3f} words={len(result['words'])} "
            f"phones={len(result['phones'])} oov={len(transcript.oov_words)} "
            f"quality_status={quality['alignment_quality_status']} "
            f"raw_coverage={quality['metrics']['raw_audio_coverage_ratio']:.3f} "
            f"phone_span_fill={quality['metrics']['phone_span_fill_ratio']:.3f} "
            f"warnings={','.join(quality['alignment_quality_warnings']) or 'none'} "
            f"failures={','.join(quality['alignment_quality_failures']) or 'none'}"
        )
        succeeded = True
        return result
    except AlignmentError as exc:
        exc.details.setdefault("runtime_seconds", round(time.monotonic() - runtime_started, 3))
        if debug_enabled and configured_debug_dir:
            _write_debug_artifacts(
                configured_debug_dir,
                job_id=job_id,
                prepared_audio=working_audio.path if working_audio else None,
                transcript_lab=transcript_lab,
                stdout=last_stdout,
                stderr=last_stderr,
                diagnostic=last_diagnostic,
            )
        print(f"alignment_event=mfa_failed job_id={job_id or 'unknown'} error_category={exc.code}")
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
