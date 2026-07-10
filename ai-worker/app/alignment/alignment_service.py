from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from app.alignment.fallback_aligner import align_prompt_fallback
from app.alignment.mfa_aligner import run_mfa_alignment
from app.contracts.alignment_contract import (
    FALLBACK_ALIGNMENT_NOTE,
    AlignmentError,
    build_alignment_result,
)

FALLBACK_ALIGNMENT_WARNING = "Fallback alignment is approximate and has limited location reliability."
SENSITIVE_REASON_MARKERS = (
    "c:\\",
    "/tmp/",
    "appdata\\local\\temp",
    ".textgrid",
    ".wav",
    ".webm",
    ".m4a",
    "x-amz-signature=",
    "token=",
    "sig=",
    "signature=",
)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _failed_alignment(method: str, error: str, error_code: str = "alignment_failed") -> dict[str, Any]:
    return build_alignment_result(
        segments=[],
        status="failed",
        method=method,
        note=error,
        metadata={
            "is_forced_alignment": False,
            "is_fallback": False,
            "fallback_alignment": False,
            "mfa_used": False,
            "textgrid_parse_success": False,
            "error": error,
            "error_code": error_code,
            "alignment_source": "none",
        },
        quality={"status": "failed", "quality_score": 0.0, "issues": [error_code], "metrics": {}},
        error={"code": error_code, "message": error},
    )


def _sanitize_alignment_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    sanitized = str(reason)
    sanitized = re.sub(r"[A-Za-z]:\\[^:;\r\n]+", "[redacted-local-path]", sanitized)
    sanitized = re.sub(r"/tmp/[^\s:;]+", "[redacted-local-path]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"https?://\S+", "[redacted-url]", sanitized, flags=re.IGNORECASE)
    normalized = sanitized.lower()
    if any(marker in normalized for marker in SENSITIVE_REASON_MARKERS):
        return "MFA alignment failed before a forced-alignment result was produced."
    return sanitized


def _fallback_alignment(
    audio_path: str | Path,
    prompt_text: str | None,
    canonical_phones: list[str] | tuple[str, ...] | None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    result = align_prompt_fallback(audio_path, prompt_text=prompt_text, canonical_phones=canonical_phones)
    sanitized_reason = _sanitize_alignment_reason(fallback_reason)
    result["status"] = "fallback"
    result["alignment_status"] = "fallback"
    result["metadata"]["is_forced_alignment"] = False
    result["metadata"]["is_fallback"] = True
    result["metadata"]["fallback_alignment"] = True
    result["metadata"]["mfa_used"] = False
    result["metadata"]["textgrid_parse_success"] = False
    result["metadata"]["alignment_status"] = "fallback"
    result["metadata"]["alignment_note"] = FALLBACK_ALIGNMENT_WARNING
    result["metadata"]["location_reliability"] = "limited_fallback_alignment"
    result["metadata"]["alignment_source"] = "fallback"
    result["metadata"]["alignment_confidence"] = 0.0
    result["alignment_source"] = "fallback"
    result["quality"] = {
        "status": "warning",
        "quality_score": 0.0,
        "issues": ["fallback_alignment_not_forced"],
        "metrics": {"number_of_phones": len(result.get("phones") or [])},
    }
    if sanitized_reason:
        result["metadata"]["fallback_reason"] = sanitized_reason
    result["note"] = FALLBACK_ALIGNMENT_WARNING
    return result


def align_audio(
    audio_path: str | Path,
    prompt_text: str | None,
    audio_duration: float | None = None,
    canonical_phones: list[str] | tuple[str, ...] | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Select an alignment provider and return the normalized alignment contract."""

    mode = os.getenv("ALIGNMENT_MODE", "fallback").strip().lower()
    allow_fallback = _env_bool("ALLOW_ALIGNMENT_FALLBACK", True)

    if mode == "none":
        result = _failed_alignment("none", "Alignment disabled by ALIGNMENT_MODE=none.", "alignment_disabled")
        result["metadata"]["requested_alignment_mode"] = mode
        result["metadata"]["requested_alignment_method"] = mode
        return result

    if mode == "fallback":
        result = _fallback_alignment(audio_path, prompt_text, canonical_phones)
        if audio_duration is not None:
            result["metadata"]["requested_audio_duration_seconds"] = float(audio_duration)
        result["metadata"]["requested_alignment_mode"] = mode
        result["metadata"]["requested_alignment_method"] = mode
        return result

    if mode == "mfa":
        try:
            result = run_mfa_alignment(
                audio_path,
                prompt_text or "",
                dictionary_path=os.getenv("MFA_DICTIONARY_PATH"),
                acoustic_model_path=os.getenv("MFA_ACOUSTIC_MODEL_PATH"),
                job_id=job_id,
            )
            result["metadata"]["requested_alignment_mode"] = mode
            result["metadata"]["requested_alignment_method"] = mode
            return result
        except AlignmentError as exc:
            print(f"alignment_event=mfa_fallback job_id={job_id or 'unknown'} error_category={exc.code}")
            if allow_fallback:
                result = _fallback_alignment(audio_path, prompt_text, canonical_phones, fallback_reason=str(exc))
                result["metadata"]["requested_alignment_mode"] = mode
                result["metadata"]["requested_alignment_method"] = mode
                result["metadata"]["mfa_attempted"] = True
                result["metadata"]["mfa_error"] = exc.as_dict()
                return result
            result = _failed_alignment("mfa", str(exc), exc.code)
            result["metadata"]["requested_alignment_mode"] = mode
            result["metadata"]["requested_alignment_method"] = mode
            result["metadata"]["mfa_attempted"] = True
            result["metadata"]["mfa_error"] = exc.as_dict()
            return result

    error = f"Unsupported ALIGNMENT_MODE={mode!r}. Use mfa, fallback, or none."
    if allow_fallback:
        result = _fallback_alignment(audio_path, prompt_text, canonical_phones, fallback_reason=error)
        result["metadata"]["requested_alignment_mode"] = mode
        result["metadata"]["requested_alignment_method"] = mode
        return result
    result = _failed_alignment(mode, error, "alignment_mode_invalid")
    result["metadata"]["requested_alignment_mode"] = mode
    result["metadata"]["requested_alignment_method"] = mode
    return result
