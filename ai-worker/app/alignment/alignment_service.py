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


SAFE_METADATA_KEYS = {
    "alignment_method",
    "alignment_status",
    "is_forced_alignment",
    "requested_alignment_method",
    "fallback_alignment",
    "mfa_used",
    "mfa_exit_code",
    "textgrid_parse_success",
    "word_segments_count",
    "phone_segments_count",
    "fallback_reason",
}


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_reason(reason: str) -> str:
    sanitized = " ".join(str(reason or "").split())
    sanitized = re.sub(r"[A-Za-z]:\\\S+", "[local-path]", sanitized)
    sanitized = re.sub(r"/\S+", "[local-path]", sanitized)
    return sanitized[:500] or "Alignment failed."


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key in SAFE_METADATA_KEYS}


def _failed_alignment(method: str, error: str, requested_method: str | None = None) -> dict[str, Any]:
    sanitized_error = _sanitize_reason(error)
    return build_alignment_result(
        segments=[],
        status="failed",
        method=method,
        note=sanitized_error,
        metadata=_safe_metadata(
            {
                "alignment_method": method,
                "alignment_status": "failed",
                "is_forced_alignment": False,
                "requested_alignment_method": requested_method or method,
                "fallback_alignment": False,
                "mfa_used": False,
            }
        ),
    )


def _fallback_alignment(
    audio_path: str | Path,
    prompt_text: str | None,
    canonical_phones: list[str] | tuple[str, ...] | None,
    fallback_reason: str | None = None,
    requested_method: str | None = None,
) -> dict[str, Any]:
    result = align_prompt_fallback(audio_path, prompt_text=prompt_text, canonical_phones=canonical_phones)
    result["status"] = "fallback" if requested_method else result.get("status", "completed")
    result["alignment_status"] = result["status"]
    result["metadata"].update(
        {
            "alignment_method": "fallback_even_split",
            "alignment_status": result["alignment_status"],
            "is_forced_alignment": False,
            "requested_alignment_method": requested_method or "fallback",
            "fallback_alignment": True,
            "mfa_used": False,
            "word_segments_count": len(result.get("words") or []),
            "phone_segments_count": len(result.get("phones") or []),
        }
    )
    if fallback_reason:
        sanitized_reason = _sanitize_reason(fallback_reason)
        result["metadata"]["fallback_reason"] = sanitized_reason
        result["note"] = f"{FALLBACK_ALIGNMENT_NOTE} Fallback reason: {sanitized_reason}"
    result["metadata"] = _safe_metadata(result["metadata"])
    return result


def align_audio(
    audio_path: str | Path,
    prompt_text: str | None,
    audio_duration: float | None = None,
    canonical_phones: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Select an alignment provider and return the normalized alignment contract."""

    mode = os.getenv("ALIGNMENT_MODE", "fallback").strip().lower()
    allow_fallback = _env_bool("ALLOW_ALIGNMENT_FALLBACK", True)

    if mode == "none":
        return _failed_alignment("none", "Alignment disabled by ALIGNMENT_MODE=none.")

    if mode == "fallback":
        result = _fallback_alignment(audio_path, prompt_text, canonical_phones)
        if audio_duration is not None:
            result["metadata"]["requested_audio_duration_seconds"] = float(audio_duration)
        return result

    if mode == "mfa":
        try:
            result = run_mfa_alignment(
                audio_path,
                prompt_text or "",
                dictionary_path=os.getenv("MFA_DICTIONARY_PATH"),
                acoustic_model_path=os.getenv("MFA_ACOUSTIC_MODEL_PATH"),
            )
            result["metadata"] = _safe_metadata(result.get("metadata") or {})
            return result
        except AlignmentError as exc:
            if allow_fallback:
                return _fallback_alignment(
                    audio_path,
                    prompt_text,
                    canonical_phones,
                    fallback_reason=str(exc),
                    requested_method="mfa",
                )
            return _failed_alignment("mfa", str(exc), requested_method="mfa")

    error = f"Unsupported ALIGNMENT_MODE={mode!r}. Use mfa, fallback, or none."
    if allow_fallback:
        return _fallback_alignment(audio_path, prompt_text, canonical_phones, fallback_reason=error)
    return _failed_alignment(mode, error)
