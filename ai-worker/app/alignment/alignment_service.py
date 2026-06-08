from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.alignment.fallback_aligner import align_prompt_fallback
from app.alignment.mfa_aligner import run_mfa_alignment
from app.contracts.alignment_contract import (
    FALLBACK_ALIGNMENT_NOTE,
    AlignmentError,
    build_alignment_result,
)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _failed_alignment(method: str, error: str) -> dict[str, Any]:
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
        },
    )


def _fallback_alignment(
    audio_path: str | Path,
    prompt_text: str | None,
    canonical_phones: list[str] | tuple[str, ...] | None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    result = align_prompt_fallback(audio_path, prompt_text=prompt_text, canonical_phones=canonical_phones)
    result["metadata"]["is_forced_alignment"] = False
    result["metadata"]["is_fallback"] = True
    result["metadata"]["fallback_alignment"] = True
    result["metadata"]["mfa_used"] = False
    result["metadata"]["textgrid_parse_success"] = False
    if fallback_reason:
        result["metadata"]["fallback_reason"] = fallback_reason
        result["note"] = f"{FALLBACK_ALIGNMENT_NOTE} Fallback reason: {fallback_reason}"
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
        result = _failed_alignment("none", "Alignment disabled by ALIGNMENT_MODE=none.")
        result["metadata"]["requested_alignment_mode"] = mode
        return result

    if mode == "fallback":
        result = _fallback_alignment(audio_path, prompt_text, canonical_phones)
        if audio_duration is not None:
            result["metadata"]["requested_audio_duration_seconds"] = float(audio_duration)
        result["metadata"]["requested_alignment_mode"] = mode
        return result

    if mode == "mfa":
        try:
            result = run_mfa_alignment(
                audio_path,
                prompt_text or "",
                dictionary_path=os.getenv("MFA_DICTIONARY_PATH"),
                acoustic_model_path=os.getenv("MFA_ACOUSTIC_MODEL_PATH"),
            )
            result["metadata"]["requested_alignment_mode"] = mode
            return result
        except AlignmentError as exc:
            if allow_fallback:
                result = _fallback_alignment(audio_path, prompt_text, canonical_phones, fallback_reason=str(exc))
                result["metadata"]["requested_alignment_mode"] = mode
                result["metadata"]["mfa_attempted"] = True
                return result
            result = _failed_alignment("mfa", str(exc))
            result["metadata"]["requested_alignment_mode"] = mode
            result["metadata"]["mfa_attempted"] = True
            return result

    error = f"Unsupported ALIGNMENT_MODE={mode!r}. Use mfa, fallback, or none."
    if allow_fallback:
        result = _fallback_alignment(audio_path, prompt_text, canonical_phones, fallback_reason=error)
        result["metadata"]["requested_alignment_mode"] = mode
        return result
    result = _failed_alignment(mode, error)
    result["metadata"]["requested_alignment_mode"] = mode
    return result
