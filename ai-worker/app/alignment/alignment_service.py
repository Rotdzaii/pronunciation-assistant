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
        return _failed_alignment("none", "Alignment disabled by ALIGNMENT_MODE=none.")

    if mode == "fallback":
        result = _fallback_alignment(audio_path, prompt_text, canonical_phones)
        if audio_duration is not None:
            result["metadata"]["requested_audio_duration_seconds"] = float(audio_duration)
        return result

    if mode == "mfa":
        try:
            return run_mfa_alignment(
                audio_path,
                prompt_text or "",
                dictionary_path=os.getenv("MFA_DICTIONARY_PATH"),
                acoustic_model_path=os.getenv("MFA_ACOUSTIC_MODEL_PATH"),
            )
        except AlignmentError as exc:
            if allow_fallback:
                return _fallback_alignment(audio_path, prompt_text, canonical_phones, fallback_reason=str(exc))
            return _failed_alignment("mfa", str(exc))

    error = f"Unsupported ALIGNMENT_MODE={mode!r}. Use mfa, fallback, or none."
    if allow_fallback:
        return _fallback_alignment(audio_path, prompt_text, canonical_phones, fallback_reason=error)
    return _failed_alignment(mode, error)
