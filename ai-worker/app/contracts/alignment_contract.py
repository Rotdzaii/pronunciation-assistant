from __future__ import annotations

from typing import Any


FALLBACK_ALIGNMENT_NOTE = (
    "Fallback alignment is approximate scaffolding only and is not real forced alignment."
)
MFA_ALIGNMENT_NOTE = "MFA TextGrid forced alignment parsed successfully."


class AlignmentError(RuntimeError):
    """Raised when an alignment provider cannot produce a valid alignment."""


def build_alignment_segment(
    *,
    start: float,
    end: float,
    phone: str | None = None,
    word: str | None = None,
    index: int | None = None,
    segment_type: str = "phone",
) -> dict[str, Any]:
    return {
        "index": index,
        "type": segment_type,
        "phone": phone,
        "word": word,
        "start": round(float(start), 3),
        "end": round(float(end), 3),
    }


def build_alignment_result(
    *,
    segments: list[dict[str, Any]],
    status: str = "completed",
    method: str = "fallback_even_split",
    note: str = FALLBACK_ALIGNMENT_NOTE,
    metadata: dict[str, Any] | None = None,
    words: list[dict[str, Any]] | None = None,
    phones: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result_words = list(words or [segment for segment in segments if segment.get("type") == "word"])
    result_phones = list(phones or [segment for segment in segments if segment.get("type") == "phone"])
    return {
        "status": status,
        "alignment_status": status,
        "method": method,
        "alignment_method": method,
        "segments": list(segments),
        "words": result_words,
        "phones": result_phones,
        "note": note,
        "metadata": dict(metadata or {}),
    }


def get_alignment_segments(alignment_result: dict[str, Any]) -> list[dict[str, Any]]:
    segments = alignment_result.get("segments") or []
    return [segment for segment in segments if isinstance(segment, dict)]
