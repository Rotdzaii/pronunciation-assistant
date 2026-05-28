from __future__ import annotations

from typing import Any


FALLBACK_ALIGNMENT_NOTE = (
    "Fallback alignment is approximate scaffolding only and is not real forced alignment."
)


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
) -> dict[str, Any]:
    return {
        "status": status,
        "method": method,
        "segments": list(segments),
        "note": note,
        "metadata": dict(metadata or {}),
    }


def get_alignment_segments(alignment_result: dict[str, Any]) -> list[dict[str, Any]]:
    segments = alignment_result.get("segments") or []
    return [segment for segment in segments if isinstance(segment, dict)]
