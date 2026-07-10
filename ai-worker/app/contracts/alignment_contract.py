from __future__ import annotations

from typing import Any


FALLBACK_ALIGNMENT_NOTE = (
    "Fallback alignment is approximate scaffolding only and is not real forced alignment."
)
MFA_ALIGNMENT_NOTE = "MFA TextGrid forced alignment parsed successfully."


class AlignmentError(RuntimeError):
    """A recoverable alignment failure with a stable error category."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "alignment_failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": self.details}


def build_alignment_segment(
    *,
    start: float,
    end: float,
    phone: str | None = None,
    word: str | None = None,
    index: int | None = None,
    segment_type: str = "phone",
) -> dict[str, Any]:
    normalized_start = round(float(start), 3)
    normalized_end = round(float(end), 3)
    return {
        "index": index,
        "type": segment_type,
        "phone": phone,
        "word": word,
        "start": normalized_start,
        "end": normalized_end,
        "duration": round(normalized_end - normalized_start, 3),
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
    quality: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
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
        "alignment_source": "mfa" if method == "mfa" else ("fallback" if method.startswith("fallback") else "none"),
        "quality": dict(quality or {}),
        "error": dict(error) if error else None,
    }


def get_alignment_segments(alignment_result: dict[str, Any]) -> list[dict[str, Any]]:
    segments = alignment_result.get("segments") or []
    return [segment for segment in segments if isinstance(segment, dict)]
