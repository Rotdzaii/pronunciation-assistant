from __future__ import annotations

from typing import Any


SCORING_NOTE = "Pronunciation score unavailable: Phoenix v2 has no learned correctness or quality head."


def clamp_score(value: float | int | None) -> None:
    """Legacy helper retained for callers; Phoenix v2 never emits a score."""
    del value
    return None


def severity_from_score(score: float | int | None) -> str:
    del score
    return "unavailable"


def build_phone_score(
    *,
    phone: str | None,
    word: str | None = None,
    start: float | None = None,
    end: float | None = None,
    phone_score: float | int | None = None,
    gop_score_raw: float | int | None = None,
    gop_score_calibrated: float | int | None = None,
    duration_mismatch: float | int | None = None,
    severity: str | None = None,
    source: str = "unavailable",
) -> dict[str, Any]:
    """Compatibility shape without exposing heuristic phone quality values."""
    del phone_score, gop_score_raw, gop_score_calibrated, duration_mismatch, severity, source
    return {
        "phone": phone,
        "word": word,
        "start": round(float(start), 3) if start is not None else None,
        "end": round(float(end), 3) if end is not None else None,
        "score": None,
        "source": "unavailable",
    }


def build_word_score(
    *,
    word: str | None,
    start: float | None = None,
    end: float | None = None,
    score: float | int | None = None,
    phones: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    del score
    return {
        "word": word,
        "start": round(float(start), 3) if start is not None else None,
        "end": round(float(end), 3) if end is not None else None,
        "score": None,
        "phones": list(phones or []),
    }


def build_scoring_result(
    *,
    scoring_status: str,
    scoring_method: str,
    utterance_segmental_score: float | int | None = None,
    words: list[dict[str, Any]] | None = None,
    phones: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del scoring_status, scoring_method, utterance_segmental_score, words, phones
    return {
        "scoring_status": "unavailable",
        "scoring_method": "unavailable",
        "utterance_segmental_score": None,
        "words": [],
        "phones": [],
        "metadata": {
            "is_real_gop": False,
            "is_heuristic": False,
            "diagnostic_only": True,
            "note": SCORING_NOTE,
            **(metadata or {}),
        },
    }


def has_public_pronunciation_score(scoring_result: dict[str, Any]) -> bool:
    del scoring_result
    return False


def build_failed_scoring_result(scoring_method: str = "unavailable", error: str | None = None) -> dict[str, Any]:
    del scoring_method
    metadata: dict[str, Any] = {"is_real_gop": False, "is_heuristic": False, "note": SCORING_NOTE}
    if error:
        metadata["error"] = error
    return build_scoring_result(scoring_status="unavailable", scoring_method="unavailable", metadata=metadata)
