from __future__ import annotations

from typing import Any


SCORING_NOTE = "Heuristic score is not a production GOP score."
SEVERITIES = {"low", "medium", "high", "unknown"}


def clamp_score(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, min(100.0, float(value))), 1)


def severity_from_score(score: float | int | None) -> str:
    if score is None:
        return "unknown"
    if float(score) < 60:
        return "high"
    if float(score) < 75:
        return "medium"
    return "low"


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
    source: str = "heuristic",
) -> dict[str, Any]:
    normalized_score = clamp_score(phone_score)
    normalized_severity = severity if severity in SEVERITIES else severity_from_score(normalized_score)
    return {
        "phone": phone,
        "word": word,
        "start": round(float(start), 3) if start is not None else None,
        "end": round(float(end), 3) if end is not None else None,
        "phone_score": normalized_score,
        "gop_score_raw": float(gop_score_raw) if gop_score_raw is not None else None,
        "gop_score_calibrated": clamp_score(gop_score_calibrated),
        "duration_mismatch": round(float(duration_mismatch), 3) if duration_mismatch is not None else None,
        "severity": normalized_severity,
        "source": source,
    }


def build_word_score(
    *,
    word: str | None,
    start: float | None = None,
    end: float | None = None,
    score: float | int | None = None,
    phones: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "word": word,
        "start": round(float(start), 3) if start is not None else None,
        "end": round(float(end), 3) if end is not None else None,
        "score": clamp_score(score),
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
    result_metadata = {
        "is_real_gop": False,
        "is_heuristic": scoring_method == "heuristic_gop",
        "note": SCORING_NOTE if scoring_method == "heuristic_gop" else "",
        **(metadata or {}),
    }
    return {
        "scoring_status": scoring_status,
        "scoring_method": scoring_method,
        "utterance_segmental_score": clamp_score(utterance_segmental_score),
        "words": list(words or []),
        "phones": list(phones or []),
        "metadata": result_metadata,
    }


def build_failed_scoring_result(scoring_method: str = "none", error: str | None = None) -> dict[str, Any]:
    metadata = {
        "is_real_gop": False,
        "is_heuristic": False,
        "note": "No pronunciation segmental score was produced.",
    }
    if error:
        metadata["error"] = error
    return build_scoring_result(
        scoring_status="failed",
        scoring_method=scoring_method,
        utterance_segmental_score=None,
        words=[],
        phones=[],
        metadata=metadata,
    )
