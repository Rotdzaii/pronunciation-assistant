from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.contracts.scoring_contract import (
    SCORING_NOTE,
    build_phone_score,
    build_scoring_result,
    build_word_score,
    clamp_score,
)


ERROR_PENALTIES = {
    "deletion": 34.0,
    "substitution": 28.0,
    "addition": 24.0,
    "unknown": 8.0,
    None: 6.0,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _segment_key(segment: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        segment.get("index"),
        segment.get("phone"),
        segment.get("word"),
        segment.get("start"),
    )


def _prediction_by_segment(segment_predictions: list[dict[str, Any]] | None) -> dict[tuple[Any, Any, Any, Any], dict[str, Any]]:
    return {_segment_key(prediction): prediction for prediction in (segment_predictions or [])}


def _fallback_prediction(segment: dict[str, Any], predictions: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not predictions:
        return None
    for prediction in predictions:
        if prediction.get("phone") and prediction.get("phone") == segment.get("phone"):
            return prediction
        if prediction.get("word") and prediction.get("word") == segment.get("word"):
            return prediction
    return None


def _duration_mismatch(segment: dict[str, Any]) -> float | None:
    start = segment.get("start")
    end = segment.get("end")
    if start is None or end is None:
        return None
    duration = max(_as_float(end) - _as_float(start), 0.0)
    target_duration = 0.12 if segment.get("phone") else 0.45
    return min(abs(duration - target_duration) / target_duration, 1.0)


def _heuristic_phone_score(
    segment: dict[str, Any],
    prediction: dict[str, Any] | None,
    *,
    fallback_alignment: bool,
) -> tuple[float, float | None]:
    error_type = (prediction or {}).get("predicted_error_type")
    confidence = _as_float((prediction or {}).get("diagnosis_confidence"), 0.0)
    duration_mismatch = _duration_mismatch(segment)

    if error_type not in ERROR_PENALTIES:
        error_type = "unknown"

    confidence_weight = confidence if confidence >= 0.45 else 0.35
    score = 88.0 - (ERROR_PENALTIES[error_type] * confidence_weight)
    if duration_mismatch is not None:
        score -= duration_mismatch * 8.0
    if fallback_alignment:
        score -= 3.0

    return clamp_score(score) or 0.0, duration_mismatch


def _word_scores(phone_scores: list[dict[str, Any]], alignment_result: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for phone_score in phone_scores:
        word = phone_score.get("word")
        if word:
            grouped[str(word)].append(phone_score)

    words = []
    for word_segment in alignment_result.get("words") or []:
        word = word_segment.get("word")
        phones = grouped.get(str(word), [])
        score_values = [phone["phone_score"] for phone in phones if phone.get("phone_score") is not None]
        word_score = sum(score_values) / len(score_values) if score_values else None
        words.append(
            build_word_score(
                word=word,
                start=word_segment.get("start"),
                end=word_segment.get("end"),
                score=word_score,
                phones=phones,
            )
        )

    if words:
        return words

    word_like_scores = [score for score in phone_scores if score.get("word") and not score.get("phone")]
    return [
        build_word_score(
            word=score.get("word"),
            start=score.get("start"),
            end=score.get("end"),
            score=score.get("phone_score"),
            phones=[],
        )
        for score in word_like_scores
    ]


def score_heuristic_gop(
    alignment_result: dict[str, Any],
    segment_predictions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    alignment_metadata = alignment_result.get("metadata") or {}
    fallback_alignment = bool(
        alignment_metadata.get("is_fallback")
        or alignment_metadata.get("fallback_alignment")
        or str(alignment_result.get("method") or "").startswith("fallback")
    )
    segments = alignment_result.get("phones") or alignment_result.get("segments") or []
    prediction_lookup = _prediction_by_segment(segment_predictions)

    phone_scores = []
    for segment in segments:
        prediction = prediction_lookup.get(_segment_key(segment)) or _fallback_prediction(segment, segment_predictions)
        phone_score, duration_mismatch = _heuristic_phone_score(
            segment,
            prediction,
            fallback_alignment=fallback_alignment,
        )
        phone_scores.append(
            build_phone_score(
                phone=segment.get("phone"),
                word=segment.get("word"),
                start=segment.get("start"),
                end=segment.get("end"),
                phone_score=phone_score,
                gop_score_raw=None,
                gop_score_calibrated=phone_score,
                duration_mismatch=duration_mismatch,
                source="heuristic",
            )
        )

    score_values = [score["phone_score"] for score in phone_scores if score.get("phone_score") is not None]
    utterance_score = sum(score_values) / len(score_values) if score_values else None
    metadata_note = SCORING_NOTE
    if fallback_alignment:
        metadata_note += " Fallback alignment limits score reliability."

    return build_scoring_result(
        scoring_status="heuristic",
        scoring_method="heuristic_gop",
        utterance_segmental_score=utterance_score,
        words=_word_scores(phone_scores, alignment_result),
        phones=phone_scores,
        metadata={
            "is_real_gop": False,
            "is_heuristic": True,
            "note": metadata_note,
            "alignment_method": alignment_result.get("method"),
            "alignment_status": alignment_result.get("status"),
            "fallback_alignment": fallback_alignment,
            "segment_predictions_used": bool(segment_predictions),
        },
    )
