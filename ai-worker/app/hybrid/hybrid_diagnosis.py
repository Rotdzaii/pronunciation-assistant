from __future__ import annotations

from typing import Any


ERROR_LABELS = {
    "deletion": "bo am",
    "substitution": "thay the am",
    "addition": "them am",
    "unknown": "chua ro",
}


def _as_float(value: Any, default: float | None = None) -> float | None:
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


def _score_lookup(scoring_result: dict[str, Any]) -> dict[tuple[Any, Any, Any, Any], dict[str, Any]]:
    lookup = {}
    for index, phone_score in enumerate(scoring_result.get("phones") or []):
        key = (
            phone_score.get("index", index),
            phone_score.get("phone"),
            phone_score.get("word"),
            phone_score.get("start"),
        )
        lookup[key] = phone_score
    return lookup


def _matching_score(
    prediction: dict[str, Any],
    scoring_result: dict[str, Any],
    lookup: dict[tuple[Any, Any, Any, Any], dict[str, Any]],
) -> dict[str, Any] | None:
    score = lookup.get(_segment_key(prediction))
    if score:
        return score

    for phone_score in scoring_result.get("phones") or []:
        if prediction.get("phone") and prediction.get("phone") == phone_score.get("phone"):
            return phone_score
        if prediction.get("word") and prediction.get("word") == phone_score.get("word"):
            return phone_score
    return None


def determine_severity(
    phone_score: float | int | None,
    diagnosis_confidence: float | int | None,
    error_type: str | None,
) -> str:
    score = _as_float(phone_score)
    confidence = _as_float(diagnosis_confidence, 0.0) or 0.0
    has_error = error_type not in {None, "unknown"}

    if score is not None:
        if score < 60:
            return "high"
        if score < 75:
            return "medium" if has_error or confidence >= 0.55 else "low"
        return "medium" if has_error and confidence >= 0.75 else "low"

    if has_error and confidence >= 0.75:
        return "high"
    if has_error and confidence >= 0.55:
        return "medium"
    if has_error:
        return "low"
    return "unknown"


def should_display_feedback(
    severity: str,
    diagnosis_confidence: float | int | None,
    scoring_status: str | None,
) -> bool:
    confidence = _as_float(diagnosis_confidence, 0.0) or 0.0
    if severity in {"high", "medium"}:
        return True
    if severity == "low" and confidence >= 0.65:
        return True
    return scoring_status in {"success", "heuristic"} and confidence >= 0.55


def select_top_issues(
    segment_predictions: list[dict[str, Any]],
    scoring_result: dict[str, Any],
    max_issues: int = 3,
) -> list[dict[str, Any]]:
    lookup = _score_lookup(scoring_result)
    issues = []
    for prediction in segment_predictions:
        error_type = prediction.get("predicted_error_type")
        if error_type in {None, "unknown"}:
            continue

        phone_score = _matching_score(prediction, scoring_result, lookup)
        score_value = _as_float((phone_score or {}).get("phone_score"))
        confidence = _as_float(prediction.get("diagnosis_confidence"), 0.0) or 0.0
        severity = determine_severity(score_value, confidence, error_type)
        issues.append(
            {
                "phone": prediction.get("phone"),
                "word": prediction.get("word"),
                "start": prediction.get("start"),
                "end": prediction.get("end"),
                "predicted_error_type": error_type,
                "diagnosis_confidence": confidence,
                "phone_score": score_value,
                "severity": severity,
                "class_probabilities": prediction.get("class_probabilities") or {},
                "source": "cnn_attention_plus_scoring",
            }
        )

    severity_rank = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    issues.sort(
        key=lambda issue: (
            severity_rank.get(str(issue.get("severity")), 3),
            issue.get("phone_score") if issue.get("phone_score") is not None else 101.0,
            -float(issue.get("diagnosis_confidence") or 0.0),
        )
    )
    return issues[:max_issues]


def build_hybrid_feedback(top_issues: list[dict[str, Any]]) -> dict[str, Any] | None:
    display_issues = [
        issue for issue in top_issues if should_display_feedback(
            str(issue.get("severity") or "unknown"),
            issue.get("diagnosis_confidence"),
            "heuristic",
        )
    ]
    if not display_issues:
        return None

    primary = display_issues[0]
    phone = primary.get("phone") or primary.get("word") or "muc tieu"
    error_label = ERROR_LABELS.get(str(primary.get("predicted_error_type")), "loi phat am")
    return {
        "summary": f"He thong phat hien kha nang co loi {error_label} tai {phone}.",
        "tips": [
            "Luyen lai am hoac tu duoc danh dau voi toc do cham.",
            "So sanh ban thu am voi phat am mau va thu lai trong moi truong yen tinh.",
            "Luu y: muc do va vi tri loi hien tai phu thuoc vao alignment/scoring dang dung.",
        ],
    }


def _location_reliability(alignment_result: dict[str, Any]) -> str:
    metadata = alignment_result.get("metadata") or {}
    if metadata.get("is_forced_alignment") and alignment_result.get("method") == "mfa":
        return "forced_alignment"
    if metadata.get("is_fallback") or metadata.get("fallback_alignment") or str(alignment_result.get("method") or "").startswith("fallback"):
        return "limited_fallback_alignment"
    return "unknown"


def build_hybrid_diagnosis(
    alignment_result: dict[str, Any],
    segment_predictions: list[dict[str, Any]],
    scoring_result: dict[str, Any],
) -> dict[str, Any]:
    scoring_metadata = scoring_result.get("metadata") or {}
    top_issues = select_top_issues(segment_predictions, scoring_result)
    primary_issue = top_issues[0] if top_issues else None
    feedback = build_hybrid_feedback(top_issues)
    severity = str(primary_issue.get("severity")) if primary_issue else "unknown"
    primary_error_type = primary_issue.get("predicted_error_type") if primary_issue else "unknown"

    problem_phonemes = []
    for issue in top_issues:
        phone = issue.get("phone")
        if phone and phone not in problem_phonemes:
            problem_phonemes.append(phone)

    return {
        "hybrid_status": "success",
        "hybrid_method": "cnn_attention_plus_heuristic_scoring"
        if scoring_metadata.get("is_heuristic")
        else "cnn_attention_plus_scoring",
        "primary_error_type": primary_error_type,
        "severity": severity,
        "top_issues": top_issues,
        "problem_phonemes": problem_phonemes,
        "feedback": feedback,
        "location_reliability": _location_reliability(alignment_result),
        "scoring_is_heuristic": bool(scoring_metadata.get("is_heuristic")),
        "scoring_method": scoring_result.get("scoring_method"),
        "score_used_for_severity": scoring_result.get("utterance_segmental_score") is not None,
        "notes": [
            "Classifier confidence is diagnosis confidence, not pronunciation score.",
            "Heuristic scoring is not production GOP." if scoring_metadata.get("is_heuristic") else "",
            "Fallback alignment limits location reliability."
            if _location_reliability(alignment_result) == "limited_fallback_alignment"
            else "",
        ],
    }
