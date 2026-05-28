from __future__ import annotations

from typing import Any


REQUIRED_TOP_LEVEL_FIELDS = {
    "status",
    "score",
    "problem_phonemes",
    "predicted_error_type",
    "diagnosis",
    "feedback",
    "scorer",
    "metadata",
}


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _contains_any(value: Any, terms: tuple[str, ...]) -> bool:
    text = str(value or "").lower()
    return any(term in text for term in terms)


def _append_missing_type_issues(result: dict[str, Any], issues: list[str]) -> None:
    for field_name in sorted(REQUIRED_TOP_LEVEL_FIELDS):
        if field_name not in result:
            issues.append(f"Missing required top-level field: {field_name}.")

    if "problem_phonemes" in result and not isinstance(result.get("problem_phonemes"), list):
        issues.append("problem_phonemes must be a list.")
    if "diagnosis" in result and not _is_mapping(result.get("diagnosis")):
        issues.append("diagnosis must be an object.")
    if "feedback" in result and not _is_mapping(result.get("feedback")):
        issues.append("feedback must be an object.")
    if "scorer" in result and not _is_mapping(result.get("scorer")):
        issues.append("scorer must be an object.")
    if "metadata" in result and not _is_mapping(result.get("metadata")):
        issues.append("metadata must be an object.")


def _append_completed_issues(result: dict[str, Any], issues: list[str]) -> None:
    feedback = result.get("feedback") if _is_mapping(result.get("feedback")) else {}
    diagnosis = result.get("diagnosis") if _is_mapping(result.get("diagnosis")) else {}
    scorer = result.get("scorer") if _is_mapping(result.get("scorer")) else {}
    metadata = result.get("metadata") if _is_mapping(result.get("metadata")) else {}

    if not str(feedback.get("summary") or "").strip():
        issues.append("Completed result feedback.summary must exist.")
    if not isinstance(feedback.get("tips"), list):
        issues.append("Completed result feedback.tips must be a list.")
    if not str(diagnosis.get("confidence_note") or "").strip():
        issues.append("Completed result diagnosis.confidence_note must exist.")
    if "diagnosis_score" in diagnosis or "score" in diagnosis:
        issues.append("diagnosis must not contain a score field; diagnosis_confidence is not pronunciation score.")
    if not str(scorer.get("name") or "").strip():
        issues.append("Completed result scorer.name must exist.")
    if "model_output_is_scoring" not in metadata:
        issues.append("Completed result metadata.model_output_is_scoring must exist.")


def _append_failed_issues(result: dict[str, Any], issues: list[str]) -> None:
    metadata = result.get("metadata") if _is_mapping(result.get("metadata")) else {}
    feedback = result.get("feedback") if _is_mapping(result.get("feedback")) else {}

    if result.get("score") is not None:
        issues.append("Failed result score should be null.")
    if "error" not in metadata and not str(feedback.get("summary") or "").strip():
        issues.append("Failed result should include metadata.error or feedback.summary.")


def _append_safety_issues(result: dict[str, Any], issues: list[str]) -> None:
    metadata = result.get("metadata") if _is_mapping(result.get("metadata")) else {}
    scoring = result.get("scoring") if _is_mapping(result.get("scoring")) else {}
    scoring_metadata = scoring.get("metadata") if _is_mapping(scoring.get("metadata")) else {}
    diagnosis = result.get("diagnosis") if _is_mapping(result.get("diagnosis")) else {}

    scoring_method = metadata.get("scoring_method") or scoring.get("scoring_method")
    score_source = result.get("pronunciation_score_source") or metadata.get("pronunciation_score_source")
    score_note = result.get("score_note") or metadata.get("score_note") or scoring_metadata.get("note")
    alignment_method = metadata.get("alignment_method")
    alignment_note = metadata.get("alignment_note") or metadata.get("location_reliability") or ""

    if metadata.get("scoring_is_heuristic") is True and not _contains_any(score_note, ("heuristic", "demo")):
        issues.append("Heuristic scoring requires score_note to mention heuristic or demo.")

    if metadata.get("gop_used") is False and scoring_method == "heuristic_gop":
        combined_claims = " ".join(
            str(value or "")
            for value in [
                metadata.get("score_note"),
                result.get("score_note"),
                scoring_metadata.get("note"),
                metadata.get("limitation"),
            ]
        ).lower()
        if "real gop" in combined_claims and "not real gop" not in combined_claims:
            issues.append("heuristic_gop result must not claim real GOP when metadata.gop_used is false.")

    if alignment_method == "fallback_even_split" and not _contains_any(
        alignment_note,
        ("approximate", "fallback", "limited"),
    ):
        issues.append("Fallback alignment metadata/location note must mention approximate, fallback, or limited reliability.")

    if score_source == "classifier_confidence":
        issues.append("pronunciation_score_source must not be classifier_confidence.")
    if metadata.get("score_source") == "classifier_confidence":
        issues.append("metadata.score_source must not be classifier_confidence.")

    if diagnosis.get("diagnosis_confidence") is not None and score_source == "diagnosis_confidence":
        issues.append("diagnosis_confidence must not be used as pronunciation_score_source.")


def validate_ai_result(result: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not isinstance(result, dict):
        return False, ["AI result must be a dictionary."]

    _append_missing_type_issues(result, issues)
    status = result.get("status")
    if status not in {"completed", "failed"}:
        issues.append("status must be either completed or failed.")
    elif status == "completed":
        _append_completed_issues(result, issues)
    elif status == "failed":
        _append_failed_issues(result, issues)

    _append_safety_issues(result, issues)
    return not issues, issues


def assert_valid_ai_result(result: dict[str, Any]) -> None:
    is_valid, issues = validate_ai_result(result)
    if not is_valid:
        raise ValueError("Invalid AI result: " + " | ".join(issues))
