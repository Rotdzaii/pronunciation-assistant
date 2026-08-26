from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.contracts.ai_result_contract import build_failed_ai_result


LEGACY_FIELDS = {"job_id", "status", "score", "problem_phonemes", "feedback"}
SENSITIVE_KEYS = {
    "checkpoint_path",
    "textgrid_path",
    "mfa_output_dir",
    "mfa_temp_dir",
    "local_path",
    "audio_path",
    "debug_artifact_dir",
    "mfa_debug_dir",
    "audio_url",
    "stderr",
    "stdout",
    "command",
    "command_args",
    "mfa_process_diagnostic",
    "debug_metadata",
}
NON_PUBLIC_SCORING_KEYS = {
    "pronunciation_score_source",
    "scoring",
    "phone_score",
    "gop_score_raw",
    "gop_score_calibrated",
    "utterance_segmental_score",
    "severity",
    "scoring_method",
    "scoring_status",
    "scoring_is_heuristic",
}
MODEL_CAPABILITY = "error_type_classifier_only"
SENSITIVE_VALUE_MARKERS = (
    "c:\\",
    "/tmp/",
    "appdata\\local\\temp",
    ".textgrid",
    ".wav",
    ".webm",
    ".m4a",
    "x-amz-signature=",
    "token=",
    "sig=",
    "signature=",
    "authorization=",
)


def _sanitize_for_webhook(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS | NON_PUBLIC_SCORING_KEYS:
                continue
            sanitized[key] = _sanitize_for_webhook(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_webhook(item) for item in value]
    if isinstance(value, str):
        normalized = value.lower()
        if any(marker in normalized for marker in SENSITIVE_VALUE_MARKERS):
            return "[redacted-sensitive-value]"
    return value


def _find_sensitive_value_paths(value: Any, path: str = "") -> list[str]:
    issues: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            issues.extend(_find_sensitive_value_paths(item, next_path))
        return issues
    if isinstance(value, list):
        for index, item in enumerate(value):
            next_path = f"{path}[{index}]"
            issues.extend(_find_sensitive_value_paths(item, next_path))
        return issues
    if isinstance(value, str):
        normalized = value.lower()
        if any(marker in normalized for marker in SENSITIVE_VALUE_MARKERS):
            issues.append(path or "<root>")
    return issues


_CANONICAL_CONTRACT_FIELDS = (
    "reliability",
    "score_type",
    "display_pronunciation",
    "pronunciation_variant",
    "expected_phones",
    "pronunciation_source",
    "phone_results",
    "primary_feedback",
    "model_capability",
)


def _feedback_with_ai_result(ai_result: dict[str, Any]) -> dict[str, Any]:
    feedback = deepcopy(ai_result.get("feedback") if isinstance(ai_result.get("feedback"), dict) else {})
    feedback.setdefault("summary", "")
    feedback.setdefault("tips", [])
    feedback["ai_result"] = _sanitize_for_webhook(ai_result)
    for field in _CANONICAL_CONTRACT_FIELDS:
        if field in ai_result:
            feedback[field] = ai_result[field]
    return feedback


def _metadata(ai_result: dict[str, Any]) -> dict[str, Any]:
    metadata = ai_result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    safe_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in NON_PUBLIC_SCORING_KEYS
    }
    return {
        "alignment_used": metadata.get("alignment_used", False),
        "alignment_status": metadata.get("alignment_status"),
        "alignment_method": metadata.get("alignment_method"),
        "gop_used": False,
        "hybrid_used": False,
        **safe_metadata,
        "gop_used": False,
        "hybrid_used": False,
    }


def build_success_webhook_payload(job_id: str, ai_result: dict[str, Any]) -> dict[str, Any]:
    sanitized_ai_result = _sanitize_for_webhook(deepcopy(ai_result))
    # Older callers can still provide an old result shape. Phoenix v2 must not
    # publish a numerical score or a confirmed pronunciation diagnosis.
    sanitized_ai_result["model_capability"] = MODEL_CAPABILITY
    sanitized_ai_result["score"] = None
    sanitized_ai_result["score_type"] = "unavailable"
    diagnosis = sanitized_ai_result.get("diagnosis")
    if not isinstance(diagnosis, dict):
        diagnosis = {}
        sanitized_ai_result["diagnosis"] = diagnosis
    diagnosis["is_confirmed_error"] = False
    reliability = sanitized_ai_result.get("reliability")
    is_invalid = isinstance(reliability, dict) and reliability.get("status") == "invalid"
    if is_invalid:
        sanitized_ai_result["score"] = None
        sanitized_ai_result["score_type"] = "unavailable"
        sanitized_ai_result["problem_phonemes"] = []
        sanitized_ai_result["primary_feedback"] = None
    metadata = _metadata(sanitized_ai_result)
    return {
        "job_id": str(job_id),
        "status": sanitized_ai_result.get("status", "completed"),
        "model_capability": sanitized_ai_result.get("model_capability", MODEL_CAPABILITY),
        "score": None,
        "score_type": "unavailable",
        "score_note": sanitized_ai_result.get("score_note") or "Pronunciation score unavailable.",
        "problem_phonemes": list(sanitized_ai_result.get("problem_phonemes") or []),
        "feedback": _feedback_with_ai_result(sanitized_ai_result),
        "predicted_error_type": sanitized_ai_result.get("predicted_error_type"),
        "diagnosis": sanitized_ai_result.get("diagnosis") if isinstance(sanitized_ai_result.get("diagnosis"), dict) else {},
        "scorer": sanitized_ai_result.get("scorer") if isinstance(sanitized_ai_result.get("scorer"), dict) else {},
        "metadata": metadata,
        "ai_result": sanitized_ai_result,
    }


def build_failed_webhook_payload(
    job_id: str,
    error_message: str,
    ai_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sanitized_error_message = _sanitize_for_webhook(error_message)
    failed_result = ai_result or build_failed_ai_result(error=error_message)
    failed_result = _sanitize_for_webhook(deepcopy(failed_result))
    failed_result["status"] = "failed"
    failed_result["score"] = None
    failed_result["score_type"] = "unavailable"
    failed_result["problem_phonemes"] = []
    failed_result["primary_feedback"] = None
    metadata = _metadata(failed_result)
    metadata["error"] = sanitized_error_message

    feedback = _feedback_with_ai_result(failed_result)
    feedback.setdefault("summary", "AI worker khong tao duoc ket qua chan doan phat am.")
    feedback.setdefault("tips", [])

    return {
        "job_id": str(job_id),
        "status": "failed",
        "model_capability": failed_result.get("model_capability", MODEL_CAPABILITY),
        "score": None,
        "score_type": "unavailable",
        "problem_phonemes": [],
        "feedback": feedback,
        "error_message": sanitized_error_message,
        "predicted_error_type": failed_result.get("predicted_error_type"),
        "diagnosis": failed_result.get("diagnosis") if isinstance(failed_result.get("diagnosis"), dict) else {},
        "scorer": failed_result.get("scorer") if isinstance(failed_result.get("scorer"), dict) else {},
        "metadata": metadata,
        "ai_result": failed_result,
    }


def validate_webhook_payload(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not isinstance(payload, dict):
        return False, ["Webhook payload must be a dictionary."]

    for field_name in sorted(LEGACY_FIELDS):
        if field_name not in payload:
            issues.append(f"Missing legacy-compatible webhook field: {field_name}.")

    status = payload.get("status")
    if status not in {"completed", "failed"}:
        issues.append("Webhook payload status must be completed or failed.")
    if not str(payload.get("job_id") or "").strip():
        issues.append("Webhook payload job_id must exist.")
    if not isinstance(payload.get("problem_phonemes"), list):
        issues.append("Webhook payload problem_phonemes must be a list.")
    if not isinstance(payload.get("feedback"), dict):
        issues.append("Webhook payload feedback must be an object.")

    if status == "completed" and payload.get("score") is None and payload.get("score_type") != "unavailable":
        issues.append("Completed webhook payload score is required unless score_type is unavailable.")
    if status == "failed" and payload.get("score") is not None:
        issues.append("Failed webhook payload score must be null.")
    if payload.get("model_capability") != MODEL_CAPABILITY:
        issues.append("Webhook payload must declare error_type_classifier_only capability.")
    if payload.get("score") is not None or payload.get("score_type") != "unavailable":
        issues.append("Phoenix v2 classifier-only payload must not publish a pronunciation score.")

    diagnosis = payload.get("diagnosis") if isinstance(payload.get("diagnosis"), dict) else {}
    if diagnosis.get("confidence_note") is None and status == "completed":
        issues.append("Completed webhook payload should preserve diagnosis.confidence_note.")
    if status == "completed" and diagnosis.get("is_confirmed_error") is not False:
        issues.append("Completed classifier-only diagnosis must set is_confirmed_error to false.")

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if metadata.get("alignment_method") == "fallback_even_split":
        fallback_warning_fields = [
            str(metadata.get("alignment_note") or ""),
            str(metadata.get("location_reliability") or ""),
            str(metadata.get("fallback_reason") or ""),
        ]
        fallback_warning_text = " ".join(fallback_warning_fields).lower()
        if not any(term in fallback_warning_text for term in ("approximate", "fallback", "limited")):
            issues.append("Fallback alignment webhook metadata must mention approximate, fallback, or limited reliability.")
    if any(key in str(payload) for key in ("pronunciation_score_source", "phone_score", "gop_score")):
        issues.append("Webhook payload must not expose heuristic or pronunciation-score source fields.")

    sensitive_value_paths = _find_sensitive_value_paths(payload)
    if sensitive_value_paths:
        issues.append(
            "Webhook payload contains sensitive local-path or signed-token markers at: "
            + ", ".join(sensitive_value_paths)
            + "."
        )

    return not issues, issues
