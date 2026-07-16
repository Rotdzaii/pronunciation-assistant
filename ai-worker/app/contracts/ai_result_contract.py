from __future__ import annotations

from typing import Any


ERROR_TYPES = {"addition", "deletion", "substitution", "unknown"}
MODEL_CAPABILITY = "error_type_classifier_only"
CONFIDENCE_NOTE = "Classifier confidence only compares the three error-type classes; it is not pronunciation correctness."
DEFAULT_CLASS_PROBABILITIES = {
    "addition": 0.0,
    "deletion": 0.0,
    "substitution": 0.0,
}
DEFAULT_SCORER = {
    "name": "cnn_attention_phone_error_classifier",
    "type": "phone_error_classifier",
    "version": "phoenix-v2-error-type-classifier",
}
DEFAULT_METADATA = {
    "model_output_is_scoring": False,
    "alignment_used": False,
    "gop_used": False,
    "hybrid_used": False,
}


FEEDBACK_BY_ERROR_TYPE = {
    "deletion": {
        "summary": "Mô hình nghi ngờ loại lỗi bỏ âm tại vị trí được căn chỉnh.",
        "tips": [
            "Luyện phát âm rõ âm bị bỏ hoặc âm cuối của từ.",
            "Đọc chậm hơn và chú ý kết thúc âm.",
            "Nghe lại phát âm mẫu rồi thử âm lại.",
        ],
    },
    "substitution": {
        "summary": "Mô hình nghi ngờ loại lỗi thay thế âm tại vị trí được căn chỉnh.",
        "tips": [
            "So sánh âm mục tiêu với âm bạn vừa phát âm.",
            "Luyện khẩu hình và vị trí lưỡi cho âm mục tiêu.",
            "Đọc từng âm chậm trước khi đọc cả từ.",
        ],
    },
    "addition": {
        "summary": "Mô hình nghi ngờ loại lỗi thêm âm tại vị trí được căn chỉnh.",
        "tips": [
            "Tránh chèn thêm nguyên âm ngắn giữa các phụ âm.",
            "Luyện đọc liền mạch các cụm phụ âm.",
            "Nghe mẫu và lặp lại với tốc độ chậm.",
        ],
    },
    "unknown": {
        "summary": "Mô hình chưa đủ dữ liệu để đưa ra loại lỗi dự đoán.",
        "tips": [
            "Hãy thử thu âm lại trong môi trường yên tĩnh hơn.",
            "Đọc rõ ràng và giữ khoảng cách micro ổn định.",
        ],
    },
}


def _normalize_error_type(error_type: str | None) -> str | None:
    if error_type is None:
        return None
    normalized = error_type.strip().lower()
    return normalized if normalized in ERROR_TYPES else "unknown"


def _normalize_class_probabilities(class_probabilities: dict[str, Any] | None) -> dict[str, float]:
    probabilities = dict(DEFAULT_CLASS_PROBABILITIES)
    for label in probabilities:
        try:
            probabilities[label] = float((class_probabilities or {}).get(label, 0.0))
        except (TypeError, ValueError):
            probabilities[label] = 0.0
    return probabilities


def map_error_type_to_feedback(error_type: str | None) -> dict[str, Any]:
    normalized = _normalize_error_type(error_type) or "unknown"
    feedback = FEEDBACK_BY_ERROR_TYPE.get(normalized, FEEDBACK_BY_ERROR_TYPE["unknown"])
    return {"summary": feedback["summary"], "tips": list(feedback["tips"])}


def estimate_demo_score(error_type: str | None, diagnosis_confidence: float | None) -> dict[str, Any]:
    """Legacy-compatible helper that never derives a public score."""
    return {
        "score": None,
        "is_demo_score": False,
        "score_note": "Pronunciation score unavailable: Phoenix v2 is an error-type classifier only.",
    }


def build_ai_result(
    *,
    score: int | float | None = None,
    problem_phonemes: list[str] | None = None,
    predicted_error_type: str | None = None,
    class_probabilities: dict[str, Any] | None = None,
    diagnosis_confidence: float | None = None,
    feedback: dict[str, Any] | None = None,
    scorer: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    scoring: dict[str, Any] | None = None,
    score_note: str | None = None,
    pronunciation_score_source: str | None = None,
    diagnosis_extra: dict[str, Any] | None = None,
    score_type: str = "unavailable",
) -> dict[str, Any]:
    """Build Phoenix v2's classifier-only public contract.

    Legacy arguments are accepted so older callers do not crash, but numerical
    and heuristic score values are deliberately ignored.
    """
    del score, scoring, pronunciation_score_source, score_type
    normalized_error_type = _normalize_error_type(predicted_error_type)
    suspected_problem_phone = next((str(phone) for phone in problem_phonemes or [] if str(phone).strip()), None)
    result_feedback = feedback or map_error_type_to_feedback(normalized_error_type)
    result_metadata = {**DEFAULT_METADATA, **(metadata or {})}
    for non_public_key in (
        "pronunciation_score_source",
        "scoring",
        "phone_score",
        "gop_score_raw",
        "gop_score_calibrated",
        "utterance_segmental_score",
        "severity",
    ):
        result_metadata.pop(non_public_key, None)
    result_metadata.update({
        "model_capability": MODEL_CAPABILITY,
        "public_pronunciation_score_available": False,
        "score_availability_reason": "error_type_classifier_only",
        "gop_used": False,
        "hybrid_used": False,
    })

    diagnosis = {
        "predicted_error_type": normalized_error_type,
        "suspected_problem_phone": suspected_problem_phone,
        "class_probabilities": _normalize_class_probabilities(class_probabilities),
        "diagnosis_confidence": diagnosis_confidence,
        "confidence_note": CONFIDENCE_NOTE,
        "is_confirmed_error": False,
    }
    if diagnosis_extra:
        diagnosis.update(diagnosis_extra)
    diagnosis["is_confirmed_error"] = False

    return {
        "status": "completed",
        "model_capability": MODEL_CAPABILITY,
        "score": None,
        "score_type": "unavailable",
        "score_note": score_note or "Pronunciation score unavailable: Phoenix v2 classifies possible error types only.",
        # Retained as a legacy storage field. Its entries are suspected, never confirmed.
        "problem_phonemes": list(problem_phonemes or []),
        "predicted_error_type": normalized_error_type,
        "diagnosis": diagnosis,
        "feedback": {
            "summary": str(result_feedback.get("summary") or ""),
            "tips": list(result_feedback.get("tips") or []),
        },
        "scorer": {**DEFAULT_SCORER, **(scorer or {})},
        "metadata": result_metadata,
    }


def build_failed_ai_result(
    *,
    error: str,
    scorer: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_metadata = {**DEFAULT_METADATA, **(metadata or {})}
    result_metadata.update({"error": error, "model_capability": MODEL_CAPABILITY, "gop_used": False, "hybrid_used": False})
    return {
        "status": "failed",
        "model_capability": MODEL_CAPABILITY,
        "score": None,
        "score_type": "unavailable",
        "score_note": "Pronunciation score unavailable.",
        "problem_phonemes": [],
        "predicted_error_type": None,
        "diagnosis": {
            "predicted_error_type": None,
            "suspected_problem_phone": None,
            "class_probabilities": dict(DEFAULT_CLASS_PROBABILITIES),
            "diagnosis_confidence": None,
            "confidence_note": CONFIDENCE_NOTE,
            "is_confirmed_error": False,
        },
        "feedback": {
            "summary": "AI worker không tạo được kết quả phân tích loại lỗi.",
            "tips": ["Hãy thử thu âm lại trong môi trường yên tĩnh hơn."],
        },
        "scorer": {**DEFAULT_SCORER, **(scorer or {})},
        "metadata": result_metadata,
    }
