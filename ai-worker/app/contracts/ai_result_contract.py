from __future__ import annotations

from typing import Any


ERROR_TYPES = {"addition", "deletion", "substitution", "unknown"}
CONFIDENCE_NOTE = "Classifier confidence, not pronunciation correctness."
DEFAULT_CLASS_PROBABILITIES = {
    "addition": 0.0,
    "deletion": 0.0,
    "substitution": 0.0,
}
DEFAULT_SCORER = {
    "name": "cnn_attention_phone_error_classifier",
    "type": "phone_error_classifier",
    "version": "demo-contract-v1",
}
DEFAULT_METADATA = {
    "model_output_is_scoring": False,
    "alignment_used": False,
    "gop_used": False,
    "hybrid_used": False,
}


FEEDBACK_BY_ERROR_TYPE = {
    "deletion": {
        "summary": "He thong phat hien kha nang co loi bo am trong phat am.",
        "tips": [
            "Luyen phat am ro am bi bo hoac am cuoi cua tu.",
            "Doc cham hon va chu y ket thuc am.",
            "Nghe lai phat am mau roi thu am lai.",
        ],
    },
    "substitution": {
        "summary": "He thong phat hien kha nang co loi thay the am.",
        "tips": [
            "So sanh am muc tieu voi am ban vua phat am.",
            "Luyen khau hinh va vi tri luoi cho am muc tieu.",
            "Doc tung am cham truoc khi doc ca tu.",
        ],
    },
    "addition": {
        "summary": "He thong phat hien kha nang co loi them am.",
        "tips": [
            "Tranh chen them nguyen am ngan giua cac phu am.",
            "Luyen doc lien mach cac cum phu am.",
            "Nghe mau va lap lai voi toc do cham.",
        ],
    },
    "unknown": {
        "summary": "He thong chua du chac chan de xac dinh loai loi phat am.",
        "tips": [
            "Hay thu thu am lai trong moi truong yen tinh hon.",
            "Doc ro rang va giu khoang cach micro on dinh.",
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
    return {
        "summary": feedback["summary"],
        "tips": list(feedback["tips"]),
    }


def estimate_demo_score(error_type: str | None, diagnosis_confidence: float | None) -> dict[str, Any]:
    """Return a clearly marked placeholder score for demos only.

    Classifier confidence is not pronunciation correctness. This helper exists only
    until alignment, GOP/CaGOP, or hybrid scoring provides a real score.
    """
    normalized = _normalize_error_type(error_type) or "unknown"
    try:
        confidence = float(diagnosis_confidence) if diagnosis_confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    base_scores = {
        "deletion": 72.0,
        "substitution": 75.0,
        "addition": 74.0,
        "unknown": None,
    }
    score = base_scores[normalized]
    if score is not None and confidence is not None:
        score = round(max(55.0, min(88.0, score + ((0.5 - confidence) * 10))), 1)

    return {
        "score": score,
        "is_demo_score": True,
        "score_note": (
            "Demo heuristic only. This is not real pronunciation scoring and must "
            "not be derived from classifier confidence in production."
        ),
    }


def build_ai_result(
    *,
    score: int | float | None,
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
) -> dict[str, Any]:
    normalized_error_type = _normalize_error_type(predicted_error_type)
    result_feedback = feedback or map_error_type_to_feedback(normalized_error_type)
    result_metadata = {**DEFAULT_METADATA, **(metadata or {})}
    resolved_score_note = score_note if score_note is not None else result_metadata.get("score_note")
    resolved_score_source = (
        pronunciation_score_source
        if pronunciation_score_source is not None
        else result_metadata.get("pronunciation_score_source")
    )

    result = {
        "status": "completed",
        "score": score,
        "score_note": resolved_score_note,
        "pronunciation_score_source": resolved_score_source,
        "problem_phonemes": list(problem_phonemes or []),
        "predicted_error_type": normalized_error_type,
        "diagnosis": {
            "primary_error_type": normalized_error_type,
            "class_probabilities": _normalize_class_probabilities(class_probabilities),
            "diagnosis_confidence": diagnosis_confidence,
            "confidence_note": CONFIDENCE_NOTE,
        },
        "feedback": {
            "summary": str(result_feedback.get("summary") or ""),
            "tips": list(result_feedback.get("tips") or []),
        },
        "scorer": {**DEFAULT_SCORER, **(scorer or {})},
        "metadata": result_metadata,
    }
    if scoring is not None:
        result["scoring"] = scoring
    return result


def build_failed_ai_result(
    *,
    error: str,
    scorer: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_metadata = {**DEFAULT_METADATA, **(metadata or {})}
    result_metadata["error"] = error

    return {
        "status": "failed",
        "score": None,
        "score_note": None,
        "pronunciation_score_source": None,
        "problem_phonemes": [],
        "predicted_error_type": None,
        "diagnosis": {
            "primary_error_type": None,
            "class_probabilities": dict(DEFAULT_CLASS_PROBABILITIES),
            "diagnosis_confidence": None,
            "confidence_note": CONFIDENCE_NOTE,
        },
        "feedback": {
            "summary": "AI worker khong tao duoc ket qua chan doan phat am.",
            "tips": [
                "Hay thu thu am lai trong moi truong yen tinh hon.",
                "Neu loi tiep tuc xay ra, vui long thu lai sau.",
            ],
        },
        "scorer": {**DEFAULT_SCORER, **(scorer or {})},
        "metadata": result_metadata,
    }
