from __future__ import annotations

import hashlib
from typing import Any


LOW_CONFIDENCE_WARNING = (
    "Độ tin cậy của kết quả chưa cao, bạn nên ghi âm lại trong môi trường yên tĩnh hơn."
)


def _unit_value(seed: str, salt: str) -> float:
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def score_pronunciation(job: dict[str, Any], confidence_threshold: float) -> dict[str, Any]:
    """Return deterministic mock pronunciation feedback for a practice job."""
    job_id = str(job.get("job_id") or "")
    target_word = str(job.get("target_word") or "").strip()
    seed = job_id or target_word or "practice-job"

    length_factor = min(len(target_word), 14) * 1.3
    score_variation = _unit_value(seed, "score") * 18
    score = round(min(96.0, 68.0 + length_factor + score_variation), 1)
    confidence = round(0.52 + (_unit_value(seed, "confidence") * 0.43), 2)
    is_reliable = confidence >= confidence_threshold

    problem_phonemes = []
    if score < 88 or not is_reliable:
        problem_phonemes.append(
            {
                "phoneme": "/t/",
                "type": "final_sound",
                "severity": "medium" if score >= 75 else "high",
                "confidence": round(max(0.5, confidence - 0.04), 2),
                "tip": "Bạn cần bật rõ âm cuối /t/ hơn.",
            }
        )

    tips = [
        "Nói chậm hơn một chút và giữ nhịp ổn định.",
        "Nghe lại bản ghi rồi lặp lại từ mục tiêu thêm một lần.",
    ]
    if not is_reliable:
        tips.insert(0, LOW_CONFIDENCE_WARNING)

    return {
        "status": "completed",
        "score": score,
        "problem_phonemes": problem_phonemes,
        "feedback": {
            "summary": (
                "Kết quả mô phỏng cho thấy phát âm khá rõ."
                if is_reliable
                else LOW_CONFIDENCE_WARNING
            ),
            "target_word": target_word,
            "tips": tips,
            "model_confidence": {
                "value": confidence,
                "threshold": confidence_threshold,
                "level": _confidence_level(confidence),
                "is_reliable": is_reliable,
            },
            "scorer": "mock",
        },
    }
