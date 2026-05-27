from __future__ import annotations

import json
import sys
from pathlib import Path


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.contracts.ai_result_contract import (  # noqa: E402
    build_ai_result,
    build_failed_ai_result,
    estimate_demo_score,
)


SAMPLE_PROBABILITIES = {
    "deletion": {"addition": 0.12, "deletion": 0.73, "substitution": 0.15},
    "substitution": {"addition": 0.08, "deletion": 0.21, "substitution": 0.71},
    "addition": {"addition": 0.66, "deletion": 0.13, "substitution": 0.21},
    "unknown": {"addition": 0.31, "deletion": 0.34, "substitution": 0.35},
}


def build_demo_result(error_type: str) -> dict[str, object]:
    diagnosis_confidence = max(SAMPLE_PROBABILITIES[error_type].values())
    demo_score = estimate_demo_score(error_type, diagnosis_confidence)
    result = build_ai_result(
        score=demo_score["score"],
        problem_phonemes=["/t/"] if error_type != "unknown" else [],
        predicted_error_type=error_type,
        class_probabilities=SAMPLE_PROBABILITIES[error_type],
        diagnosis_confidence=diagnosis_confidence,
        metadata={
            "is_demo_score": demo_score["is_demo_score"],
            "score_note": demo_score["score_note"],
        },
    )
    return result


def main() -> int:
    for error_type in ["deletion", "substitution", "addition", "unknown"]:
        print(f"=== {error_type} ===")
        print(json.dumps(build_demo_result(error_type), indent=2, ensure_ascii=False))

    print("=== failed ===")
    print(
        json.dumps(
            build_failed_ai_result(error="demo failure"),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
