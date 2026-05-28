from __future__ import annotations

import json
import sys
from pathlib import Path


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.contracts.ai_result_contract import build_ai_result, build_failed_ai_result  # noqa: E402
from app.contracts.ai_result_validator import validate_ai_result  # noqa: E402
from app.contracts.alignment_contract import build_alignment_result, build_alignment_segment  # noqa: E402
from app.hybrid.hybrid_diagnosis import build_hybrid_diagnosis  # noqa: E402
from app.scoring.scoring_service import score_pronunciation_segments  # noqa: E402


def _sample_alignment() -> dict[str, object]:
    phones = [
        build_alignment_segment(index=0, segment_type="phone", phone="EH", word="example", start=0.0, end=0.1),
        build_alignment_segment(index=1, segment_type="phone", phone="G", word="example", start=0.1, end=0.24),
    ]
    return build_alignment_result(
        segments=phones,
        words=[build_alignment_segment(index=0, segment_type="word", word="example", start=0.0, end=0.24)],
        metadata={
            "is_forced_alignment": False,
            "is_fallback": True,
            "fallback_alignment": True,
        },
    )


def _sample_predictions() -> list[dict[str, object]]:
    return [
        {
            "index": 0,
            "phone": "EH",
            "word": "example",
            "start": 0.0,
            "end": 0.1,
            "predicted_error_type": "deletion",
            "diagnosis_confidence": 0.84,
            "class_probabilities": {"addition": 0.05, "deletion": 0.84, "substitution": 0.11},
        },
        {
            "index": 1,
            "phone": "G",
            "word": "example",
            "start": 0.1,
            "end": 0.24,
            "predicted_error_type": "substitution",
            "diagnosis_confidence": 0.67,
            "class_probabilities": {"addition": 0.12, "deletion": 0.21, "substitution": 0.67},
        },
    ]


def build_completed_sample() -> dict[str, object]:
    alignment_result = _sample_alignment()
    predictions = _sample_predictions()
    scoring_result = score_pronunciation_segments(alignment_result, predictions)
    hybrid_result = build_hybrid_diagnosis(alignment_result, predictions, scoring_result)
    primary_issue = hybrid_result["top_issues"][0]

    return build_ai_result(
        score=scoring_result["utterance_segmental_score"],
        score_note="Heuristic/demo score, not production GOP.",
        pronunciation_score_source="heuristic_gop",
        problem_phonemes=hybrid_result["problem_phonemes"],
        predicted_error_type=hybrid_result["primary_error_type"],
        class_probabilities=primary_issue["class_probabilities"],
        diagnosis_confidence=primary_issue["diagnosis_confidence"],
        feedback=hybrid_result["feedback"],
        scorer={
            "name": "cnn_attention",
            "type": "phone_error_classifier",
            "version": "cnn_attention_selected_baseline",
        },
        scoring=scoring_result,
        diagnosis_extra={
            "top_issues": hybrid_result["top_issues"],
            "severity": hybrid_result["severity"],
        },
        metadata={
            "model_output_is_scoring": False,
            "alignment_used": True,
            "alignment_method": "fallback_even_split",
            "alignment_note": "Fallback alignment is approximate and location reliability is limited.",
            "gop_used": False,
            "hybrid_used": True,
            "hybrid_method": hybrid_result["hybrid_method"],
            "location_reliability": hybrid_result["location_reliability"],
            "scoring_method": scoring_result["scoring_method"],
            "scoring_is_heuristic": True,
            "score_note": "Heuristic/demo score, not production GOP.",
        },
    )


def _print_validation(label: str, result: dict[str, object]) -> None:
    is_valid, issues = validate_ai_result(result)
    print(f"=== {label} ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"validation_valid={is_valid}")
    if issues:
        print("validation_issues=")
        for issue in issues:
            print(f"- {issue}")


def main() -> int:
    print("Note: confidence is not pronunciation score.")
    print("Note: fallback alignment is approximate.")
    print("Note: heuristic_gop is not real GOP.")

    _print_validation("completed_sample", build_completed_sample())
    _print_validation(
        "failed_sample",
        build_failed_ai_result(
            error="demo failure",
            scorer={"name": "cnn_attention", "type": "phone_error_classifier"},
            metadata={"model_output_is_scoring": False},
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
