from __future__ import annotations

import json
import sys
from pathlib import Path


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.contracts.alignment_contract import build_alignment_result, build_alignment_segment  # noqa: E402
from app.hybrid.hybrid_diagnosis import build_hybrid_diagnosis  # noqa: E402
from app.scoring.scoring_service import score_pronunciation_segments  # noqa: E402


def _sample_alignment() -> dict[str, object]:
    phones = [
        build_alignment_segment(index=0, segment_type="phone", phone="EH", word="example", start=0.0, end=0.1),
        build_alignment_segment(index=1, segment_type="phone", phone="G", word="example", start=0.1, end=0.24),
        build_alignment_segment(index=2, segment_type="phone", phone="Z", word="example", start=0.24, end=0.42),
    ]
    return build_alignment_result(
        segments=phones,
        words=[build_alignment_segment(index=0, segment_type="word", word="example", start=0.0, end=0.42)],
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
        {
            "index": 2,
            "phone": "Z",
            "word": "example",
            "start": 0.24,
            "end": 0.42,
            "predicted_error_type": "unknown",
            "diagnosis_confidence": 0.38,
            "class_probabilities": {"addition": 0.31, "deletion": 0.31, "substitution": 0.38},
        },
    ]


def main() -> int:
    print("Note: fallback alignment is approximate and not real forced alignment.")
    print("Note: heuristic_gop is not real GOP.")
    print("Note: classifier confidence is not pronunciation score.")

    alignment_result = _sample_alignment()
    segment_predictions = _sample_predictions()
    scoring_result = score_pronunciation_segments(alignment_result, segment_predictions)
    hybrid_result = build_hybrid_diagnosis(alignment_result, segment_predictions, scoring_result)

    print(json.dumps(hybrid_result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
