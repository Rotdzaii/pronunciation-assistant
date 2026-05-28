from __future__ import annotations

import json
import sys
from pathlib import Path


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.contracts.alignment_contract import build_alignment_result, build_alignment_segment  # noqa: E402
from app.scoring.scoring_service import score_pronunciation_segments  # noqa: E402


def _sample_alignment() -> dict[str, object]:
    return build_alignment_result(
        segments=[
            build_alignment_segment(index=0, segment_type="phone", phone="EH", word="example", start=0.0, end=0.12),
            build_alignment_segment(index=1, segment_type="phone", phone="G", word="example", start=0.12, end=0.24),
            build_alignment_segment(index=2, segment_type="phone", phone="Z", word="example", start=0.24, end=0.42),
        ],
        words=[
            build_alignment_segment(index=0, segment_type="word", word="example", start=0.0, end=0.42),
        ],
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
            "end": 0.12,
            "predicted_error_type": "deletion",
            "diagnosis_confidence": 0.82,
        },
        {
            "index": 1,
            "phone": "G",
            "word": "example",
            "start": 0.12,
            "end": 0.24,
            "predicted_error_type": "substitution",
            "diagnosis_confidence": 0.69,
        },
        {
            "index": 2,
            "phone": "Z",
            "word": "example",
            "start": 0.24,
            "end": 0.42,
            "predicted_error_type": "addition",
            "diagnosis_confidence": 0.51,
        },
    ]


def main() -> int:
    print("Note: heuristic score is not production GOP.")
    result = score_pronunciation_segments(_sample_alignment(), _sample_predictions())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
