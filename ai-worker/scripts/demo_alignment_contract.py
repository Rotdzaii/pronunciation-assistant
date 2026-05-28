from __future__ import annotations

import json
import sys
from pathlib import Path


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.contracts.alignment_contract import build_alignment_result, build_alignment_segment  # noqa: E402


def main() -> int:
    result = build_alignment_result(
        segments=[
            build_alignment_segment(index=0, segment_type="phone", phone="EH", word="example", start=0.0, end=0.1),
            build_alignment_segment(index=1, segment_type="phone", phone="G", word="example", start=0.1, end=0.2),
        ]
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
