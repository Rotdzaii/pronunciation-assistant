from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.alignment.textgrid_parser import parse_textgrid  # noqa: E402


SAMPLE_TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.45
            text = "example"
        intervals [2]:
            xmin = 0.45
            xmax = 1.0
            text = "text"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.2
            text = "EH"
        intervals [2]:
            xmin = 0.2
            xmax = 0.45
            text = "G"
        intervals [3]:
            xmin = 0.45
            xmax = 1.0
            text = "T"
"""


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="textgrid-parser-demo-") as temp_dir:
        textgrid_path = Path(temp_dir) / "sample.TextGrid"
        textgrid_path.write_text(SAMPLE_TEXTGRID, encoding="utf-8")
        result = parse_textgrid(textgrid_path)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
