from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import argparse
import re


DATASET_ROOT = Path("ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0")
VIETNAMESE_SPEAKERS = ("HQTV", "PNV", "THV", "TLV")


@dataclass
class TextGridInterval:
    speaker_id: str
    utterance_id: str
    tier_name: str
    start_time: float
    end_time: float
    label: str


def read_textgrid(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _quoted_value(line: str) -> str:
    match = re.search(r'"(.*)"', line)
    return match.group(1) if match else ""


def parse_textgrid(path: Path, speaker_id: str | None = None) -> list[TextGridInterval]:
    """Parse Praat long-text TextGrid interval tiers.

    L2-ARCTIC annotation and alignment files are Praat TextGrid files using
    IntervalTier blocks. This parser intentionally ignores point tiers and any
    unsupported metadata while preserving interval labels verbatim.
    """
    speaker = speaker_id or path.parent.parent.name
    utterance_id = path.stem
    intervals: list[TextGridInterval] = []

    current_tier = ""
    in_interval_tier = False
    pending_start: float | None = None
    pending_end: float | None = None

    for raw_line in read_textgrid(path):
        line = raw_line.strip()

        if line.startswith("class ="):
            in_interval_tier = _quoted_value(line) == "IntervalTier"
            current_tier = ""
            pending_start = None
            pending_end = None
            continue

        if in_interval_tier and line.startswith("name ="):
            current_tier = _quoted_value(line)
            continue

        if not in_interval_tier or not current_tier:
            continue

        if line.startswith("intervals ["):
            pending_start = None
            pending_end = None
            continue

        if line.startswith("xmin ="):
            pending_start = float(line.split("=", 1)[1].strip())
            continue

        if line.startswith("xmax ="):
            pending_end = float(line.split("=", 1)[1].strip())
            continue

        if line.startswith("text =") and pending_start is not None and pending_end is not None:
            intervals.append(
                TextGridInterval(
                    speaker_id=speaker,
                    utterance_id=utterance_id,
                    tier_name=current_tier,
                    start_time=pending_start,
                    end_time=pending_end,
                    label=_quoted_value(line),
                )
            )
            pending_start = None
            pending_end = None

    return intervals


def iter_textgrids(source: str):
    for speaker_id in VIETNAMESE_SPEAKERS:
        folder = DATASET_ROOT / speaker_id / source

        if not folder.exists():
            continue

        yield from sorted(folder.glob("*.TextGrid"))


def main():
    parser = argparse.ArgumentParser(description="Parse L2-ARCTIC TextGrid interval tiers.")
    parser.add_argument(
        "--source",
        choices=("annotation", "textgrid", "both"),
        default="annotation",
        help="TextGrid source folder to summarize.",
    )
    args = parser.parse_args()

    sources = ("annotation", "textgrid") if args.source == "both" else (args.source,)
    tier_counts: Counter[str] = Counter()
    file_count = 0
    row_count = 0

    for source in sources:
        for path in iter_textgrids(source):
            intervals = parse_textgrid(path)
            file_count += 1
            row_count += len(intervals)
            tier_counts.update(interval.tier_name for interval in intervals)

    print(f"TextGrid files parsed: {file_count}")
    print(f"Intervals extracted: {row_count}")
    print("Tier names found:")

    for tier_name, count in sorted(tier_counts.items()):
        print(f"- {tier_name}: {count}")


if __name__ == "__main__":
    main()
