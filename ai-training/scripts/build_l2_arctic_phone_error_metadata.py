from collections import Counter
from pathlib import Path
import csv
import re
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parse_l2_arctic_textgrid import parse_textgrid


DATASET_ROOT = Path("ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0")
OUTPUT_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_annotations_raw.csv")

VIETNAMESE_SPEAKERS = {
    "HQTV": {"l1": "Vietnamese", "gender": "M"},
    "PNV": {"l1": "Vietnamese", "gender": "M"},
    "THV": {"l1": "Vietnamese", "gender": "F"},
    "TLV": {"l1": "Vietnamese", "gender": "F"},
}

FIELDNAMES = [
    "dataset",
    "speaker_id",
    "l1",
    "gender",
    "utterance_id",
    "textgrid_path",
    "tier_name",
    "start_time",
    "end_time",
    "label",
    "normalized_label",
    "possible_error_type",
]


def normalize_label(label: str) -> str:
    return " ".join(label.strip().split())


def infer_possible_error_type(label: str) -> str:
    normalized = normalize_label(label).lower()

    if not normalized:
        return "unknown"

    parts = [part.strip().lower() for part in normalized.split(",")]
    code = parts[-1] if len(parts) >= 2 else ""

    if code in {"s", "sub", "substitution"}:
        return "substitution"

    if code in {"d", "del", "delete", "deletion"}:
        return "deletion"

    if code in {"a", "add", "addition", "i", "ins", "insert", "insertion"}:
        return "addition"

    if code in {"c", "corr", "correct", "ok"}:
        return "correct"

    if re.search(r"\b(sub|substitution)\b", normalized):
        return "substitution"

    if re.search(r"\b(del|deletion|deleted)\b", normalized):
        return "deletion"

    if re.search(r"\b(add|addition|insert|insertion|inserted)\b", normalized):
        return "addition"

    if re.search(r"\b(correct|corr|ok)\b", normalized):
        return "correct"

    return "unknown"


def iter_annotation_files():
    for speaker_id in VIETNAMESE_SPEAKERS:
        annotation_dir = DATASET_ROOT / speaker_id / "annotation"

        if not annotation_dir.exists():
            continue

        yield speaker_id, from_dir_sorted(annotation_dir)


def from_dir_sorted(annotation_dir: Path):
    return sorted(annotation_dir.glob("*.TextGrid"))


def build_rows() -> tuple[list[dict[str, str]], int]:
    rows = []
    file_count = 0

    for speaker_id, textgrid_paths in iter_annotation_files():
        speaker_info = VIETNAMESE_SPEAKERS[speaker_id]

        for textgrid_path in textgrid_paths:
            file_count += 1
            intervals = parse_textgrid(textgrid_path, speaker_id=speaker_id)

            for interval in intervals:
                normalized = normalize_label(interval.label)
                rows.append(
                    {
                        "dataset": "l2_arctic",
                        "speaker_id": speaker_id,
                        "l1": speaker_info["l1"],
                        "gender": speaker_info["gender"],
                        "utterance_id": interval.utterance_id,
                        "textgrid_path": str(textgrid_path).replace("\\", "/"),
                        "tier_name": interval.tier_name,
                        "start_time": f"{interval.start_time:.6f}",
                        "end_time": f"{interval.end_time:.6f}",
                        "label": interval.label,
                        "normalized_label": normalized,
                        "possible_error_type": infer_possible_error_type(normalized),
                    }
                )

    return rows, file_count


def print_counter(title: str, counter: Counter[str], limit: int | None = None):
    print(title)
    items = counter.most_common(limit)

    for key, count in items:
        display = key if key else "(empty)"
        display = display.encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"- {display}: {count}")


def main():
    rows, file_count = build_rows()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    tier_counts = Counter(row["tier_name"] for row in rows)
    label_counts = Counter(row["normalized_label"] for row in rows if row["normalized_label"])
    error_counts = Counter(row["possible_error_type"] for row in rows)

    print(f"Saved raw phone annotation metadata to: {OUTPUT_CSV}")
    print(f"Annotation TextGrid files processed: {file_count}")
    print(f"Total extracted rows: {len(rows)}")
    print()
    print_counter("Discovered tier names:", tier_counts)
    print()
    print_counter("Discovered non-empty labels:", label_counts, limit=50)
    print()
    print_counter("possible_error_type distribution:", error_counts)


if __name__ == "__main__":
    main()
