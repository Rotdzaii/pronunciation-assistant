from collections import Counter, defaultdict
from pathlib import Path
import csv
import json


INPUT_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_annotations_raw.csv")
OUTPUT_JSON = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_summary.json")


def read_rows() -> list[dict[str, str]]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Raw annotation CSV not found: {INPUT_CSV}")

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def print_counts(title: str, counts: Counter[str], limit: int | None = None):
    print(title)

    for key, count in counts.most_common(limit):
        display = key if key else "(empty)"
        display = display.encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"- {display}: {count}")


def main():
    rows = read_rows()

    by_speaker = Counter(row["speaker_id"] for row in rows)
    by_tier = Counter(row["tier_name"] for row in rows)
    by_error = Counter(row["possible_error_type"] for row in rows)
    by_label = Counter(row["normalized_label"] for row in rows if row["normalized_label"])
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        error_type = row["possible_error_type"]

        if len(examples[error_type]) < 5:
            examples[error_type].append(row)

    unknown_rows = by_error.get("unknown", 0)

    print(f"Total rows: {len(rows)}")
    print()
    print_counts("Rows by speaker:", by_speaker)
    print()
    print_counts("Rows by tier_name:", by_tier)
    print()
    print_counts("Rows by possible_error_type:", by_error)
    print()
    print_counts("Top 30 normalized labels:", by_label, limit=30)
    print()
    print(f"Unknown rows: {unknown_rows}")
    print()
    print("Example rows by possible_error_type:")

    for error_type in sorted(examples):
        print(f"- {error_type}:")

        for row in examples[error_type]:
            print(
                "  "
                f"{row['speaker_id']} {row['utterance_id']} "
                f"{row['tier_name']} {row['start_time']}-{row['end_time']} "
                f"label={row['label'].encode('ascii', errors='backslashreplace').decode('ascii')!r}"
            )

    summary = {
        "total_rows": len(rows),
        "rows_by_speaker": dict(by_speaker),
        "rows_by_tier_name": dict(by_tier),
        "rows_by_possible_error_type": dict(by_error),
        "top_30_normalized_labels": dict(by_label.most_common(30)),
        "unknown_rows": unknown_rows,
        "examples_by_possible_error_type": {
            key: value for key, value in sorted(examples.items())
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print()
    print(f"Saved summary JSON to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
