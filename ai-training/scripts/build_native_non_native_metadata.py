from collections import Counter
from pathlib import Path
import csv


CMU_METADATA_CSV = Path("ai-training/datasets/cmu-arctic/metadata/native_speakers_metadata.csv")
L2_METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_speakers_metadata.csv")
OUTPUT_CSV = Path("ai-training/datasets/combined/metadata/native_non_native_metadata.csv")

FIELDNAMES = [
    "dataset",
    "speaker_id",
    "l1",
    "gender",
    "split",
    "audio_path",
    "target_text",
    "label",
    "label_name",
    "source_type",
    "utterance_id",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def normalize_cmu_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {field: row.get(field, "") for field in FIELDNAMES}
    normalized["label"] = "0"
    normalized["label_name"] = "native_reference"
    normalized["source_type"] = row.get("source_type") or "native_reference"
    return normalized


def normalize_l2_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {field: row.get(field, "") for field in FIELDNAMES}
    normalized["label"] = "1"
    normalized["label_name"] = "non_native"
    normalized["source_type"] = row.get("source_type") or "l2_learner"
    return normalized


def keep_existing_audio(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    kept = []
    dropped = []

    for row in rows:
        audio_path = row.get("audio_path", "")

        if audio_path and Path(audio_path).exists():
            kept.append(row)
        else:
            dropped.append(row)

    return kept, dropped


def print_counts(rows: list[dict[str, str]], column: str):
    counts = Counter(row[column] for row in rows)

    print(f"Counts by {column}:")
    for key in sorted(counts):
        print(f"- {key}: {counts[key]}")


def main():
    cmu_rows = [normalize_cmu_row(row) for row in read_rows(CMU_METADATA_CSV)]
    l2_rows = [normalize_l2_row(row) for row in read_rows(L2_METADATA_CSV)]

    rows, dropped_rows = keep_existing_audio(cmu_rows + l2_rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metadata to: {OUTPUT_CSV}")
    print(f"Total rows: {len(rows)}")
    print(f"Dropped rows with missing audio_path: {len(dropped_rows)}")
    print()

    print_counts(rows, "dataset")
    print()
    print_counts(rows, "label_name")
    print()
    print_counts(rows, "split")


if __name__ == "__main__":
    main()
