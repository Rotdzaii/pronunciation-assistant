from collections import Counter
from pathlib import Path
import csv


RAW_ANNOTATIONS_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_annotations_raw.csv")
UTTERANCE_METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_speakers_metadata.csv")
OUTPUT_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification.csv")

FIELDNAMES = [
    "dataset",
    "speaker_id",
    "l1",
    "gender",
    "split",
    "audio_path",
    "utterance_id",
    "start_time",
    "end_time",
    "label",
    "error_type",
    "target_text",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def is_valid_segment(row: dict[str, str]) -> bool:
    try:
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
    except ValueError:
        return False

    return start_time >= 0 and end_time > start_time


def main():
    annotation_rows = read_csv(RAW_ANNOTATIONS_CSV)
    utterance_rows = read_csv(UTTERANCE_METADATA_CSV)
    utterance_lookup = {
        (row["speaker_id"], row["utterance_id"]): row for row in utterance_rows
    }

    rows = []
    skipped_missing_audio = 0
    skipped_unknown = 0
    skipped_non_phone_tier = 0
    skipped_invalid_time = 0

    for row in annotation_rows:
        error_type = row["possible_error_type"]

        if error_type == "unknown":
            skipped_unknown += 1
            continue

        if row["tier_name"] not in {"phones", "IPA"}:
            skipped_non_phone_tier += 1
            continue

        if not is_valid_segment(row):
            skipped_invalid_time += 1
            continue

        utterance = utterance_lookup.get((row["speaker_id"], row["utterance_id"]))

        if not utterance or not Path(utterance["audio_path"]).exists():
            skipped_missing_audio += 1
            continue

        rows.append(
            {
                "dataset": row["dataset"],
                "speaker_id": row["speaker_id"],
                "l1": row["l1"],
                "gender": row["gender"],
                "split": utterance["split"],
                "audio_path": utterance["audio_path"],
                "utterance_id": row["utterance_id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "label": row["label"],
                "error_type": error_type,
                "target_text": utterance["target_text"],
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["error_type"] for row in rows)

    print(f"Saved classification dataset to: {OUTPUT_CSV}")
    print(f"Rows: {len(rows)}")
    print("Rows by error_type:")

    for error_type, count in sorted(counts.items()):
        print(f"- {error_type}: {count}")

    print("Skipped rows:")
    print(f"- unknown error_type: {skipped_unknown}")
    print(f"- non-phone tier: {skipped_non_phone_tier}")
    print(f"- invalid segment time: {skipped_invalid_time}")
    print(f"- missing audio metadata/file: {skipped_missing_audio}")

    valid_classes = [error_type for error_type, count in counts.items() if count >= 50]

    if len(valid_classes) < 2:
        print("WARNING: Fewer than 2 error classes have at least 50 rows. Skip training.")


if __name__ == "__main__":
    main()
