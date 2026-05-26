from pathlib import Path

import pandas as pd


INPUT_RAW_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_annotations_raw.csv")
BASE_METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_speakers_metadata.csv")
OUTPUT_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")

CONTEXT_SECONDS = 0.25
MIN_CONTEXT_DURATION = 0.50
MAX_CONTEXT_DURATION = 1.00

VIETNAMESE_SPEAKERS = {
    "HQTV": {"l1": "Vietnamese", "gender": "M"},
    "PNV": {"l1": "Vietnamese", "gender": "F"},
    "THV": {"l1": "Vietnamese", "gender": "F"},
    "TLV": {"l1": "Vietnamese", "gender": "M"},
}


def normalize_error_type(value: str) -> str:
    value = str(value).strip().lower()

    if value in {"addition", "insertion"}:
        return "addition"

    if value == "deletion":
        return "deletion"

    if value == "substitution":
        return "substitution"

    return "unknown"


def build_audio_lookup() -> dict[tuple[str, str], dict]:
    if not BASE_METADATA_CSV.exists():
        raise FileNotFoundError(f"Base metadata not found: {BASE_METADATA_CSV}")

    base_df = pd.read_csv(BASE_METADATA_CSV)

    lookup = {}

    for _, row in base_df.iterrows():
        key = (row["speaker_id"], row["utterance_id"])
        lookup[key] = {
            "audio_path": row["audio_path"],
            "target_text": row.get("target_text", ""),
            "split": row.get("split", "train"),
        }

    return lookup


def add_context_window(row: pd.Series) -> tuple[float, float]:
    start = float(row["start_time"])
    end = float(row["end_time"])

    center = (start + end) / 2.0

    context_start = max(0.0, start - CONTEXT_SECONDS)
    context_end = end + CONTEXT_SECONDS

    duration = context_end - context_start

    if duration < MIN_CONTEXT_DURATION:
        half = MIN_CONTEXT_DURATION / 2.0
        context_start = max(0.0, center - half)
        context_end = center + half

    duration = context_end - context_start

    if duration > MAX_CONTEXT_DURATION:
        half = MAX_CONTEXT_DURATION / 2.0
        context_start = max(0.0, center - half)
        context_end = context_start + MAX_CONTEXT_DURATION

    return round(context_start, 6), round(context_end, 6)


def main():
    if not INPUT_RAW_CSV.exists():
        raise FileNotFoundError(f"Raw annotation CSV not found: {INPUT_RAW_CSV}")

    raw_df = pd.read_csv(INPUT_RAW_CSV)
    audio_lookup = build_audio_lookup()

    required_columns = {
        "speaker_id",
        "utterance_id",
        "tier_name",
        "start_time",
        "end_time",
        "label",
        "possible_error_type",
    }

    missing_columns = required_columns - set(raw_df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    print("Raw rows:", len(raw_df))
    print("Raw tier distribution:")
    print(raw_df["tier_name"].value_counts())
    print()

    df = raw_df.copy()

    df["error_type"] = df["possible_error_type"].apply(normalize_error_type)

    df = df[df["error_type"] != "unknown"].copy()
    print("After removing unknown:", len(df))
    print(df["error_type"].value_counts())
    print()

    if "tier_name" in df.columns:
        phones_df = df[df["tier_name"].astype(str).str.lower() == "phones"].copy()

        if len(phones_df) > 0:
            df = phones_df
            print("Using tier_name == phones")
        else:
            print("No phones tier found, using all non-unknown rows.")

    print("After tier filtering:", len(df))
    print(df["error_type"].value_counts())
    print()

    df["original_duration"] = df["end_time"].astype(float) - df["start_time"].astype(float)

    before_dedup = len(df)

    dedup_columns = [
        "speaker_id",
        "utterance_id",
        "start_time",
        "end_time",
        "error_type",
    ]

    df = df.drop_duplicates(subset=dedup_columns).copy()

    print("Removed duplicates:", before_dedup - len(df))
    print("After dedup:", len(df))
    print(df["error_type"].value_counts())
    print()

    rows = []

    missing_audio_lookup = 0

    for _, row in df.iterrows():
        speaker_id = row["speaker_id"]
        utterance_id = row["utterance_id"]
        key = (speaker_id, utterance_id)

        base_info = audio_lookup.get(key)

        if base_info is None:
            missing_audio_lookup += 1
            continue

        context_start, context_end = add_context_window(row)

        audio_path = base_info["audio_path"]

        rows.append(
            {
                "dataset": "l2_arctic",
                "speaker_id": speaker_id,
                "l1": VIETNAMESE_SPEAKERS.get(speaker_id, {}).get("l1", "Vietnamese"),
                "gender": VIETNAMESE_SPEAKERS.get(speaker_id, {}).get("gender", ""),
                "split": base_info["split"],
                "audio_path": audio_path,
                "utterance_id": utterance_id,
                "target_text": base_info.get("target_text", ""),
                "tier_name": row["tier_name"],
                "label": row["label"],
                "error_type": row["error_type"],
                "start_time": round(float(row["start_time"]), 6),
                "end_time": round(float(row["end_time"]), 6),
                "original_duration": round(float(row["original_duration"]), 6),
                "context_start_time": context_start,
                "context_end_time": context_end,
                "context_duration": round(context_end - context_start, 6),
            }
        )

    output_df = pd.DataFrame(rows)

    output_df["audio_exists"] = output_df["audio_path"].apply(lambda path: Path(path).exists())

    print("Missing audio lookup rows:", missing_audio_lookup)
    print("Output rows:", len(output_df))
    print()
    print("Output error distribution:")
    print(output_df["error_type"].value_counts())
    print()
    print("Output split distribution:")
    print(output_df["split"].value_counts())
    print()
    print("Audio exists:")
    print(output_df["audio_exists"].value_counts())
    print()
    print("Original duration summary:")
    print(output_df.groupby("error_type")["original_duration"].describe().round(4))
    print()
    print("Context duration summary:")
    print(output_df.groupby("error_type")["context_duration"].describe().round(4))

    output_df = output_df.drop(columns=["audio_exists"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print()
    print(f"Saved clean v2 dataset to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()