from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv")
OUTPUT_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
VIETNAMESE_SPEAKERS = {"HQTV", "PNV", "THV", "TLV"}


def value_counts_frame(df: pd.DataFrame, column: str, count_name: str = "rows") -> pd.DataFrame:
    return df[column].value_counts(dropna=False).rename_axis(column).reset_index(name=count_name)


def two_way_counts(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return df.groupby(group_columns, dropna=False).size().reset_index(name="rows")


def duration_summary(df: pd.DataFrame, column: str, group_column: str = "error_type") -> pd.DataFrame:
    return df.groupby(group_column)[column].describe().round(6).reset_index()


def to_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().items()}


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "speaker_id",
        "l1",
        "split",
        "audio_path",
        "error_type",
        "start_time",
        "end_time",
        "original_duration",
    }
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    df["audio_exists"] = df["audio_path"].apply(lambda path: Path(path).exists())

    error_distribution = value_counts_frame(df, "error_type")
    l1_distribution = value_counts_frame(df, "l1")
    speaker_distribution = value_counts_frame(df, "speaker_id")
    split_distribution = value_counts_frame(df, "split")
    error_by_l1 = two_way_counts(df, ["l1", "error_type"])
    error_by_speaker = two_way_counts(df, ["speaker_id", "l1", "error_type"])
    split_by_error = two_way_counts(df, ["split", "error_type"])
    audio_distribution = value_counts_frame(df, "audio_exists")
    original_duration_summary = duration_summary(df, "original_duration")

    duration_outputs = [original_duration_summary.assign(duration_column="original_duration")]

    if "context_duration" in df.columns:
        context_duration_summary = duration_summary(df, "context_duration")
        duration_outputs.append(context_duration_summary.assign(duration_column="context_duration"))
    else:
        context_duration_summary = pd.DataFrame()

    duration_output = pd.concat(duration_outputs, ignore_index=True)

    vietnamese_df = df[(df["l1"] == "Vietnamese") | (df["speaker_id"].isin(VIETNAMESE_SPEAKERS))].copy()
    addition_df = df[df["error_type"] == "addition"].copy()
    addition_by_l1 = two_way_counts(addition_df, ["l1", "speaker_id"])

    error_distribution.to_csv(OUTPUT_DIR / "all_speakers_error_distribution.csv", index=False, encoding="utf-8")
    error_by_l1.to_csv(OUTPUT_DIR / "all_speakers_error_by_l1.csv", index=False, encoding="utf-8")
    error_by_speaker.to_csv(OUTPUT_DIR / "all_speakers_error_by_speaker.csv", index=False, encoding="utf-8")
    split_distribution.to_csv(OUTPUT_DIR / "all_speakers_split_distribution.csv", index=False, encoding="utf-8")
    duration_output.to_csv(OUTPUT_DIR / "all_speakers_duration_summary.csv", index=False, encoding="utf-8")

    review = {
        "input_csv": str(INPUT_CSV),
        "total_rows": int(len(df)),
        "columns": list(df.columns),
        "speaker_count": int(df["speaker_id"].nunique()),
        "l1_group_count": int(df["l1"].nunique()),
        "error_type_distribution": value_counts_dict(df["error_type"]),
        "l1_group_distribution": value_counts_dict(df["l1"]),
        "speaker_distribution": value_counts_dict(df["speaker_id"]),
        "split_distribution": value_counts_dict(df["split"]),
        "split_by_error_type": to_records(split_by_error),
        "audio_exists_distribution": {str(key): int(value) for key, value in df["audio_exists"].value_counts().items()},
        "duration_summary_by_error_type": to_records(original_duration_summary),
        "context_duration_summary_by_error_type": to_records(context_duration_summary) if not context_duration_summary.empty else [],
        "vietnamese_subset": {
            "rows": int(len(vietnamese_df)),
            "speaker_count": int(vietnamese_df["speaker_id"].nunique()),
            "error_type_distribution": value_counts_dict(vietnamese_df["error_type"]),
            "split_distribution": value_counts_dict(vietnamese_df["split"]),
        },
        "addition_distribution_by_l1_speaker": to_records(addition_by_l1),
    }

    review_path = OUTPUT_DIR / "all_speakers_dataset_review.json"
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")

    print("Dataset shape:", df.shape)
    print()
    print("Error type distribution:")
    print(error_distribution.to_string(index=False))
    print()
    print("L1 group distribution:")
    print(l1_distribution.to_string(index=False))
    print()
    print("Speaker distribution:")
    print(speaker_distribution.to_string(index=False))
    print()
    print("Error type by L1:")
    print(error_by_l1.to_string(index=False))
    print()
    print("Split distribution:")
    print(split_distribution.to_string(index=False))
    print()
    print("Split by error_type:")
    print(split_by_error.to_string(index=False))
    print()
    print("Audio exists:")
    print(audio_distribution.to_string(index=False))
    print()
    print("Duration summary by error_type:")
    print(original_duration_summary.to_string(index=False))
    print()
    if not context_duration_summary.empty:
        print("Context duration summary by error_type:")
        print(context_duration_summary.to_string(index=False))
        print()
    print("Vietnamese subset rows:", len(vietnamese_df))
    print(vietnamese_df["error_type"].value_counts())
    print()
    print("Addition distribution by L1/speaker:")
    print(addition_by_l1.to_string(index=False))
    print()
    print(f"Saved review JSON: {review_path}")


if __name__ == "__main__":
    main()
