from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv")
OUTPUT_DIR = Path("ai-training/datasets/l2-arctic/evaluation")


def value_counts_frame(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="rows")


def missing_counts(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        df[~df["audio_exists"]]
        .groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="missing_audio_rows")
        .sort_values("missing_audio_rows", ascending=False)
    )


def to_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().items()}


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required_columns = {
        "speaker_id",
        "l1",
        "split",
        "audio_path",
        "error_type",
    }
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df["audio_exists"] = df["audio_path"].apply(lambda path: Path(path).exists())

    availability = value_counts_frame(df, "audio_exists")
    missing_by_speaker = missing_counts(df, ["speaker_id", "l1"])
    missing_by_l1 = missing_counts(df, ["l1"])
    missing_by_split = missing_counts(df, ["split"])
    missing_by_error_type = missing_counts(df, ["error_type"])

    speaker_availability = (
        df.groupby(["speaker_id", "l1"], dropna=False)
        .agg(
            rows=("audio_path", "size"),
            audio_exists_rows=("audio_exists", "sum"),
            missing_audio_rows=("audio_exists", lambda values: int((~values).sum())),
        )
        .reset_index()
    )
    speaker_availability["audio_ready"] = speaker_availability["missing_audio_rows"] == 0

    availability.to_csv(
        OUTPUT_DIR / "all_speakers_audio_availability.csv",
        index=False,
        encoding="utf-8",
    )
    missing_by_speaker.to_csv(
        OUTPUT_DIR / "all_speakers_missing_audio_by_speaker.csv",
        index=False,
        encoding="utf-8",
    )

    review = {
        "input_csv": str(INPUT_CSV),
        "total_rows": int(len(df)),
        "audio_exists_distribution": value_counts_dict(df["audio_exists"]),
        "speaker_count": int(df["speaker_id"].nunique()),
        "speakers_ready": int(speaker_availability["audio_ready"].sum()),
        "speakers_missing_audio": int((~speaker_availability["audio_ready"]).sum()),
        "missing_audio_by_speaker": to_records(missing_by_speaker),
        "missing_audio_by_l1_group": to_records(missing_by_l1),
        "missing_audio_by_split": to_records(missing_by_split),
        "missing_audio_by_error_type": to_records(missing_by_error_type),
        "all_audio_ready_for_training": bool(df["audio_exists"].all()),
    }

    review_path = OUTPUT_DIR / "all_speakers_audio_validation.json"
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")

    print("Total rows:", len(df))
    print()
    print("Audio exists true/false:")
    print(availability.to_string(index=False))
    print()
    print("Missing audio by speaker:")
    print(missing_by_speaker.to_string(index=False) if len(missing_by_speaker) else "(none)")
    print()
    print("Missing audio by L1 group:")
    print(missing_by_l1.to_string(index=False) if len(missing_by_l1) else "(none)")
    print()
    print("Missing audio by split:")
    print(missing_by_split.to_string(index=False) if len(missing_by_split) else "(none)")
    print()
    print("Missing audio by error_type:")
    print(missing_by_error_type.to_string(index=False) if len(missing_by_error_type) else "(none)")
    print()
    print("All audio ready for training:", bool(df["audio_exists"].all()))
    print(f"Saved availability CSV: {OUTPUT_DIR / 'all_speakers_audio_availability.csv'}")
    print(f"Saved missing-by-speaker CSV: {OUTPUT_DIR / 'all_speakers_missing_audio_by_speaker.csv'}")
    print(f"Saved validation JSON: {review_path}")


if __name__ == "__main__":
    main()
