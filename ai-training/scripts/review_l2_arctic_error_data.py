from pathlib import Path

import pandas as pd


INPUT_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification.csv")
OUTPUT_DIR = Path("ai-training/datasets/l2-arctic/evaluation")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "speaker_id",
        "utterance_id",
        "error_type",
        "start_time",
        "end_time",
        "audio_path",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    df["duration"] = df["end_time"] - df["start_time"]
    df["audio_exists"] = df["audio_path"].apply(lambda path: Path(path).exists())

    print("Dataset shape:", df.shape)
    print()
    print("Error type distribution:")
    print(df["error_type"].value_counts())
    print()
    print("Speaker distribution:")
    print(df["speaker_id"].value_counts())
    print()
    print("Audio exists:")
    print(df["audio_exists"].value_counts())
    print()

    duration_summary = (
        df.groupby("error_type")["duration"]
        .describe()
        .round(4)
        .reset_index()
    )

    print("Duration summary by error_type:")
    print(duration_summary.to_string(index=False))
    print()

    short_segments = df[df["duration"] <= 0.03].copy()
    very_long_segments = df[df["duration"] >= 2.0].copy()
    invalid_segments = df[df["duration"] <= 0].copy()
    missing_audio = df[~df["audio_exists"]].copy()

    print("Invalid segments duration <= 0:", len(invalid_segments))
    print("Very short segments duration <= 0.03s:", len(short_segments))
    print("Very long segments duration >= 2.0s:", len(very_long_segments))
    print("Missing audio rows:", len(missing_audio))
    print()

    for error_type in sorted(df["error_type"].unique()):
        subset = df[df["error_type"] == error_type].copy()
        sample = subset.sample(
            n=min(20, len(subset)),
            random_state=42,
        )

        output_path = OUTPUT_DIR / f"review_sample_{error_type}.csv"
        sample.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved sample for {error_type}: {output_path}")

    duration_summary_path = OUTPUT_DIR / "error_segment_duration_summary.csv"
    duration_summary.to_csv(duration_summary_path, index=False, encoding="utf-8")
    print(f"Saved duration summary: {duration_summary_path}")

    issues_path = OUTPUT_DIR / "error_data_quality_issues.csv"

    issues = pd.concat(
        [
            invalid_segments.assign(issue="invalid_duration"),
            short_segments.assign(issue="very_short_duration"),
            very_long_segments.assign(issue="very_long_duration"),
            missing_audio.assign(issue="missing_audio"),
        ],
        ignore_index=True,
    )

    if len(issues) > 0:
        issues.to_csv(issues_path, index=False, encoding="utf-8")
        print(f"Saved quality issues: {issues_path}")
    else:
        print("No obvious quality issues found.")


if __name__ == "__main__":
    main()