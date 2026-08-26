from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv")
EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
DISJOINT_RESULTS_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_cnn_attention_results.json"

OUTPUT_JSON = EVALUATION_DIR / "speaker_disjoint_addition_analysis.json"
BY_SPEAKER_CSV = EVALUATION_DIR / "speaker_disjoint_addition_by_speaker.csv"
BY_L1_CSV = EVALUATION_DIR / "speaker_disjoint_addition_by_l1.csv"
DURATION_CSV = EVALUATION_DIR / "speaker_disjoint_addition_duration_summary.csv"
CONFUSION_CSV = EVALUATION_DIR / "speaker_disjoint_addition_confusion_analysis.csv"

VIETNAMESE_SPEAKERS = {"HQTV", "PNV", "THV", "TLV"}


def parse_label(label: str) -> dict[str, str]:
    parts = [part.strip() for part in str(label).split(",")]
    return {
        "target_phone": parts[0] if len(parts) > 0 else "",
        "observed_phone": parts[1] if len(parts) > 1 else "",
        "error_code": parts[2] if len(parts) > 2 else "",
    }


def count_frame(df: pd.DataFrame, columns: list[str], count_name: str = "rows") -> pd.DataFrame:
    return df.groupby(columns, dropna=False).size().reset_index(name=count_name)


def duration_summary(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for column in ["original_duration", "context_duration"]:
        if column in df.columns:
            frames.append(
                df.groupby("error_type")[column]
                .describe()
                .round(6)
                .reset_index()
                .assign(duration_column=column)
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_disjoint_results() -> dict:
    if not DISJOINT_RESULTS_JSON.exists():
        return {}
    return json.loads(DISJOINT_RESULTS_JSON.read_text(encoding="utf-8"))


def confusion_rows(results: dict) -> list[dict]:
    rows = []
    for fold in results.get("folds", []):
        held_out = fold["held_out_speaker"]
        matrix = fold["test"]["confusion_matrix"]
        for actual, predicted_counts in matrix.items():
            for predicted, count in predicted_counts.items():
                if actual == "addition" or predicted == "addition":
                    rows.append(
                        {
                            "held_out_speaker": held_out,
                            "actual_error_type": actual,
                            "predicted_error_type": predicted,
                            "rows": int(count),
                        }
                    )
    return rows


def misclassified_addition_summary(results: dict) -> dict:
    examples = []
    for fold in results.get("folds", []):
        examples.extend(fold["test"].get("misclassified_examples", []))

    if not examples:
        return {}

    df = pd.DataFrame(examples)
    involving_addition = df[
        (df["actual_error_type"] == "addition") | (df["predicted_error_type"] == "addition")
    ].copy()

    return {
        "total_misclassified_examples_saved": int(len(df)),
        "misclassified_involving_addition": int(len(involving_addition)),
        "actual_addition_misclassified_as": involving_addition[
            involving_addition["actual_error_type"] == "addition"
        ]["predicted_error_type"].value_counts().to_dict(),
        "non_addition_misclassified_as_addition": involving_addition[
            involving_addition["predicted_error_type"] == "addition"
        ]["actual_error_type"].value_counts().to_dict(),
    }


def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)
    additions = df[df["error_type"] == "addition"].copy()
    parsed_labels = additions["label"].apply(parse_label).apply(pd.Series)
    additions = pd.concat([additions, parsed_labels], axis=1)

    by_speaker = count_frame(additions, ["speaker_id", "l1", "split"]).sort_values(
        ["l1", "speaker_id", "split"]
    )
    vietnamese_by_speaker = count_frame(
        additions[additions["speaker_id"].isin(VIETNAMESE_SPEAKERS)],
        ["speaker_id", "split"],
    )
    by_l1 = count_frame(additions, ["l1", "split"]).sort_values(["l1", "split"])
    by_split = additions["split"].value_counts().to_dict()
    phone_pairs = count_frame(additions, ["target_phone", "observed_phone"]).sort_values(
        "rows",
        ascending=False,
    )
    duration = duration_summary(df)

    results = load_disjoint_results()
    confusion = pd.DataFrame(confusion_rows(results))
    if confusion.empty:
        confusion = pd.DataFrame(columns=["held_out_speaker", "actual_error_type", "predicted_error_type", "rows"])

    review = {
        "metadata_csv": str(METADATA_CSV),
        "total_rows": int(len(df)),
        "total_addition_rows": int(len(additions)),
        "vietnamese_addition_rows": int(additions["speaker_id"].isin(VIETNAMESE_SPEAKERS).sum()),
        "addition_by_split": {str(key): int(value) for key, value in by_split.items()},
        "addition_by_l1_total": additions["l1"].value_counts().to_dict(),
        "addition_by_vietnamese_speaker_total": additions[
            additions["speaker_id"].isin(VIETNAMESE_SPEAKERS)
        ]["speaker_id"].value_counts().to_dict(),
        "top_addition_phone_pairs": json.loads(phone_pairs.head(25).to_json(orient="records")),
        "speaker_disjoint_addition_confusion": json.loads(confusion.to_json(orient="records")),
        "speaker_disjoint_misclassified_addition_summary": misclassified_addition_summary(results),
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    by_speaker.to_csv(BY_SPEAKER_CSV, index=False, encoding="utf-8")
    by_l1.to_csv(BY_L1_CSV, index=False, encoding="utf-8")
    duration.to_csv(DURATION_CSV, index=False, encoding="utf-8")
    confusion.to_csv(CONFUSION_CSV, index=False, encoding="utf-8")
    OUTPUT_JSON.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Addition analysis")
    print("Total rows:", len(df))
    print("Total addition rows:", len(additions))
    print("Vietnamese addition rows:", review["vietnamese_addition_rows"])
    print()
    print("Vietnamese addition by speaker/split:")
    print(vietnamese_by_speaker.to_string(index=False))
    print()
    print("Addition by L1/split:")
    print(by_l1.to_string(index=False))
    print()
    print("Top addition target/observed phone labels:")
    print(phone_pairs.head(20).to_string(index=False))
    print()
    print("Addition-related speaker-disjoint confusion:")
    print(confusion.to_string(index=False))
    print()
    print("Saved outputs:")
    for path in [OUTPUT_JSON, BY_SPEAKER_CSV, BY_L1_CSV, DURATION_CSV, CONFUSION_CSV]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
