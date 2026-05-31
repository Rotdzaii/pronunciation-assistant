from __future__ import annotations

from pathlib import Path
import csv
import json


EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
OUTPUT_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_comparison.csv"

VIETNAMESE_ONLY_JSON = EVALUATION_DIR / "cnn_attention_eval_metrics.json"
ALL_SPEAKERS_JSON = EVALUATION_DIR / "all_speakers_cnn_attention_eval_metrics.json"
SPEAKER_DISJOINT_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_cnn_attention_results.json"
LABELS = ["addition", "deletion", "substitution"]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def split_metrics(data: dict, split_name: str) -> dict:
    if "evaluation_slices" in data:
        return data["evaluation_slices"].get(split_name, {}).get("metrics", {})
    if "splits" in data:
        return data["splits"].get(split_name, {}).get("metrics", {})
    return data.get(split_name, {})


def class_f1(metrics: dict, label: str):
    per_class = metrics.get("per_class", {})
    if isinstance(per_class, dict) and isinstance(per_class.get(label), dict):
        return per_class[label].get("f1")
    return None


def build_standard_row(name: str, scope: str, metrics: dict, note: str) -> dict:
    row = {
        "run": name,
        "scope": scope,
        "note": note,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "macro_f1_std": "",
        "weighted_f1": metrics.get("weighted_f1"),
        "addition_f1": class_f1(metrics, "addition"),
        "addition_f1_std": "",
        "deletion_f1": class_f1(metrics, "deletion"),
        "deletion_f1_std": "",
        "substitution_f1": class_f1(metrics, "substitution"),
        "substitution_f1_std": "",
    }
    return row


def build_disjoint_row(data: dict) -> dict:
    summary = data["summary"]
    return {
        "run": "vietnamese_speaker_disjoint_cnn_attention",
        "scope": "Mean held-out Vietnamese speaker",
        "note": "Mean across leave-one-Vietnamese-speaker-out folds.",
        "accuracy": summary["mean_accuracy"],
        "macro_f1": summary["mean_macro_f1"],
        "macro_f1_std": summary["std_macro_f1"],
        "weighted_f1": summary["mean_weighted_f1"],
        "addition_f1": summary["mean_addition_f1"],
        "addition_f1_std": summary["std_addition_f1"],
        "deletion_f1": summary["mean_deletion_f1"],
        "deletion_f1_std": summary["std_deletion_f1"],
        "substitution_f1": summary["mean_substitution_f1"],
        "substitution_f1_std": summary["std_substitution_f1"],
    }


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def main() -> None:
    vietnamese_only = load_json(VIETNAMESE_ONLY_JSON)
    all_speakers = load_json(ALL_SPEAKERS_JSON)
    disjoint = load_json(SPEAKER_DISJOINT_JSON)

    rows = [
        build_standard_row(
            "vietnamese_only_cnn_attention",
            "Vietnamese-only original test split",
            split_metrics(vietnamese_only, "test"),
            "Previous Vietnamese-only CNN Attention baseline.",
        ),
        build_standard_row(
            "all_speakers_cnn_attention",
            "Vietnamese subset original test split",
            split_metrics(all_speakers, "test_vietnamese"),
            "All-speaker model evaluated on Vietnamese subset; not speaker-disjoint.",
        ),
        build_disjoint_row(disjoint),
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved comparison to:", OUTPUT_CSV)
    print()
    print("Macro F1 comparison:")
    for row in sorted(rows, key=lambda item: safe_float(item["macro_f1"]), reverse=True):
        std = f" +/- {float(row['macro_f1_std']):.4f}" if row["macro_f1_std"] != "" else ""
        print(f"- {row['run']} ({row['scope']}): {float(row['macro_f1']):.4f}{std}")

    print()
    print("Addition F1 comparison:")
    for row in sorted(rows, key=lambda item: safe_float(item["addition_f1"]), reverse=True):
        std = f" +/- {float(row['addition_f1_std']):.4f}" if row["addition_f1_std"] != "" else ""
        print(f"- {row['run']} ({row['scope']}): {float(row['addition_f1']):.4f}{std}")

    disjoint_macro = safe_float(rows[-1]["macro_f1"])
    all_speakers_macro = safe_float(rows[1]["macro_f1"])

    print()
    if disjoint_macro >= all_speakers_macro:
        print("Decision signal: speaker-disjoint result supports further consideration, but still needs caution.")
    else:
        print("Decision signal: speaker-disjoint result does not support replacing the main model yet.")
    print("Confidence remains classifier confidence, not pronunciation correctness.")


if __name__ == "__main__":
    main()
