from __future__ import annotations

from pathlib import Path
import csv
import json


EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
OUTPUT_CSV = EVALUATION_DIR / "all_speakers_cnn_attention_comparison.csv"
LABELS = ["addition", "deletion", "substitution"]

RUNS = [
    {
        "run": "vietnamese_only_cnn_attention",
        "scope": "Vietnamese-only test",
        "path": EVALUATION_DIR / "cnn_attention_eval_metrics.json",
        "slice": "test",
        "note": "Previous Vietnamese-only CNN Attention baseline.",
    },
    {
        "run": "all_speakers_cnn_attention",
        "scope": "All-speaker test",
        "path": EVALUATION_DIR / "all_speakers_cnn_attention_eval_metrics.json",
        "slice": "test",
        "note": "CNN Attention trained on all L2-ARCTIC speakers.",
    },
    {
        "run": "all_speakers_cnn_attention",
        "scope": "Vietnamese subset test",
        "path": EVALUATION_DIR / "all_speakers_cnn_attention_eval_metrics.json",
        "slice": "test_vietnamese",
        "note": "All-speaker model evaluated only on Vietnamese rows.",
    },
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"Skipping missing metrics: {path}")
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_metrics(data: dict, slice_name: str) -> dict:
    if "evaluation_slices" in data:
        return data["evaluation_slices"].get(slice_name, {}).get("metrics", {})

    if "splits" in data:
        return data["splits"].get(slice_name, {}).get("metrics", {})

    return data.get(slice_name, {})


def metric_value(metrics: dict, key: str):
    return metrics.get(key)


def class_f1(metrics: dict, label: str):
    per_class = metrics.get("per_class", {})
    if isinstance(per_class, dict) and isinstance(per_class.get(label), dict):
        return per_class[label].get("f1")
    return None


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def build_row(run: dict, metrics: dict) -> dict:
    row = {
        "run": run["run"],
        "scope": run["scope"],
        "note": run["note"],
        "total_samples": metric_value(metrics, "total_samples"),
        "accuracy": metric_value(metrics, "accuracy"),
        "macro_f1": metric_value(metrics, "macro_f1"),
        "weighted_f1": metric_value(metrics, "weighted_f1"),
    }
    for label in LABELS:
        row[f"{label}_f1"] = class_f1(metrics, label)
    return row


def print_ranking(rows: list[dict], title: str, filter_text: str | None = None) -> None:
    print()
    print(title)
    filtered = [row for row in rows if filter_text is None or filter_text in row["scope"]]
    ranked = sorted(filtered, key=lambda row: safe_float(row["macro_f1"]), reverse=True)
    for index, row in enumerate(ranked, start=1):
        print(f"{index}. {row['run']} ({row['scope']}) - macro_f1={row['macro_f1']}")


def main() -> None:
    rows = []

    for run in RUNS:
        data = load_json(run["path"])
        if data is None:
            continue

        metrics = extract_metrics(data, run["slice"])
        if not metrics:
            print(f"Skipping missing slice {run['slice']} in {run['path']}")
            continue

        rows.append(build_row(run, metrics))

    if not rows:
        raise RuntimeError("No metric rows available for comparison.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved comparison to:", OUTPUT_CSV)
    print()
    print("Comparison summary:")
    for row in rows:
        print("-" * 60)
        print("run:", row["run"])
        print("scope:", row["scope"])
        print("accuracy:", row["accuracy"])
        print("macro_f1:", row["macro_f1"])
        print("weighted_f1:", row["weighted_f1"])
        print("addition_f1:", row["addition_f1"])
        print("deletion_f1:", row["deletion_f1"])
        print("substitution_f1:", row["substitution_f1"])

    print_ranking(rows, "Overall ranking by test macro F1:")
    print_ranking(rows, "Vietnamese subset ranking by macro F1:", filter_text="Vietnamese")

    print()
    print("Addition F1 comparison:")
    ranked_addition = sorted(rows, key=lambda row: safe_float(row["addition_f1"]), reverse=True)
    for index, row in enumerate(ranked_addition, start=1):
        print(f"{index}. {row['run']} ({row['scope']}) - addition_f1={row['addition_f1']}")


if __name__ == "__main__":
    main()
