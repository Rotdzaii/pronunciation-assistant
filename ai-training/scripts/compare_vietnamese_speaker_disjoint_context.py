from __future__ import annotations

from pathlib import Path
import csv
import json


EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
BASELINE_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_cnn_attention_results.json"
ADDITION_FOCUS_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_addition_focus_results.json"
CONTEXT_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_results.json"
OUTPUT_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_comparison.csv"

LABELS = ["addition", "deletion", "substitution"]

DOCUMENTED_ADDITION_FOCUS = {
    "run": "addition_focused_sampler",
    "scope": "Mean held-out Vietnamese speaker",
    "context_mode": "",
    "note": "Documented previous result; result JSON not present on this branch.",
    "accuracy": 0.5689,
    "accuracy_std": 0.0428,
    "macro_f1": 0.4715,
    "macro_f1_std": 0.0299,
    "weighted_f1": "",
    "weighted_f1_std": "",
    "addition_f1": 0.0958,
    "addition_f1_std": 0.0348,
    "deletion_f1": 0.6347,
    "deletion_f1_std": 0.0225,
    "substitution_f1": 0.6839,
    "substitution_f1_std": 0.0376,
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summary_row(name: str, data: dict, note: str) -> dict:
    summary = data["summary"]
    return {
        "run": name,
        "scope": "Mean held-out Vietnamese speaker",
        "context_mode": "",
        "note": note,
        "accuracy": summary["mean_accuracy"],
        "accuracy_std": summary["std_accuracy"],
        "macro_f1": summary["mean_macro_f1"],
        "macro_f1_std": summary["std_macro_f1"],
        "weighted_f1": summary["mean_weighted_f1"],
        "weighted_f1_std": summary["std_weighted_f1"],
        "addition_f1": summary["mean_addition_f1"],
        "addition_f1_std": summary["std_addition_f1"],
        "deletion_f1": summary["mean_deletion_f1"],
        "deletion_f1_std": summary["std_deletion_f1"],
        "substitution_f1": summary["mean_substitution_f1"],
        "substitution_f1_std": summary["std_substitution_f1"],
    }


def context_rows(data: dict) -> list[dict]:
    rows = []
    for item in data["summary"]["by_context_mode"]:
        rows.append(
            {
                "run": "context_window_cnn_attention",
                "scope": "Mean held-out Vietnamese speaker",
                "context_mode": item["context_mode"],
                "note": "Context-window variant using the same CNN Attention architecture and fold protocol.",
                "accuracy": item["mean_accuracy"],
                "accuracy_std": item["std_accuracy"],
                "macro_f1": item["mean_macro_f1"],
                "macro_f1_std": item["std_macro_f1"],
                "weighted_f1": item["mean_weighted_f1"],
                "weighted_f1_std": item["std_weighted_f1"],
                "addition_f1": item["mean_addition_f1"],
                "addition_f1_std": item["std_addition_f1"],
                "deletion_f1": item["mean_deletion_f1"],
                "deletion_f1_std": item["std_deletion_f1"],
                "substitution_f1": item["mean_substitution_f1"],
                "substitution_f1_std": item["std_substitution_f1"],
            }
        )
    return rows


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def format_metric(row: dict, metric: str) -> str:
    value = float(row[metric])
    std_key = f"{metric}_std"
    if row.get(std_key) not in ("", None):
        return f"{value:.4f} +/- {float(row[std_key]):.4f}"
    return f"{value:.4f}"


def main() -> None:
    baseline = load_json(BASELINE_JSON)
    context = load_json(CONTEXT_JSON)

    rows = [
        summary_row(
            "baseline_speaker_disjoint_cnn_attention",
            baseline,
            "Previous speaker-disjoint baseline using original segment crop.",
        )
    ]

    if ADDITION_FOCUS_JSON.exists():
        rows.append(
            summary_row(
                "addition_focused_sampler",
                load_json(ADDITION_FOCUS_JSON),
                "Previous addition-focused sampler variant.",
            )
        )
    else:
        rows.append(DOCUMENTED_ADDITION_FOCUS)

    rows.extend(context_rows(context))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved comparison to:", OUTPUT_CSV)
    print()
    print("Macro F1 comparison:")
    for row in sorted(rows, key=lambda item: safe_float(item["macro_f1"]), reverse=True):
        suffix = f" [{row['context_mode']}]" if row["context_mode"] else ""
        print(f"- {row['run']}{suffix}: {format_metric(row, 'macro_f1')}")

    print()
    print("Addition F1 comparison:")
    for row in sorted(rows, key=lambda item: safe_float(item["addition_f1"]), reverse=True):
        suffix = f" [{row['context_mode']}]" if row["context_mode"] else ""
        print(f"- {row['run']}{suffix}: {format_metric(row, 'addition_f1')}")

    baseline_row = rows[0]
    best_context = max([row for row in rows if row["run"] == "context_window_cnn_attention"], key=lambda row: safe_float(row["addition_f1"]))
    addition_delta = safe_float(best_context["addition_f1"]) - safe_float(baseline_row["addition_f1"])
    macro_delta = safe_float(best_context["macro_f1"]) - safe_float(baseline_row["macro_f1"])

    print()
    print(
        "Best context addition delta vs baseline: "
        f"{addition_delta:+.4f}; macro F1 delta vs baseline: {macro_delta:+.4f}."
    )
    if addition_delta > 0.0 and macro_delta >= -0.02:
        print("Recommendation signal: context is a candidate for follow-up, with limited macro F1 cost.")
    else:
        print("Recommendation signal: keep context-window model as an experiment, not a replacement.")
    print("Confidence remains classifier confidence, not pronunciation correctness.")


if __name__ == "__main__":
    main()
