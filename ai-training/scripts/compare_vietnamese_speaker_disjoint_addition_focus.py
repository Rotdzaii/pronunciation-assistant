from __future__ import annotations

from pathlib import Path
import csv
import json


EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
BASELINE_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_cnn_attention_results.json"
FOCUS_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_addition_focus_results.json"
OUTPUT_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_addition_focus_comparison.csv"
LABELS = ["addition", "deletion", "substitution"]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Results JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summary_row(name: str, data: dict, note: str) -> dict:
    summary = data["summary"]
    row = {
        "run": name,
        "note": note,
        "mean_accuracy": summary["mean_accuracy"],
        "std_accuracy": summary["std_accuracy"],
        "mean_macro_f1": summary["mean_macro_f1"],
        "std_macro_f1": summary["std_macro_f1"],
        "mean_weighted_f1": summary["mean_weighted_f1"],
        "std_weighted_f1": summary["std_weighted_f1"],
    }
    for label in LABELS:
        row[f"mean_{label}_f1"] = summary[f"mean_{label}_f1"]
        row[f"std_{label}_f1"] = summary[f"std_{label}_f1"]
    for fold in data["folds"]:
        speaker = fold["held_out_speaker"]
        row[f"{speaker}_macro_f1"] = fold["test"]["metrics"]["macro_f1"]
        row[f"{speaker}_addition_f1"] = fold["test"]["metrics"]["per_class"]["addition"]["f1"]
    return row


def print_delta(metric: str, baseline: dict, focus: dict) -> None:
    delta = float(focus[metric]) - float(baseline[metric])
    print(f"{metric}: baseline={float(baseline[metric]):.4f} focus={float(focus[metric]):.4f} delta={delta:+.4f}")


def main() -> None:
    baseline = load_json(BASELINE_JSON)
    focus = load_json(FOCUS_JSON)
    rows = [
        summary_row("speaker_disjoint_baseline", baseline, "Inverse-frequency sampler only."),
        summary_row(
            "speaker_disjoint_addition_focus",
            focus,
            "Inverse-frequency sampler with extra addition boost.",
        ),
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    baseline_row, focus_row = rows
    print("Saved comparison to:", OUTPUT_CSV)
    print()
    print("Aggregate comparison:")
    for metric in [
        "mean_accuracy",
        "mean_macro_f1",
        "mean_addition_f1",
        "mean_deletion_f1",
        "mean_substitution_f1",
    ]:
        print_delta(metric, baseline_row, focus_row)

    print()
    print("Per-fold addition F1:")
    for fold in ["HQTV", "PNV", "THV", "TLV"]:
        print_delta(f"{fold}_addition_f1", baseline_row, focus_row)

    macro_delta = float(focus_row["mean_macro_f1"]) - float(baseline_row["mean_macro_f1"])
    addition_delta = float(focus_row["mean_addition_f1"]) - float(baseline_row["mean_addition_f1"])

    print()
    if addition_delta > 0 and macro_delta > -0.02:
        print("Recommendation: addition-focused sampler improves addition without an excessive macro-F1 drop.")
    elif addition_delta > 0:
        print("Recommendation: addition improved, but macro-F1 tradeoff is material; keep as experiment.")
    else:
        print("Recommendation: addition-focused sampler does not improve the target metric; keep baseline.")
    print("Confidence remains classifier confidence, not pronunciation correctness.")


if __name__ == "__main__":
    main()
