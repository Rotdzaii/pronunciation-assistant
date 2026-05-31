from __future__ import annotations

from pathlib import Path
import csv
import json


EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
BASELINE_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_cnn_attention_results.json"
CONTEXT_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_results.json"
STABILITY_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_stability_results.json"
OUTPUT_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_stability_comparison.csv"

DOCUMENTED_ADDITION_FOCUS = {
    "run": "addition_focused_sampler",
    "scope": "Mean held-out Vietnamese speaker",
    "note": "Documented previous result; full result JSON may not exist on this branch.",
    "accuracy": 0.5689,
    "accuracy_std": 0.0428,
    "macro_f1": 0.4715,
    "macro_f1_std": 0.0299,
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


def row_from_disjoint_summary(name: str, data: dict, note: str) -> dict:
    summary = data["summary"]
    return {
        "run": name,
        "scope": "Mean held-out Vietnamese speaker",
        "note": note,
        "accuracy": summary["mean_accuracy"],
        "accuracy_std": summary["std_accuracy"],
        "macro_f1": summary["mean_macro_f1"],
        "macro_f1_std": summary["std_macro_f1"],
        "addition_f1": summary["mean_addition_f1"],
        "addition_f1_std": summary["std_addition_f1"],
        "deletion_f1": summary["mean_deletion_f1"],
        "deletion_f1_std": summary["std_deletion_f1"],
        "substitution_f1": summary["mean_substitution_f1"],
        "substitution_f1_std": summary["std_substitution_f1"],
    }


def row_from_single_context(data: dict) -> dict:
    context_rows = data["summary"]["by_context_mode"]
    context_010 = next(row for row in context_rows if row["context_mode"] == "context_0_10")
    return {
        "run": "single_seed_context_0_10",
        "scope": "Seed 42, four held-out Vietnamese speakers",
        "note": "Previous context-window result before multi-seed confirmation.",
        "accuracy": context_010["mean_accuracy"],
        "accuracy_std": context_010["std_accuracy"],
        "macro_f1": context_010["mean_macro_f1"],
        "macro_f1_std": context_010["std_macro_f1"],
        "addition_f1": context_010["mean_addition_f1"],
        "addition_f1_std": context_010["std_addition_f1"],
        "deletion_f1": context_010["mean_deletion_f1"],
        "deletion_f1_std": context_010["std_deletion_f1"],
        "substitution_f1": context_010["mean_substitution_f1"],
        "substitution_f1_std": context_010["std_substitution_f1"],
    }


def row_from_stability(data: dict) -> dict:
    overall = data["summary"]["overall"]
    return {
        "run": "multi_seed_context_0_10_stability",
        "scope": "All completed seed/fold runs",
        "note": f"Seeds={data.get('seeds')}; held_out_speakers={data.get('held_out_speakers')}.",
        "accuracy": overall["mean_accuracy"],
        "accuracy_std": overall["std_accuracy"],
        "macro_f1": overall["mean_macro_f1"],
        "macro_f1_std": overall["std_macro_f1"],
        "addition_f1": overall["mean_addition_f1"],
        "addition_f1_std": overall["std_addition_f1"],
        "deletion_f1": overall["mean_deletion_f1"],
        "deletion_f1_std": overall["std_deletion_f1"],
        "substitution_f1": overall["mean_substitution_f1"],
        "substitution_f1_std": overall["std_substitution_f1"],
    }


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
    rows = [
        row_from_disjoint_summary(
            "baseline_speaker_disjoint_cnn_attention",
            load_json(BASELINE_JSON),
            "Previous speaker-disjoint baseline using original segment crop.",
        ),
        DOCUMENTED_ADDITION_FOCUS,
        row_from_single_context(load_json(CONTEXT_JSON)),
    ]

    if STABILITY_JSON.exists():
        rows.append(row_from_stability(load_json(STABILITY_JSON)))
        stability_note = "Multi-seed stability outputs found and included."
    else:
        stability_note = (
            "Multi-seed stability outputs are not present yet. Run "
            "ai-training/scripts/run_l2_arctic_vietnamese_speaker_disjoint_context_stability.py --run-full "
            "to generate them."
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved comparison to:", OUTPUT_CSV)
    print(stability_note)
    print()
    print("Macro F1 comparison:")
    for row in sorted(rows, key=lambda item: safe_float(item["macro_f1"]), reverse=True):
        print(f"- {row['run']}: {format_metric(row, 'macro_f1')}")

    print()
    print("Addition F1 comparison:")
    for row in sorted(rows, key=lambda item: safe_float(item["addition_f1"]), reverse=True):
        print(f"- {row['run']}: {format_metric(row, 'addition_f1')}")

    print()
    print("Decision rule:")
    print("Context_0_10 should replace the current candidate only if multi-seed results consistently improve")
    print("macro F1 and addition F1 without large deletion/substitution degradation.")
    print("Confidence remains classifier confidence, not pronunciation correctness.")


if __name__ == "__main__":
    main()
