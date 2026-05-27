from pathlib import Path
import csv
import json


EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
OUTPUT_CSV = EVALUATION_DIR / "wav2vec2_attention_comparison.csv"
LABELS = ["addition", "deletion", "substitution"]

RUNS = [
    ("baseline", EVALUATION_DIR / "error_type_eval_metrics.json", "Original 3-class CNN baseline."),
    ("v2", EVALUATION_DIR / "v2_error_type_eval_metrics.json", "Best CNN V2 run."),
    ("sampler_only", EVALUATION_DIR / "sampler_error_type_eval_metrics.json", "Sampler-only CNN run."),
    ("binary_stage_pipeline", EVALUATION_DIR / "binary_stage_pipeline_eval_metrics.json", "Binary-stage CNN pipeline."),
    ("wav2vec2_encoder", EVALUATION_DIR / "wav2vec2_encoder_error_type_eval_metrics.json", "Frozen Wav2Vec2 mean pooling on original segment."),
    ("wav2vec2_attention", EVALUATION_DIR / "wav2vec2_attention_eval_metrics.json", "Frozen Wav2Vec2 context 0.15s with attention pooling."),
]


def load_json(path: Path):
    if not path.exists():
        print(f"Skipping missing metrics: {path}")
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def split_metrics(data: dict, split_name: str):
    if split_name == "validation":
        if "validation" in data:
            return data["validation"]
        if "val" in data:
            return data["val"]
        if "splits" in data and "val" in data["splits"]:
            return data["splits"]["val"].get("metrics", data["splits"]["val"])
    if split_name == "test":
        if "test" in data:
            return data["test"]
        if "splits" in data and "test" in data["splits"]:
            return data["splits"]["test"].get("metrics", data["splits"]["test"])
    return {}


def get_metric(metrics: dict, key: str):
    aliases = {
        "accuracy": ["accuracy", "acc"],
        "macro_f1": ["macro_f1", "macro_f1_score"],
        "weighted_f1": ["weighted_f1", "weighted_f1_score"],
    }
    for candidate in aliases.get(key, [key]):
        if candidate in metrics:
            return metrics[candidate]
    return None


def class_f1(metrics: dict, label: str):
    for key in ["per_class", "per_class_metrics"]:
        per_class = metrics.get(key)
        if isinstance(per_class, dict) and isinstance(per_class.get(label), dict):
            return per_class[label].get("f1")
    return None


def extract_row(name: str, data: dict, note: str):
    val = split_metrics(data, "validation")
    test = split_metrics(data, "test")
    row = {
        "run": name,
        "note": note,
        "val_accuracy": get_metric(val, "accuracy"),
        "val_macro_f1": get_metric(val, "macro_f1"),
        "val_weighted_f1": get_metric(val, "weighted_f1"),
        "test_accuracy": get_metric(test, "accuracy"),
        "test_macro_f1": get_metric(test, "macro_f1"),
        "test_weighted_f1": get_metric(test, "weighted_f1"),
    }
    for label in LABELS:
        row[f"val_{label}_f1"] = class_f1(val, label)
        row[f"test_{label}_f1"] = class_f1(test, label)
    return row


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def print_ranking(rows: list[dict], key: str, title: str):
    print()
    print(title)
    for index, row in enumerate(sorted(rows, key=lambda item: safe_float(item[key]), reverse=True), start=1):
        print(f"{index}. {row['run']} - {key}={row[key]}")


def main():
    rows = []
    context_data = load_json(EVALUATION_DIR / "wav2vec2_context_eval_metrics.json")

    for name, path, note in RUNS:
        data = load_json(path)
        if data is None and name == "wav2vec2_encoder" and context_data is not None:
            data = context_data.get("runs", {}).get("original_segment")
        if data is not None:
            rows.append(extract_row(name, data, note))

    if context_data is not None:
        for crop_mode, run in context_data.get("runs", {}).items():
            if crop_mode == "original_segment":
                run_name = "wav2vec2_context_original_segment"
            else:
                run_name = f"wav2vec2_{crop_mode}"
            rows.append(extract_row(run_name, run, "Frozen Wav2Vec2 context-window mean pooling."))

    if not rows:
        raise RuntimeError("No metric files found to compare.")

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
        print("test_accuracy:", row["test_accuracy"])
        print("test_macro_f1:", row["test_macro_f1"])
        print("test_addition_f1:", row["test_addition_f1"])
        print("test_deletion_f1:", row["test_deletion_f1"])
        print("test_substitution_f1:", row["test_substitution_f1"])
    print_ranking(rows, "test_macro_f1", "Ranking by test_macro_f1:")
    print_ranking(rows, "test_addition_f1", "Ranking by test_addition_f1:")


if __name__ == "__main__":
    main()
