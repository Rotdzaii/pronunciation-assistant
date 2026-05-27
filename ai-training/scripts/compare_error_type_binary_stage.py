from pathlib import Path
import csv
import json


EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")

RUNS = [
    {
        "name": "baseline",
        "path": EVALUATION_DIR / "error_type_eval_metrics.json",
        "note": "Original 3-class CNN baseline.",
    },
    {
        "name": "v2",
        "path": EVALUATION_DIR / "v2_error_type_eval_metrics.json",
        "note": "V2 3-class CNN with imbalance handling.",
    },
    {
        "name": "sampler_only",
        "path": EVALUATION_DIR / "sampler_error_type_eval_metrics.json",
        "note": "Clean v2 3-class CNN with WeightedRandomSampler only.",
    },
    {
        "name": "binary_stage_pipeline",
        "path": EVALUATION_DIR / "binary_stage_pipeline_eval_metrics.json",
        "note": "Stage 1 addition/non_addition, Stage 2 deletion/substitution.",
    },
]

OUTPUT_CSV = EVALUATION_DIR / "binary_stage_comparison.csv"
LABELS = ["addition", "deletion", "substitution"]


def load_json(path: Path):
    if not path.exists():
        print(f"Skipping missing metrics: {path}")
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_split_metrics(data: dict, split_name: str):
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


def get_metric(split_metrics: dict, key: str):
    aliases = {
        "accuracy": ["accuracy", "acc"],
        "macro_f1": ["macro_f1", "macro_f1_score"],
        "weighted_f1": ["weighted_f1", "weighted_f1_score"],
    }
    for candidate in aliases.get(key, [key]):
        if candidate in split_metrics:
            return split_metrics[candidate]
    return None


def get_class_f1(split_metrics: dict, label: str):
    for per_class_key in ["per_class", "per_class_metrics"]:
        per_class = split_metrics.get(per_class_key)
        if isinstance(per_class, dict):
            class_metrics = per_class.get(label)
            if isinstance(class_metrics, dict):
                return class_metrics.get("f1")
    return None


def extract_metrics(run_name: str, data: dict, note: str):
    val_metrics = get_split_metrics(data, "validation")
    test_metrics = get_split_metrics(data, "test")
    row = {
        "run": run_name,
        "note": note,
        "val_accuracy": get_metric(val_metrics, "accuracy"),
        "val_macro_f1": get_metric(val_metrics, "macro_f1"),
        "val_weighted_f1": get_metric(val_metrics, "weighted_f1"),
        "test_accuracy": get_metric(test_metrics, "accuracy"),
        "test_macro_f1": get_metric(test_metrics, "macro_f1"),
        "test_weighted_f1": get_metric(test_metrics, "weighted_f1"),
    }
    for label in LABELS:
        row[f"val_{label}_f1"] = get_class_f1(val_metrics, label)
        row[f"test_{label}_f1"] = get_class_f1(test_metrics, label)
    return row


def safe_float(value):
    if value is None:
        return -1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def print_ranking(rows: list[dict], key: str, title: str):
    ranked = sorted(rows, key=lambda row: safe_float(row[key]), reverse=True)
    print()
    print(title)
    for index, row in enumerate(ranked, start=1):
        print(f"{index}. {row['run']} - {key}={row[key]}")


def main():
    rows = []
    for run in RUNS:
        data = load_json(run["path"])
        if data is not None:
            rows.append(extract_metrics(run["name"], data, run["note"]))

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
