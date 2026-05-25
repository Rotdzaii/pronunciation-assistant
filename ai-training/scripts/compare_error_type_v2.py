from __future__ import annotations

from pathlib import Path
import csv
import json


BASELINE_METRICS_JSON = Path("ai-training/datasets/l2-arctic/evaluation/error_type_eval_metrics.json")
V2_METRICS_JSON = Path("ai-training/datasets/l2-arctic/evaluation/v2_error_type_eval_metrics.json")
OUTPUT_CSV = Path("ai-training/datasets/l2-arctic/evaluation/error_type_v2_comparison.csv")

CLASSES = ("addition", "deletion", "substitution")


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def split_metric(metrics: dict, split: str, key: str) -> float:
    return metrics["splits"][split]["metrics"][key]


def class_f1(metrics: dict, split: str, error_type: str) -> float:
    return metrics["splits"][split]["metrics"]["per_class"][error_type]["f1"]


def add_row(rows: list[dict[str, str]], metric: str, baseline: float, v2: float) -> None:
    rows.append(
        {
            "metric": metric,
            "baseline": baseline,
            "v2": v2,
            "delta": v2 - baseline,
            "improved": v2 > baseline,
        }
    )


def main():
    baseline = load_metrics(BASELINE_METRICS_JSON)
    v2 = load_metrics(V2_METRICS_JSON)

    rows = []
    for split in ("val", "test"):
        add_row(
            rows,
            f"{split}_accuracy",
            split_metric(baseline, split, "accuracy"),
            split_metric(v2, split, "accuracy"),
        )
        add_row(
            rows,
            f"{split}_macro_f1",
            split_metric(baseline, split, "macro_f1"),
            split_metric(v2, split, "macro_f1"),
        )

    for error_type in CLASSES:
        add_row(
            rows,
            f"val_{error_type}_f1",
            class_f1(baseline, "val", error_type),
            class_f1(v2, "val", error_type),
        )
        add_row(
            rows,
            f"test_{error_type}_f1",
            class_f1(baseline, "test", error_type),
            class_f1(v2, "test", error_type),
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["metric", "baseline", "v2", "delta", "improved"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Comparison:")
    for row in rows:
        print(
            f"- {row['metric']}: "
            f"baseline={row['baseline']:.4f} "
            f"v2={row['v2']:.4f} "
            f"delta={row['delta']:+.4f}"
        )
    print(f"Saved comparison to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
