from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json

import numpy as np
import torch

import run_l2_arctic_vietnamese_speaker_disjoint_cnn_attention_context as context_runner


EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
RESULTS_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_stability_results.json"
PER_SEED_FOLD_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_stability_per_seed_fold.csv"
SUMMARY_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_stability_summary.csv"
PER_CLASS_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_stability_per_class.csv"

CONTEXT_MODE = "context_0_10"
DEFAULT_SEEDS = [42, 123, 2026]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed stability check for Vietnamese speaker-disjoint CNN Attention context_0_10."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Random seeds to evaluate.")
    parser.add_argument("--max-seeds", type=int, default=None, help="Optional maximum number of seeds to run.")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional maximum number of Vietnamese folds to run.")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per seed/fold. Default: 12.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned seed/fold jobs without training.")
    parser.add_argument(
        "--run-full",
        action="store_true",
        help="Opt in to training. Without this flag the script performs a dry run.",
    )
    parser.add_argument(
        "--write-from-json",
        action="store_true",
        help="Regenerate CSV outputs from the existing stability results JSON without retraining.",
    )
    return parser.parse_args()


def selected_seeds(seeds: list[int], max_seeds: int | None) -> list[int]:
    return seeds[:max_seeds] if max_seeds else seeds


def selected_speakers(max_folds: int | None) -> list[str]:
    speakers = context_runner.VIETNAMESE_SPEAKERS
    return speakers[:max_folds] if max_folds else speakers


def checkpoint_path(seed: int, held_out_speaker: str) -> Path:
    return (
        context_runner.MODEL_DIR
        / f"l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_{seed}_{held_out_speaker}.pt"
    )


def install_stability_checkpoint_path(seed: int) -> None:
    def _checkpoint_path(_context_mode: str, held_out_speaker: str) -> Path:
        return checkpoint_path(seed, held_out_speaker)

    context_runner.checkpoint_path = _checkpoint_path


def metric_values(folds: list[dict], path: list[str]) -> list[float]:
    values = []
    for fold in folds:
        current = fold
        for key in path:
            current = current[key]
        values.append(float(current))
    return values


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


def summarize_group(folds: list[dict], group_name: str, group_value: str | int) -> dict:
    row = {"group": group_name, "value": group_value, "run_count": len(folds)}
    for output_name, path in [
        ("accuracy", ["test", "metrics", "accuracy"]),
        ("macro_f1", ["test", "metrics", "macro_f1"]),
        ("weighted_f1", ["test", "metrics", "weighted_f1"]),
    ]:
        mean_value, std_value = mean_std(metric_values(folds, path))
        row[f"mean_{output_name}"] = mean_value
        row[f"std_{output_name}"] = std_value

    for label in context_runner.LABEL_ORDER:
        mean_value, std_value = mean_std(metric_values(folds, ["test", "metrics", "per_class", label, "f1"]))
        row[f"mean_{label}_f1"] = mean_value
        row[f"std_{label}_f1"] = std_value
    return row


def build_summary(folds: list[dict]) -> dict:
    overall = summarize_group(folds, "overall", "all_seed_fold_runs")
    by_seed = []
    by_speaker = []

    for seed in sorted({fold["seed"] for fold in folds}):
        by_seed.append(summarize_group([fold for fold in folds if fold["seed"] == seed], "seed", seed))

    for speaker in context_runner.VIETNAMESE_SPEAKERS:
        speaker_folds = [fold for fold in folds if fold["held_out_speaker"] == speaker]
        if speaker_folds:
            by_speaker.append(summarize_group(speaker_folds, "held_out_speaker", speaker))

    single_seed_context = load_single_seed_context_summary()
    comparison = {}
    if single_seed_context:
        comparison = {
            "single_seed_context_mode": single_seed_context["context_mode"],
            "single_seed_macro_f1": single_seed_context["mean_macro_f1"],
            "single_seed_addition_f1": single_seed_context["mean_addition_f1"],
            "stability_minus_single_seed_macro_f1": overall["mean_macro_f1"] - single_seed_context["mean_macro_f1"],
            "stability_minus_single_seed_addition_f1": overall["mean_addition_f1"]
            - single_seed_context["mean_addition_f1"],
        }

    return {
        "overall": overall,
        "by_seed": by_seed,
        "by_held_out_speaker": by_speaker,
        "comparison_against_single_seed_context_0_10": comparison,
    }


def load_single_seed_context_summary() -> dict | None:
    path = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_summary.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if row.get("context_mode") == CONTEXT_MODE:
            return {
                key: float(value) if key not in {"context_mode", "best_fold", "worst_fold"} else value
                for key, value in row.items()
            }
    return None


def write_outputs(results: dict) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with PER_SEED_FOLD_CSV.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "seed",
            "held_out_speaker",
            "context_mode",
            "train_rows",
            "validation_rows",
            "test_rows",
            "best_epoch",
            "best_val_macro_f1",
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1",
            *[f"test_{label}_f1" for label in context_runner.LABEL_ORDER],
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for fold in results["folds"]:
            metrics = fold["test"]["metrics"]
            row = {
                "seed": fold["seed"],
                "held_out_speaker": fold["held_out_speaker"],
                "context_mode": fold["context_mode"],
                "train_rows": fold["train_rows"],
                "validation_rows": fold["validation_rows"],
                "test_rows": fold["test_rows"],
                "best_epoch": fold["best_epoch"],
                "best_val_macro_f1": fold["best_val_macro_f1"],
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_weighted_f1": metrics["weighted_f1"],
            }
            for label in context_runner.LABEL_ORDER:
                row[f"test_{label}_f1"] = metrics["per_class"][label]["f1"]
            writer.writerow(row)

    summary_rows = [results["summary"]["overall"], *results["summary"]["by_seed"], *results["summary"]["by_held_out_speaker"]]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with PER_CLASS_CSV.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "seed",
            "held_out_speaker",
            "context_mode",
            "error_type",
            "class_id",
            "support",
            "predicted_count",
            "correct",
            "precision",
            "recall",
            "f1",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for fold in results["folds"]:
            for label, metrics in fold["test"]["metrics"]["per_class"].items():
                writer.writerow(
                    {
                        "seed": fold["seed"],
                        "held_out_speaker": fold["held_out_speaker"],
                        "context_mode": fold["context_mode"],
                        "error_type": label,
                        **metrics,
                    }
                )


def print_plan(rows: list[dict], seeds: list[int], speakers: list[str], epochs: int) -> None:
    print("Dry run: no training will be performed.")
    print("Context mode:", CONTEXT_MODE)
    print("Seeds:", seeds)
    print("Held-out speakers:", speakers)
    print("Epochs per seed/fold:", epochs)
    print("Total planned training jobs:", len(seeds) * len(speakers))
    print("Protocol: train on original train rows from all non-held-out speakers; validate on original val rows")
    print("from all non-held-out speakers; test on all rows from the held-out Vietnamese speaker.")

    for seed in seeds:
        print()
        print(f"Seed {seed}")
        for speaker in speakers:
            train_rows, val_rows, test_rows = context_runner.build_fold_rows(rows, speaker)
            print(
                f"- {speaker}: train_rows={len(train_rows)} val_rows={len(val_rows)} "
                f"test_rows={len(test_rows)} checkpoint={checkpoint_path(seed, speaker)}"
            )


def main() -> None:
    args = parse_args()
    seeds = selected_seeds(args.seeds, args.max_seeds)
    speakers = selected_speakers(args.max_folds)

    if args.write_from_json:
        if not RESULTS_JSON.exists():
            raise FileNotFoundError(f"Results JSON not found: {RESULTS_JSON}")
        results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        write_outputs(results)
        print("Regenerated CSV outputs from:", RESULTS_JSON)
        return

    rows = [row for row in context_runner.read_rows() if context_runner.valid_segment(row)]
    should_train = args.run_full and not args.dry_run

    print("Vietnamese speaker-disjoint context_0_10 stability check")
    print("Metadata:", context_runner.METADATA_CSV)
    print("Note: confidence is model confidence, not pronunciation correctness.")
    context_runner.print_gpu_telemetry("before stability plan")

    if not should_train:
        print_plan(rows, seeds, speakers, args.epochs)
        print()
        print("To run full training, add --run-full and omit --dry-run.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("Using device:", device)

    label_to_index = {label: index for index, label in enumerate(context_runner.LABEL_ORDER)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    folds = []

    for seed in seeds:
        context_runner.RANDOM_SEED = seed
        install_stability_checkpoint_path(seed)
        for speaker in speakers:
            fold = context_runner.run_fold(
                CONTEXT_MODE,
                speaker,
                rows,
                args.epochs,
                label_to_index,
                index_to_label,
                device,
            )
            fold["seed"] = seed
            folds.append(fold)
            context_runner.print_gpu_telemetry(f"after seed {seed} fold {speaker}")

    results = {
        "metadata_csv": str(context_runner.METADATA_CSV),
        "context_mode": CONTEXT_MODE,
        "context_seconds": context_runner.CONTEXT_MODES[CONTEXT_MODE],
        "seeds": seeds,
        "held_out_speakers": speakers,
        "label_order": context_runner.LABEL_ORDER,
        "config": context_runner.training_config(args.epochs, [CONTEXT_MODE]),
        "note": "Confidence is classifier confidence, not pronunciation correctness.",
        "folds": folds,
        "summary": build_summary(folds),
    }
    write_outputs(results)

    overall = results["summary"]["overall"]
    print()
    print("Context stability overall summary:")
    print(f"mean_macro_f1={overall['mean_macro_f1']:.4f} +/- {overall['std_macro_f1']:.4f}")
    print(f"mean_addition_f1={overall['mean_addition_f1']:.4f} +/- {overall['std_addition_f1']:.4f}")
    print(f"mean_accuracy={overall['mean_accuracy']:.4f} +/- {overall['std_accuracy']:.4f}")
    print("Generated files:")
    for path in [RESULTS_JSON, PER_SEED_FOLD_CSV, SUMMARY_CSV, PER_CLASS_CSV]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
