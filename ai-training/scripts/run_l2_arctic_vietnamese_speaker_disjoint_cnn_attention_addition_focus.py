from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json

import torch

import run_l2_arctic_vietnamese_speaker_disjoint_cnn_attention as base


EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
MODEL_DIR = Path("ai-training/models")

RESULTS_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_addition_focus_results.json"
PER_FOLD_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_addition_focus_per_fold.csv"
SUMMARY_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_addition_focus_summary.csv"
PER_CLASS_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_addition_focus_per_class.csv"
CONFUSION_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_addition_focus_confusion_matrices.csv"
MISCLASSIFIED_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_addition_focus_misclassified_examples.csv"

ADDITION_SAMPLER_BOOST = 1.5
ORIGINAL_BASE_TRAINING_CONFIG = base.training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run addition-focused Vietnamese speaker-disjoint CNN Attention.")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per fold. Default: 12.")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional maximum number of folds to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print fold composition without training.")
    parser.add_argument(
        "--addition-boost",
        type=float,
        default=ADDITION_SAMPLER_BOOST,
        help="Extra sampler multiplier for addition rows after inverse-frequency weighting.",
    )
    parser.add_argument(
        "--write-from-json",
        action="store_true",
        help="Regenerate CSV outputs from the existing addition-focus results JSON without retraining.",
    )
    return parser.parse_args()


def checkpoint_path(held_out_speaker: str) -> Path:
    return MODEL_DIR / f"l2_arctic_cnn_attention_speaker_disjoint_addition_focus_{held_out_speaker}.pt"


def build_addition_focus_sampler(rows: list[dict[str, str]]) -> torch.utils.data.WeightedRandomSampler:
    counts = Counter(row["error_type"] for row in rows)
    sample_weights = []
    for row in rows:
        weight = 1.0 / counts[row["error_type"]]
        if row["error_type"] == "addition":
            weight *= build_addition_focus_sampler.addition_boost
        sample_weights.append(weight)

    return torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


build_addition_focus_sampler.addition_boost = ADDITION_SAMPLER_BOOST


def apply_output_paths() -> None:
    base.RESULTS_JSON = RESULTS_JSON
    base.PER_FOLD_CSV = PER_FOLD_CSV
    base.SUMMARY_CSV = SUMMARY_CSV
    base.PER_CLASS_CSV = PER_CLASS_CSV
    base.CONFUSION_CSV = CONFUSION_CSV
    base.MISCLASSIFIED_CSV = MISCLASSIFIED_CSV
    base.checkpoint_path = checkpoint_path
    base.build_sampler = build_addition_focus_sampler


def training_config(epochs: int, addition_boost: float) -> dict:
    config = ORIGINAL_BASE_TRAINING_CONFIG(epochs)
    config.update(
        {
            "sampler": "inverse_frequency_with_addition_boost",
            "addition_sampler_boost": addition_boost,
            "loss": "cross_entropy_unweighted",
            "architecture": "same_cnn_attention",
        }
    )
    return config


def main() -> None:
    args = parse_args()
    apply_output_paths()
    build_addition_focus_sampler.addition_boost = args.addition_boost

    if args.write_from_json:
        if not RESULTS_JSON.exists():
            raise FileNotFoundError(f"Results JSON not found: {RESULTS_JSON}")
        results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        base.write_outputs(results)
        print("Regenerated addition-focus CSV outputs from:", RESULTS_JSON)
        for path in [PER_FOLD_CSV, SUMMARY_CSV, PER_CLASS_CSV, CONFUSION_CSV, MISCLASSIFIED_CSV]:
            print(f"- {path}")
        return

    base.set_seed(base.RANDOM_SEED)
    rows = [row for row in base.read_rows() if base.valid_segment(row)]
    speakers = base.VIETNAMESE_SPEAKERS[: args.max_folds] if args.max_folds else base.VIETNAMESE_SPEAKERS
    label_to_index = {label: index for index, label in enumerate(base.LABEL_ORDER)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    config = training_config(args.epochs, args.addition_boost)

    print("Addition-focused Vietnamese speaker-disjoint CNN Attention")
    print("Metadata:", base.METADATA_CSV)
    print("Speakers:", speakers)
    print("Training config:", config)
    print("Note: confidence is model confidence, not pronunciation correctness.")
    base.print_gpu_telemetry("before folds")

    if args.dry_run:
        base.print_dry_run(rows, speakers)
        print("Addition sampler boost:", args.addition_boost)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("Using device:", device)

    base.training_config = lambda epochs: training_config(epochs, args.addition_boost)

    try:
        folds = []
        for speaker in speakers:
            folds.append(base.run_fold(speaker, rows, args.epochs, label_to_index, index_to_label, device))
            base.print_gpu_telemetry(f"after fold {speaker}")
    finally:
        base.training_config = ORIGINAL_BASE_TRAINING_CONFIG

    summary = base.build_summary(folds)
    results = {
        "metadata_csv": str(base.METADATA_CSV),
        "protocol": {
            "held_out_speakers": speakers,
            "train_rule": "Rows where speaker_id != held_out_speaker and split == train.",
            "validation_rule": "Rows where speaker_id != held_out_speaker and split == val.",
            "test_rule": "All rows where speaker_id == held_out_speaker.",
            "non_vietnamese_training": "Included through original train/validation splits when not held out.",
        },
        "label_order": base.LABEL_ORDER,
        "config": config,
        "note": "Confidence is classifier confidence, not pronunciation correctness.",
        "folds": folds,
        "summary": summary,
    }
    base.write_outputs(results)

    print()
    print("Addition-focused speaker-disjoint summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("Generated files:")
    for path in [RESULTS_JSON, PER_FOLD_CSV, SUMMARY_CSV, PER_CLASS_CSV, CONFUSION_CSV, MISCLASSIFIED_CSV]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
