from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import run_l2_arctic_correctness_r2a as r2a
import run_l2_arctic_correctness_r2b_audio_phone as r2b


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r2b_audio_phone_seed42"
CHECKPOINT_PATH = EXPERIMENT_DIR / "R2B_audio_phone_binary_correctness_seed42_best_validation_macro_f1.pt"
CHECKPOINT_SHA256 = "C80130A9BB533344521610752F8699FC1FBE684AE7FAC5258F05F5F8019ABE6C"
METADATA_CSV = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_phone_correctness_v3.csv"
DATASET_SHA256 = "433F006AB0ABCE47955C2305FCD131F2FFD9741417891BE125798163ADD28F7E"

ROW_EXPORT = EXPERIMENT_DIR / "validation_row_predictions.csv"
SUBSET_EXPORT = EXPERIMENT_DIR / "r2b2_phone_balanced_validation_rows.csv"
REPORT_JSON = EXPERIMENT_DIR / "r2b2_phone_balanced_report.json"

VALIDATION_SPEAKERS = frozenset(("ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI"))
EXPECTED_VALIDATION_ROWS = 29_767
SEED = 42
MIN_CLASS_SUPPORT = 10
MODES = ("full", "no_audio", "no_phone")
SAVED_REPORTS = {
    "full": EXPERIMENT_DIR / "selected_validation_full.json",
    "no_audio": EXPERIMENT_DIR / "selected_validation_no_audio.json",
    "no_phone": EXPERIMENT_DIR / "selected_validation_no_phone.json",
}
FOCUS_PHONES = ("DH", "Z", "F", "TH", "R", "D", "T", "M")


def load_validation_rows(audio_root: Path) -> list[dict[str, Any]]:
    if r2a.sha256_file(METADATA_CSV) != DATASET_SHA256:
        raise RuntimeError("V3 dataset SHA-256 mismatch")
    rows: list[dict[str, Any]] = []
    with METADATA_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for source_index, source in enumerate(reader):
            # Do not resolve or inspect TEST audio/phone metadata.
            if source["speaker_id"] not in VALIDATION_SPEAKERS:
                continue
            if source["error_type"] == "addition":
                continue
            if source["error_type"] not in {"correct", "substitution", "deletion"}:
                raise RuntimeError(f"Unexpected validation class at CSV row {source_index + 2}")
            start = float(source["start_time"])
            end = float(source["end_time"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise RuntimeError(f"Invalid validation interval at CSV row {source_index + 2}")
            canonical = r2b.canonicalize_phone(source["expected_phone"])
            row = dict(source)
            row["_source_index"] = source_index
            row["_start"] = start
            row["_end"] = end
            row["_duration"] = end - start
            row["_binary_label"] = 0 if source["error_type"] == "correct" else 1
            row["_canonical_phone"] = canonical
            row["_phone_id"] = r2b.PHONE_TO_ID.get(canonical, 0)
            row["_audio_path"] = r2a.resolve_audio_path(source["audio_path"], audio_root)
            rows.append(row)
    if len(rows) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(f"Validation row count differs: expected {EXPECTED_VALIDATION_ROWS}, got {len(rows)}")
    if {row["speaker_id"] for row in rows} != VALIDATION_SPEAKERS:
        raise RuntimeError("Validation speaker set differs")
    counts = Counter(row["_binary_label"] for row in rows)
    subtypes = Counter(row["error_type"] for row in rows)
    if counts != Counter({0: 25_303, 1: 4_464}) or subtypes != Counter(
        {"correct": 25_303, "substitution": 3_485, "deletion": 979}
    ):
        raise RuntimeError(f"Validation distribution differs: binary={counts}, subtype={subtypes}")
    if any(row["_phone_id"] == 0 for row in rows):
        raise RuntimeError("Unexpected <UNK> row in validation")
    return rows


def verify_saved_configuration() -> dict[str, Any]:
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    checkpoint_hash = r2a.sha256_file(CHECKPOINT_PATH)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: expected {CHECKPOINT_SHA256}, got {checkpoint_hash}")
    if config["dataset_sha256"] != DATASET_SHA256:
        raise RuntimeError("Saved R2-B config dataset SHA-256 differs")
    expected_vocab = {phone: index for index, phone in enumerate(r2b.PHONE_VOCAB)}
    if config["phone_vocabulary"] != expected_vocab:
        raise RuntimeError("Saved R2-B canonical phone vocabulary differs")
    expected_audio = {
        "sample_rate": 16_000,
        "mono": True,
        "samples": 16_000,
    }
    for key, expected in expected_audio.items():
        if config["audio"][key] != expected:
            raise RuntimeError(f"Saved audio config differs for {key}")
    expected_log_mel = {"n_mels": 64, "n_fft": 2_048, "hop_length": 512, "win_length": 2_048}
    for key, expected in expected_log_mel.items():
        if config["log_mel"][key] != expected:
            raise RuntimeError(f"Saved log-mel config differs for {key}")
    return {"checkpoint_sha256": checkpoint_hash, "dataset_sha256": config["dataset_sha256"], "config": config}


def evaluate_with_probabilities(
    model: r2b.AudioPhoneCorrectnessModel,
    dataset: r2b.ConditionedFeatureDataset,
    rows: list[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    mode: str,
) -> tuple[dict[str, Any], list[int], list[list[float]]]:
    loader = DataLoader(
        dataset,
        batch_size=r2b.BATCH_SIZE,
        shuffle=False,
        num_workers=r2b.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    seen_indexes: list[int] = []
    total_loss = 0.0
    with torch.no_grad():
        for features, targets, phone_ids, indexes in tqdm(loader, desc=f"R2-B2 validation {mode}"):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            phone_ids = phone_ids.to(device, non_blocking=True)
            logits = model(features, phone_ids, mode=mode)
            probs = torch.softmax(logits, dim=1)
            total_loss += criterion(logits, targets).item() * len(targets)
            labels.extend(targets.cpu().tolist())
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            seen_indexes.extend(indexes.tolist())
    if seen_indexes != list(range(len(rows))):
        raise RuntimeError(f"{mode} inference row order differs")
    if labels != [row["_binary_label"] for row in rows]:
        raise RuntimeError(f"{mode} inference labels differ from metadata")
    metrics = r2a.binary_metrics(labels, predictions)
    metrics["loss"] = total_loss / len(labels)
    metrics["mode"] = mode
    metrics = r2a.add_audit_metrics(metrics, rows, predictions)
    return metrics, predictions, probabilities


def compare_recursive(actual: Any, expected: Any, path: str = "metrics") -> list[str]:
    differences: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: type differs"]
        for key, expected_value in expected.items():
            if key not in actual:
                differences.append(f"{path}.{key}: missing")
            else:
                differences.extend(compare_recursive(actual[key], expected_value, f"{path}.{key}"))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [f"{path}: list shape differs"]
        for index, expected_value in enumerate(expected):
            differences.extend(compare_recursive(actual[index], expected_value, f"{path}[{index}]"))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-6):
            differences.append(f"{path}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        differences.append(f"{path}: expected {expected!r}, got {actual!r}")
    return differences


def subset_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    metrics = r2a.binary_metrics(labels, predictions)
    return {
        "rows": metrics["rows"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "incorrect_precision": metrics["incorrect"]["precision"],
        "incorrect_recall": metrics["incorrect"]["recall"],
        "incorrect_f1": metrics["incorrect"]["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def create_phone_balanced_subset(rows: list[dict[str, Any]]) -> tuple[list[int], dict[str, Any]]:
    by_phone: dict[str, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
    for index, row in enumerate(rows):
        by_phone[row["_canonical_phone"]][row["_binary_label"]].append(index)
    rng = random.Random(SEED)
    selected: list[int] = []
    included: dict[str, Any] = {}
    excluded: dict[str, Any] = {}
    for phone in sorted(by_phone):
        correct = by_phone[phone][0]
        incorrect = by_phone[phone][1]
        if len(correct) < MIN_CLASS_SUPPORT or len(incorrect) < MIN_CLASS_SUPPORT:
            reasons = []
            if len(correct) < MIN_CLASS_SUPPORT:
                reasons.append("correct_support<10")
            if len(incorrect) < MIN_CLASS_SUPPORT:
                reasons.append("incorrect_support<10")
            excluded[phone] = {
                "validation_correct": len(correct),
                "validation_incorrect": len(incorrect),
                "reason": "+".join(reasons),
            }
            continue
        count = min(len(correct), len(incorrect))
        selected_correct = rng.sample(correct, count)
        selected_incorrect = rng.sample(incorrect, count)
        selected.extend(selected_correct)
        selected.extend(selected_incorrect)
        included[phone] = {
            "validation_correct": len(correct),
            "validation_incorrect": len(incorrect),
            "selected_correct": count,
            "selected_incorrect": count,
            "selected_rows": 2 * count,
        }
    selected.sort()
    labels = Counter(rows[index]["_binary_label"] for index in selected)
    if labels != Counter({0: 4_441, 1: 4_441}) or len(selected) != 8_882:
        raise RuntimeError(f"Phone-balanced support differs: labels={labels}, rows={len(selected)}")
    return selected, {
        "seed": SEED,
        "sampling": "per canonical phone; equal correct/incorrect; without replacement",
        "minimum_support_per_class": MIN_CLASS_SUPPORT,
        "included_phone_count": len(included),
        "included_phones": included,
        "excluded_phones": excluded,
        "correct_rows": labels[0],
        "incorrect_rows": labels[1],
        "total_rows": len(selected),
    }


def per_phone_diagnostics(
    rows: list[dict[str, Any]], selected: list[int], mode_predictions: dict[str, list[int]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    included_phones = sorted({rows[index]["_canonical_phone"] for index in selected})
    per_mode: dict[str, Any] = {}
    focus: dict[str, Any] = {}
    for mode, all_predictions in mode_predictions.items():
        phone_metrics = {}
        macro_f1_values = []
        incorrect_f1_values = []
        for phone in included_phones:
            indexes = [index for index in selected if rows[index]["_canonical_phone"] == phone]
            labels = [rows[index]["_binary_label"] for index in indexes]
            predictions = [all_predictions[index] for index in indexes]
            metrics = subset_metrics(labels, predictions)
            correct_indexes = [index for index in indexes if rows[index]["_binary_label"] == 0]
            incorrect_indexes = [index for index in indexes if rows[index]["_binary_label"] == 1]
            phone_metrics[phone] = {
                **metrics,
                "selected_correct_support": len(correct_indexes),
                "selected_incorrect_support": len(incorrect_indexes),
                "correct_recall": sum(all_predictions[index] == 0 for index in correct_indexes) / len(correct_indexes),
                "incorrect_recall": sum(all_predictions[index] == 1 for index in incorrect_indexes) / len(incorrect_indexes),
            }
            macro_f1_values.append(metrics["macro_f1"])
            incorrect_f1_values.append(metrics["incorrect_f1"])
        per_mode[mode] = {
            "mean_per_phone_macro_f1": sum(macro_f1_values) / len(macro_f1_values),
            "mean_per_phone_incorrect_f1": sum(incorrect_f1_values) / len(incorrect_f1_values),
            "per_phone": phone_metrics,
        }
    for phone in FOCUS_PHONES:
        if phone not in included_phones:
            counts = Counter(row["_binary_label"] for row in rows if row["_canonical_phone"] == phone)
            focus[phone] = {
                "status": "EXCLUDED_LOW_SUPPORT",
                "validation_correct": counts[0],
                "validation_incorrect": counts[1],
            }
        else:
            focus_entry: dict[str, Any] = {"status": "INCLUDED"}
            for mode in MODES:
                focus_entry[mode] = {
                    "selected_correct_support": per_mode[mode]["per_phone"][phone]["selected_correct_support"],
                    "selected_incorrect_support": per_mode[mode]["per_phone"][phone]["selected_incorrect_support"],
                    "correct_recall": per_mode[mode]["per_phone"][phone]["correct_recall"],
                    "incorrect_recall": per_mode[mode]["per_phone"][phone]["incorrect_recall"],
                }
            focus[phone] = focus_entry
    return per_mode, focus


def metric_deltas(full: dict[str, Any], other: dict[str, Any], full_phone: dict[str, Any], other_phone: dict[str, Any]) -> dict[str, float]:
    return {
        "macro_f1": full["macro_f1"] - other["macro_f1"],
        "incorrect_f1": full["incorrect_f1"] - other["incorrect_f1"],
        "mean_per_phone_macro_f1": full_phone["mean_per_phone_macro_f1"] - other_phone["mean_per_phone_macro_f1"],
        "mean_per_phone_incorrect_f1": full_phone["mean_per_phone_incorrect_f1"] - other_phone["mean_per_phone_incorrect_f1"],
    }


def decision_from_deltas(deltas: dict[str, float]) -> str:
    macro_delta = deltas["macro_f1"]
    incorrect_delta = deltas["incorrect_f1"]
    if macro_delta >= 0.05 and incorrect_delta >= 0.05:
        return "R2B2_AUDIO_SIGNAL_CONFIRMED"
    if macro_delta < 0 or incorrect_delta < 0 or (macro_delta < 0.02 and incorrect_delta < 0.02):
        return "R2B2_AUDIO_SIGNAL_NOT_CONFIRMED"
    if (0.02 <= macro_delta < 0.05) or (0.02 <= incorrect_delta < 0.05):
        return "R2B2_AUDIO_SIGNAL_WEAK"
    return "R2B2_AUDIO_SIGNAL_NOT_CONFIRMED"


def write_row_export(
    rows: list[dict[str, Any]], predictions: dict[str, list[int]], probabilities: dict[str, list[list[float]]]
) -> None:
    fields = [
        "row_index", "source_csv_row_index", "speaker_id", "audio_reference", "utterance_id", "start_time", "end_time",
        "canonical_phone", "raw_expected_phone", "binary_label", "source_error_type",
        "full_pred", "full_prob_correct", "full_prob_incorrect",
        "no_audio_pred", "no_audio_prob_correct", "no_audio_prob_incorrect",
        "no_phone_pred", "no_phone_prob_correct", "no_phone_prob_incorrect",
    ]
    with ROW_EXPORT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = {
                "row_index": index,
                "source_csv_row_index": row["_source_index"],
                "speaker_id": row["speaker_id"],
                "audio_reference": row["audio_path"],
                "utterance_id": row["utterance_id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "canonical_phone": row["_canonical_phone"],
                "raw_expected_phone": row["expected_phone"],
                "binary_label": row["_binary_label"],
                "source_error_type": row["error_type"],
            }
            for mode in MODES:
                output[f"{mode}_pred"] = predictions[mode][index]
                output[f"{mode}_prob_correct"] = format(probabilities[mode][index][0], ".10g")
                output[f"{mode}_prob_incorrect"] = format(probabilities[mode][index][1], ".10g")
            writer.writerow(output)


def write_subset_export(rows: list[dict[str, Any]], selected: list[int]) -> None:
    fields = [
        "selection_order", "row_index", "source_csv_row_index", "speaker_id", "audio_reference", "utterance_id",
        "start_time", "end_time", "canonical_phone", "binary_label", "source_error_type",
    ]
    with SUBSET_EXPORT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for order, index in enumerate(selected):
            row = rows[index]
            writer.writerow(
                {
                    "selection_order": order,
                    "row_index": index,
                    "source_csv_row_index": row["_source_index"],
                    "speaker_id": row["speaker_id"],
                    "audio_reference": row["audio_path"],
                    "utterance_id": row["utterance_id"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "canonical_phone": row["_canonical_phone"],
                    "binary_label": row["_binary_label"],
                    "source_error_type": row["error_type"],
                }
            )


def main() -> int:
    for output in (ROW_EXPORT, SUBSET_EXPORT, REPORT_JSON):
        if output.exists():
            raise RuntimeError(f"Refusing to overwrite existing R2-B2 artifact: {output}")
    identity = verify_saved_configuration()
    audio_root = r2a.require_audio_root()
    rows = load_validation_rows(audio_root)
    audio_preflight = r2a.preflight_audio(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r2a.set_seed(SEED)
    audio_features = r2a.materialize_audio_features(rows, device, "R2-B2 VALIDATION only")
    dataset = r2b.ConditionedFeatureDataset(audio_features, rows)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if checkpoint["epoch"] != 12 or checkpoint["phone_to_id"] != r2b.PHONE_TO_ID:
        raise RuntimeError("Checkpoint epoch/vocabulary identity differs")
    model = r2b.AudioPhoneCorrectnessModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.assert_unk_is_zero()
    model.eval()
    class_weights = identity["config"]["class_weights"]
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([class_weights["correct"], class_weights["incorrect"]], dtype=torch.float32, device=device)
    )

    reproduced_metrics: dict[str, Any] = {}
    mode_predictions: dict[str, list[int]] = {}
    mode_probabilities: dict[str, list[list[float]]] = {}
    reproduction_differences: dict[str, list[str]] = {}
    for mode in MODES:
        metrics, predictions, probabilities = evaluate_with_probabilities(model, dataset, rows, criterion, device, mode)
        expected = json.loads(SAVED_REPORTS[mode].read_text(encoding="utf-8"))["metrics"]
        differences = compare_recursive(metrics, expected)
        if differences:
            reproduction_differences[mode] = differences
        reproduced_metrics[mode] = metrics
        mode_predictions[mode] = predictions
        mode_probabilities[mode] = probabilities
    if reproduction_differences:
        print(json.dumps({"status": "R2B2_REPRODUCTION_FAIL", "differences": reproduction_differences}, indent=2))
        return 2

    selected, support = create_phone_balanced_subset(rows)
    subset_mode_metrics = {}
    labels = [rows[index]["_binary_label"] for index in selected]
    for mode in MODES:
        predictions = [mode_predictions[mode][index] for index in selected]
        subset_mode_metrics[mode] = subset_metrics(labels, predictions)
    per_phone, focus = per_phone_diagnostics(rows, selected, mode_predictions)
    full_no_audio = metric_deltas(
        subset_mode_metrics["full"], subset_mode_metrics["no_audio"], per_phone["full"], per_phone["no_audio"]
    )
    full_no_phone = metric_deltas(
        subset_mode_metrics["full"], subset_mode_metrics["no_phone"], per_phone["full"], per_phone["no_phone"]
    )
    decision = decision_from_deltas(full_no_audio)

    write_row_export(rows, mode_predictions, mode_probabilities)
    write_subset_export(rows, selected)
    checkpoint_hash_after = r2a.sha256_file(CHECKPOINT_PATH)
    if checkpoint_hash_after != CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint changed during R2-B2 diagnostic")
    report = {
        "status": decision,
        "research_only": True,
        "training_performed": False,
        "checkpoint_modified": False,
        "test_accessed": False,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_sha256_before_after": [identity["checkpoint_sha256"], checkpoint_hash_after],
        "dataset_sha256": identity["dataset_sha256"],
        "device": str(device),
        "validation_rows": len(rows),
        "validation_speakers": sorted(VALIDATION_SPEAKERS),
        "validation_unique_audio_files": audio_preflight["unique_wav_files"],
        "reproduction": {
            "pass": True,
            "absolute_numeric_tolerance": 1e-6,
            "aggregate_metrics": reproduced_metrics,
        },
        "row_prediction_export": ROW_EXPORT.name,
        "subset_identity_export": SUBSET_EXPORT.name,
        "phone_balanced_support": support,
        "phone_balanced_metrics": subset_mode_metrics,
        "mean_per_phone_metrics": {
            mode: {
                "mean_per_phone_macro_f1": per_phone[mode]["mean_per_phone_macro_f1"],
                "mean_per_phone_incorrect_f1": per_phone[mode]["mean_per_phone_incorrect_f1"],
            }
            for mode in MODES
        },
        "full_minus_no_audio": full_no_audio,
        "full_minus_no_phone": full_no_phone,
        "focus_phone_diagnostics": focus,
        "per_phone_metrics": per_phone,
        "decision_thresholds": {
            "confirmed": "FULL-NO_AUDIO Macro-F1 >=0.05 AND Incorrect-F1 >=0.05",
            "weak": "positive improvement and either key delta in [0.02,0.05)",
            "not_confirmed": "both key deltas <0.02 or FULL worse than NO_AUDIO",
        },
    }
    r2a.write_json(REPORT_JSON, report)
    print(json.dumps({"status": decision, "test_accessed": False, "support": support, "metrics": subset_mode_metrics, "full_minus_no_audio": full_no_audio, "full_minus_no_phone": full_no_phone}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
