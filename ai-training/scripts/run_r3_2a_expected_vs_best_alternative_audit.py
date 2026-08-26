from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_l2_arctic_observed_phone_r3_1a as r3a  # noqa: E402


REPO_ROOT = r3a.REPO_ROOT
R3D_DIR = REPO_ROOT / "ai-training/experiments/r3_1d_observed_phone_seed42_48epochs"
CHECKPOINT_PATH = R3D_DIR / "R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt"
EXPECTED_CHECKPOINT_SHA256 = "5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E"
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r3_2a_expected_vs_best_alternative"
REPRODUCTION_TOLERANCE = 1e-6
IMPROVEMENT_MACRO_F1 = 0.05
IMPROVEMENT_SUBSTITUTION_F1 = 0.10
MIN_SUBSTITUTION_RECALL = 0.50
IMPORTANT_EXPECTED_PHONES = ("TH", "DH", "R", "V", "D", "T", "S", "Z")
IMPORTANT_PAIRS = (("TH", "T"), ("DH", "D"), ("R", "L"), ("V", "W"), ("Z", "S"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(truth, predicted, labels=[0, 1]).astype(int)
    tn, fp, fn, tp = (int(item) for item in matrix.ravel())
    correct_precision = tn / (tn + fn) if tn + fn else 0.0
    correct_recall = tn / (tn + fp) if tn + fp else 0.0
    correct_f1 = 2 * correct_precision * correct_recall / (correct_precision + correct_recall) if correct_precision + correct_recall else 0.0
    substitution_precision = tp / (tp + fp) if tp + fp else 0.0
    substitution_recall = tp / (tp + fn) if tp + fn else 0.0
    substitution_f1 = (
        2 * substitution_precision * substitution_recall / (substitution_precision + substitution_recall)
        if substitution_precision + substitution_recall else 0.0
    )
    return {
        "rows": int(len(truth)),
        "accuracy": float((tn + tp) / len(truth)),
        "balanced_accuracy": float((correct_recall + substitution_recall) / 2.0),
        "macro_f1": float((correct_f1 + substitution_f1) / 2.0),
        "substitution_precision": float(substitution_precision),
        "substitution_recall": float(substitution_recall),
        "substitution_f1": float(substitution_f1),
        "confusion_matrix": matrix.tolist(),
    }


def threshold_search(score: np.ndarray, truth: np.ndarray, name: str) -> tuple[dict[str, Any], np.ndarray]:
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    sorted_truth = truth[order]
    unique_values, first_indexes, counts = np.unique(sorted_score, return_index=True, return_counts=True)
    total_positive = int(truth.sum())
    total_negative = int(len(truth) - total_positive)
    cumulative_positive = np.cumsum(sorted_truth)
    cumulative_negative = np.cumsum(1 - sorted_truth)
    candidates: list[dict[str, Any]] = []

    below_min = float(np.nextafter(unique_values[0], -np.inf))
    candidate_values = [below_min, *[float(item) for item in unique_values]]
    candidate_tp = [0]
    candidate_fp = [0]
    for start, count in zip(first_indexes, counts):
        end = int(start + count - 1)
        candidate_tp.append(int(cumulative_positive[end]))
        candidate_fp.append(int(cumulative_negative[end]))

    best_key: tuple[float, float, float, float] | None = None
    best: dict[str, Any] | None = None
    for threshold, tp, fp in zip(candidate_values, candidate_tp, candidate_fp):
        fn, tn = total_positive - tp, total_negative - fp
        correct_precision = tn / (tn + fn) if tn + fn else 0.0
        correct_recall = tn / (tn + fp) if tn + fp else 0.0
        correct_f1 = 2 * correct_precision * correct_recall / (correct_precision + correct_recall) if correct_precision + correct_recall else 0.0
        sub_precision = tp / (tp + fp) if tp + fp else 0.0
        sub_recall = tp / (tp + fn) if tp + fn else 0.0
        sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall) if sub_precision + sub_recall else 0.0
        macro_f1 = (correct_f1 + sub_f1) / 2.0
        row = {
            "threshold": threshold,
            "macro_f1": macro_f1,
            "substitution_f1": sub_f1,
            "substitution_precision": sub_precision,
            "substitution_recall": sub_recall,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }
        candidates.append(row)
        # Lower threshold means fewer substitution calls and is the registered conservative tie-break.
        key = (macro_f1, sub_f1, sub_precision, -threshold)
        if best_key is None or key > best_key:
            best_key, best = key, row
    assert best is not None
    predicted = (score <= best["threshold"]).astype(np.int64)
    selected = {
        "score": name,
        "direction": "predict substitution when score <= threshold",
        "candidate_thresholds_evaluated": len(candidates),
        "tie_break": "higher Macro-F1; higher substitution F1; higher substitution precision; lower/more-conservative threshold",
        "threshold": best["threshold"],
        "metrics": binary_metrics(truth, predicted),
    }
    curve_path = EXPERIMENT_DIR / f"{name}_threshold_curve.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    return selected, predicted


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)), "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)), "p90": float(np.percentile(values, 90)),
    }


def compare_fields(actual: dict[str, float], expected: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    deltas = {field: abs(float(actual[field]) - float(expected[field])) for field in fields}
    failures = [f"{field}: delta={delta:.12g}" for field, delta in deltas.items() if delta > REPRODUCTION_TOLERANCE]
    return {"tolerance": REPRODUCTION_TOLERANCE, "absolute_deltas": deltas, "failures": failures}


def main() -> int:
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite existing audit directory: {EXPERIMENT_DIR}")
    checkpoint_sha = sha256_file(CHECKPOINT_PATH)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha}")

    r3a.set_seed(r3a.SEED)
    audio_root = r3a.require_audio_root()
    split_rows, summary = r3a.load_and_validate_rows(audio_root)
    validation_rows = split_rows["validation"]
    audio_preflight = r3a.preflight_audio(validation_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_dataset = r3a.materialize_audio_features(validation_rows, device, "R3-2A validation")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if int(checkpoint["epoch"]) != 47:
        raise RuntimeError(f"Expected selected epoch 47, got {checkpoint['epoch']}")
    model = r3a.SmallPronunciationCNNAttention().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logits_batches: list[torch.Tensor] = []
    labels: list[int] = []
    class_weights_payload = json.loads((R3D_DIR / "class_weights.json").read_text(encoding="utf-8"))["weights"]
    class_weights = torch.tensor(
        [class_weights_payload[phone] for phone in r3a.PHONE_VOCAB], dtype=torch.float32, device=device
    )
    reproduction_criterion = nn.CrossEntropyLoss(weight=class_weights)
    reproduction_loss_sum = 0.0
    loader = DataLoader(
        validation_dataset, batch_size=r3a.BATCH_SIZE, shuffle=False,
        num_workers=r3a.NUM_WORKERS, pin_memory=device.type == "cuda",
    )
    inference_started = time.perf_counter()
    with torch.no_grad():
        for features, targets, _ in loader:
            targets_device = targets.to(device, non_blocking=True)
            output = model(features.to(device, non_blocking=True))
            # Match the locked R3 evaluator exactly: per-batch weighted CE mean,
            # multiplied by raw batch size, then divided by total rows.
            reproduction_loss_sum += reproduction_criterion(output, targets_device).item() * len(targets)
            logits_batches.append(output.cpu())
            labels.extend(targets.tolist())
    inference_seconds = time.perf_counter() - inference_started
    logits_tensor = torch.cat(logits_batches, dim=0)
    probabilities_tensor = torch.softmax(logits_tensor, dim=1)
    logits = logits_tensor.numpy().astype(np.float64)
    probabilities = probabilities_tensor.numpy().astype(np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions = np.argmax(probabilities, axis=1)
    top3 = np.argsort(-probabilities, axis=1)[:, :3]

    observed = r3a.metric_block(labels, predictions.tolist(), r3a.ALL_LABELS)
    observed.update({
        "loss": float(reproduction_loss_sum / len(labels)),
        "top3_accuracy": float(np.mean([truth in guesses for truth, guesses in zip(labels, top3.tolist())])),
    })
    expected_observed = json.loads((R3D_DIR / "selected_validation_report.json").read_text(encoding="utf-8"))["metrics"]
    observed_reproduction = compare_fields(
        observed, expected_observed,
        ("loss", "accuracy", "macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "top3_accuracy"),
    )

    truth = np.asarray([0 if row["relation"] == "correct" else 1 for row in validation_rows], dtype=np.int64)
    expected_ids = np.asarray([r3a.PHONE_TO_ID[row["expected_phone_canonical"]] for row in validation_rows], dtype=np.int64)
    hard_predictions = (predictions != expected_ids).astype(np.int64)
    hard_metrics = binary_metrics(truth, hard_predictions)
    expected_hard = json.loads((R3D_DIR / "downstream_binary_diagnostic.json").read_text(encoding="utf-8"))
    hard_reference = {
        "macro_f1": expected_hard["macro_f1"],
        "substitution_precision": expected_hard["substitution"]["precision"],
        "substitution_recall": expected_hard["substitution"]["recall"],
        "substitution_f1": expected_hard["substitution"]["f1"],
    }
    hard_reproduction = compare_fields(
        hard_metrics, hard_reference,
        ("macro_f1", "substitution_precision", "substitution_recall", "substitution_f1"),
    )
    reproduction_failures = observed_reproduction["failures"] + hard_reproduction["failures"]

    EXPERIMENT_DIR.mkdir(parents=True)
    r3a.write_json(EXPERIMENT_DIR / "config.json", {
        "experiment": "R3-2A expected-vs-best-alternative scoring audit",
        "markers": ["RESEARCH_ONLY", "VALIDATION_ONLY", "NO_TRAINING", "TEST_CLOSED"],
        "checkpoint": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_sha,
        "selected_epoch": 47,
        "validation_speakers": list(r3a.VALIDATION_SPEAKERS),
        "validation_rows": len(validation_rows),
        "scores": {
            "expected_posterior": "P(expected_phone | audio)",
            "expected_margin": "expected_logit - max(other_phone_logits)",
            "lower_indicates": "substitution",
        },
        "threshold_search": {
            "scope": "one global scalar per score on validation",
            "candidates": "below-min sentinel plus every unique observed score; substitution when score <= threshold",
            "selection": "highest Macro-F1; higher substitution F1; higher substitution precision; lower threshold",
            "per_phone_threshold": False,
            "per_speaker_threshold": False,
        },
        "improvement_gate": {
            "macro_f1_improvement": IMPROVEMENT_MACRO_F1,
            "or_substitution_f1_improvement": IMPROVEMENT_SUBSTITUTION_F1,
            "minimum_substitution_recall": MIN_SUBSTITUTION_RECALL,
        },
        "calibration_suitability_preregistered": {
            "scoring_gate_must_pass": True,
            "roc_auc_min": 0.65,
            "pr_auc_min": "max(0.20, substitution_prevalence + 0.05)",
            "every_speaker_substitution_recall_min": 0.30,
            "per_speaker_macro_f1_range_max": 0.20,
        },
        "test_audio_accessed": False,
    })
    r3a.write_json(EXPERIMENT_DIR / "environment.json", {
        **r3a.environment_payload(audio_root, device, summary["dataset_sha256"]),
        "experiment": "R3-2A", "checkpoint_sha256": checkpoint_sha,
        "inference_seconds": inference_seconds, "test_audio_accessed": False,
    })
    reproduction = {
        "status": "PASS" if not reproduction_failures else "FAIL",
        "checkpoint_sha256": checkpoint_sha,
        "observed_phone": observed_reproduction,
        "hard_argmax": hard_reproduction,
        "actual_observed_phone_metrics": observed,
        "actual_hard_argmax_metrics": hard_metrics,
        "failures": reproduction_failures,
        "test_audio_accessed": False,
    }
    r3a.write_json(EXPERIMENT_DIR / "reproduction.json", reproduction)
    r3a.write_json(EXPERIMENT_DIR / "preflight.json", {
        "dataset": summary, "validation_audio": audio_preflight, "checkpoint_sha256": checkpoint_sha,
        "validation_rows": len(validation_rows), "status": "PASS", "test_audio_accessed": False,
    })
    if reproduction_failures:
        r3a.write_json(EXPERIMENT_DIR / "final_status.json", {
            "status": "R3_2A_REPRODUCTION_FAIL", "test_opened": False, "test_eligible": False,
            "failures": reproduction_failures,
        })
        print("R3_2A_REPRODUCTION_FAIL; TEST not accessed")
        return 0

    row_indexes = np.arange(len(validation_rows))
    expected_posterior = probabilities[row_indexes, expected_ids]
    expected_logits = logits[row_indexes, expected_ids]
    alternative_logits = logits.copy()
    alternative_logits[row_indexes, expected_ids] = -np.inf
    best_alternative_ids = np.argmax(alternative_logits, axis=1)
    best_alternative_logits = alternative_logits[row_indexes, best_alternative_ids]
    best_alternative_probabilities = probabilities[row_indexes, best_alternative_ids]
    expected_margin = expected_logits - best_alternative_logits

    evidence_fields = [
        "row_index", "source_csv_row", "speaker_id", "audio_path", "utterance_id", "start_time", "end_time",
        "expected_phone_canonical", "true_observed_phone_canonical", "relation_origin", "predicted_top1_phone",
        "top1_probability", "expected_phone_probability", "best_alternative_phone", "best_alternative_probability",
        "expected_logit", "best_alternative_logit", "expected_margin",
    ]
    with (EXPERIMENT_DIR / "validation_phone_evidence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evidence_fields)
        writer.writeheader()
        for index, row in enumerate(validation_rows):
            writer.writerow({
                "row_index": index, "source_csv_row": row["_source_index"] + 2, "speaker_id": row["speaker_id"],
                "audio_path": row["audio_path"], "utterance_id": row["utterance_id"],
                "start_time": row["start_time"], "end_time": row["end_time"],
                "expected_phone_canonical": row["expected_phone_canonical"],
                "true_observed_phone_canonical": row["observed_phone_canonical"], "relation_origin": row["relation"],
                "predicted_top1_phone": r3a.PHONE_VOCAB[predictions[index]],
                "top1_probability": probabilities[index, predictions[index]],
                "expected_phone_probability": expected_posterior[index],
                "best_alternative_phone": r3a.PHONE_VOCAB[best_alternative_ids[index]],
                "best_alternative_probability": best_alternative_probabilities[index],
                "expected_logit": expected_logits[index], "best_alternative_logit": best_alternative_logits[index],
                "expected_margin": expected_margin[index],
            })

    score_arrays = {"expected_posterior": expected_posterior, "expected_margin": expected_margin}
    diagnostics: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    predictions_by_score: dict[str, np.ndarray] = {}
    for name, score in score_arrays.items():
        diagnostics[name] = {
            "roc_auc_substitution": float(roc_auc_score(truth, -score)),
            "pr_auc_substitution": float(average_precision_score(truth, -score)),
            "correct_origin": distribution(score[truth == 0]),
            "substitution_origin": distribution(score[truth == 1]),
        }
        selections[name], predictions_by_score[name] = threshold_search(score, truth, name)

    ranking = sorted(
        selections,
        key=lambda name: (
            selections[name]["metrics"]["macro_f1"], selections[name]["metrics"]["substitution_f1"],
            selections[name]["metrics"]["substitution_precision"], name,
        ),
        reverse=True,
    )
    best_score = ranking[0]
    best_predictions = predictions_by_score[best_score]
    best_metrics = selections[best_score]["metrics"]
    improvements = {
        name: {
            "macro_f1": item["metrics"]["macro_f1"] - hard_metrics["macro_f1"],
            "substitution_f1": item["metrics"]["substitution_f1"] - hard_metrics["substitution_f1"],
            "substitution_recall": item["metrics"]["substitution_recall"],
            "meaningful": bool(
                (
                    item["metrics"]["macro_f1"] - hard_metrics["macro_f1"] >= IMPROVEMENT_MACRO_F1
                    or item["metrics"]["substitution_f1"] - hard_metrics["substitution_f1"] >= IMPROVEMENT_SUBSTITUTION_F1
                )
                and item["metrics"]["substitution_recall"] >= MIN_SUBSTITUTION_RECALL
            ),
        }
        for name, item in selections.items()
    }

    phone_diagnostics: dict[str, Any] = {}
    for phone in IMPORTANT_EXPECTED_PHONES:
        positions = np.asarray([index for index, row in enumerate(validation_rows) if row["expected_phone_canonical"] == phone])
        correct_positions = positions[truth[positions] == 0]
        substitution_positions = positions[truth[positions] == 1]
        phone_diagnostics[phone] = {
            "correct_origin_count": int(len(correct_positions)),
            "substitution_origin_count": int(len(substitution_positions)),
            "median_expected_posterior_correct": float(np.median(expected_posterior[correct_positions])) if len(correct_positions) else None,
            "median_expected_posterior_substitution": float(np.median(expected_posterior[substitution_positions])) if len(substitution_positions) else None,
            "median_expected_margin_correct": float(np.median(expected_margin[correct_positions])) if len(correct_positions) else None,
            "median_expected_margin_substitution": float(np.median(expected_margin[substitution_positions])) if len(substitution_positions) else None,
        }

    substitution_pairs = Counter(
        (row["expected_phone_canonical"], row["observed_phone_canonical"])
        for row in validation_rows if row["relation"] == "substitution"
    )
    pair_diagnostics: dict[str, Any] = {}
    requested_pair_set = set(IMPORTANT_PAIRS)
    pairs_to_report = list(IMPORTANT_PAIRS)
    pairs_to_report.extend(pair for pair, _ in substitution_pairs.most_common(20) if pair not in requested_pair_set)
    for expected_phone, observed_phone in pairs_to_report:
        positions = np.asarray([
            index for index, row in enumerate(validation_rows)
            if row["relation"] == "substitution"
            and row["expected_phone_canonical"] == expected_phone
            and row["observed_phone_canonical"] == observed_phone
        ])
        pair_diagnostics[f"{expected_phone}->{observed_phone}"] = {
            "count": int(len(positions)),
            "median_expected_posterior": float(np.median(expected_posterior[positions])) if len(positions) else None,
            "median_expected_margin": float(np.median(expected_margin[positions])) if len(positions) else None,
            "requested_pair": (expected_phone, observed_phone) in requested_pair_set,
        }

    speaker_metrics: dict[str, Any] = {}
    for speaker in r3a.VALIDATION_SPEAKERS:
        positions = np.asarray([index for index, row in enumerate(validation_rows) if row["speaker_id"] == speaker])
        speaker_metrics[speaker] = binary_metrics(truth[positions], best_predictions[positions])
    speaker_macro_values = [item["macro_f1"] for item in speaker_metrics.values()]
    speaker_recall_values = [item["substitution_recall"] for item in speaker_metrics.values()]

    any_meaningful = any(item["meaningful"] for item in improvements.values())
    any_improvement = any(
        item["macro_f1"] > 0 or item["substitution_f1"] > 0 for item in improvements.values()
    )
    any_separation = any(item["roc_auc_substitution"] > 0.5 for item in diagnostics.values())
    if any_meaningful:
        status = "R3_2A_SCORING_PASS"
    elif any_improvement and any_separation:
        status = "R3_2A_SCORING_WEAK"
    else:
        status = "R3_2A_SCORING_FAIL"

    prevalence = float(np.mean(truth))
    calibration = (
        status == "R3_2A_SCORING_PASS"
        and diagnostics[best_score]["roc_auc_substitution"] >= 0.65
        and diagnostics[best_score]["pr_auc_substitution"] >= max(0.20, prevalence + 0.05)
        and min(speaker_recall_values) >= 0.30
        and max(speaker_macro_values) - min(speaker_macro_values) <= 0.20
    )
    calibration_verdict = "SUITABLE_FOR_CALIBRATION" if calibration else "NOT_YET_SUITABLE"

    r3a.write_json(EXPERIMENT_DIR / "score_diagnostics.json", diagnostics)
    r3a.write_json(EXPERIMENT_DIR / "threshold_results.json", {
        "hard_argmax": hard_metrics, "continuous": selections, "improvements": improvements,
        "selected_best_score": best_score,
    })
    r3a.write_json(EXPERIMENT_DIR / "phone_diagnostics.json", phone_diagnostics)
    r3a.write_json(EXPERIMENT_DIR / "substitution_pair_diagnostics.json", pair_diagnostics)
    r3a.write_json(EXPERIMENT_DIR / "per_speaker_metrics.json", {
        "score": best_score, "global_threshold": selections[best_score]["threshold"], "speakers": speaker_metrics,
        "macro_f1_range": max(speaker_macro_values) - min(speaker_macro_values),
    })
    r3a.write_json(EXPERIMENT_DIR / "final_status.json", {
        "status": status, "best_score": best_score,
        "calibration_suitability": calibration_verdict,
        "hard_argmax": hard_metrics, "best_continuous": selections[best_score],
        "improvement": improvements[best_score], "test_opened": False, "test_audio_accessed": False,
    })
    print(json.dumps({
        "status": status, "reproduction": "PASS", "best_score": best_score,
        "hard_argmax": hard_metrics, "scores": diagnostics, "thresholds": selections,
        "improvements": improvements, "calibration": calibration_verdict,
        "test_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
