from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_l2_arctic_deletion_r4_1 as r4  # noqa: E402
import run_l2_arctic_observed_phone_r3_1a as r3  # noqa: E402


REPO_ROOT = r3.REPO_ROOT
R3D_DIR = REPO_ROOT / "ai-training/experiments/r3_1d_observed_phone_seed42_48epochs"
CHECKPOINT_PATH = R3D_DIR / "R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt"
EXPECTED_CHECKPOINT_SHA256 = "5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E"
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r4_2a_frozen_r3_expected_evidence"
IMPORTANT_PHONES = ("D", "T", "R", "L", "N", "Z", "V", "K", "AH", "HH", "S")
DTRL = frozenset(("D", "T", "R", "L"))
MIN_TRAIN_RECALL = 0.45
REPRODUCTION_TOLERANCE = 1e-6

FULL_REFERENCES = {
    "duration": {"macro_f1": 0.668146, "balanced_accuracy": 0.706301, "deletion_f1": 0.364164},
    "r4_1_full": {"macro_f1": 0.657336, "balanced_accuracy": 0.683828, "deletion_f1": 0.341612},
}
MATCHED_REFERENCES = {
    "duration": {"macro_f1": 0.485019, "balanced_accuracy": 0.499361},
    "r4_1_full": {"macro_f1": 0.603263, "balanced_accuracy": 0.622123, "deletion_f1": 0.516762},
}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)), "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)), "p90": float(np.percentile(values, 90)),
    }


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(truth, predicted, labels=[0, 1]).astype(int)
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    non_precision = tn / (tn + fn) if tn + fn else 0.0
    non_recall = tn / (tn + fp) if tn + fp else 0.0
    non_f1 = 2 * non_precision * non_recall / (non_precision + non_recall) if non_precision + non_recall else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": int(len(truth)), "non_deletion_support": int(np.sum(1 - truth)),
        "deletion_support": int(np.sum(truth)), "accuracy": float((tn + tp) / len(truth)),
        "balanced_accuracy": float((non_recall + recall) / 2), "macro_f1": float((non_f1 + f1) / 2),
        "deletion_precision": float(precision), "deletion_recall": float(recall), "deletion_f1": float(f1),
        "confusion_matrix": matrix.tolist(),
    }


def binary_metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, Any]:
    truth = np.r_[np.zeros(tn + fp, dtype=np.int8), np.ones(fn + tp, dtype=np.int8)]
    predicted = np.r_[np.zeros(tn, dtype=np.int8), np.ones(fp, dtype=np.int8),
                      np.zeros(fn, dtype=np.int8), np.ones(tp, dtype=np.int8)]
    return binary_metrics(truth, predicted)


def score_diagnostics(score: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    return {
        "roc_auc_deletion": float(roc_auc_score(truth, -score)),
        "pr_auc_deletion": float(average_precision_score(truth, -score)),
        "non_deletion": distribution(score[truth == 0]),
        "deletion": distribution(score[truth == 1]),
    }


def fit_recall_constrained_threshold(score: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    order = np.argsort(score, kind="stable")
    values, labels = score[order], truth[order]
    cumulative_positive = np.cumsum(labels)
    cumulative_negative = np.cumsum(1 - labels)
    ends = np.flatnonzero(np.r_[values[1:] != values[:-1], True])
    total_positive = int(np.sum(truth))
    total_negative = len(truth) - total_positive
    best_key: tuple[float, ...] | None = None
    best: dict[str, Any] | None = None
    feasible = 0
    for end in ends:
        threshold = float(values[end])
        tp, fp = int(cumulative_positive[end]), int(cumulative_negative[end])
        metrics = binary_metrics_from_counts(total_negative - fp, fp, total_positive - tp, tp)
        if metrics["deletion_recall"] + 1e-15 < MIN_TRAIN_RECALL:
            continue
        feasible += 1
        # Lower threshold predicts fewer deletions and is the conservative final tie-break.
        key = (metrics["macro_f1"], metrics["deletion_f1"], metrics["deletion_precision"], -threshold)
        if best_key is None or key > best_key:
            best_key = key
            best = {"threshold": threshold, "train_metrics": metrics}
    if best is None:
        raise RuntimeError("No TRAIN threshold satisfies deletion recall >= 0.45")
    return {
        **best, "candidate_thresholds": int(len(ends)), "feasible_thresholds": feasible,
        "rule": "expected_margin <= threshold => deletion",
        "selection": "TRAIN only: recall >=0.45; max Macro-F1; deletion F1; precision; lower threshold",
    }


def apply_threshold(score: np.ndarray, threshold: float) -> np.ndarray:
    return (score <= threshold).astype(np.int64)


def infer(rows: list[dict[str, Any]], model: torch.nn.Module, device: torch.device, name: str) -> dict[str, Any]:
    dataset = r3.materialize_audio_features(rows, device, name)
    loader = DataLoader(dataset, batch_size=r3.BATCH_SIZE, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    batches: list[torch.Tensor] = []
    with torch.no_grad():
        for features, _, _ in loader:
            batches.append(model(features.to(device, non_blocking=True)).cpu())
    logits_tensor = torch.cat(batches, dim=0)
    probabilities = torch.softmax(logits_tensor, dim=1).numpy().astype(np.float64)
    logits = logits_tensor.numpy().astype(np.float64)
    expected_ids = np.asarray([r3.PHONE_TO_ID[row["expected_phone_canonical"]] for row in rows], dtype=np.int64)
    indexes = np.arange(len(rows))
    expected_logits = logits[indexes, expected_ids]
    expected_posterior = probabilities[indexes, expected_ids]
    alternatives = logits.copy()
    alternatives[indexes, expected_ids] = -np.inf
    best_ids = np.argmax(alternatives, axis=1)
    best_logits = alternatives[indexes, best_ids]
    return {
        "logits": logits, "predicted_ids": np.argmax(logits, axis=1),
        "expected_posterior": expected_posterior, "expected_logit": expected_logits,
        "best_ids": best_ids, "best_logits": best_logits, "margin": expected_logits - best_logits,
    }


def subset_report(rows: list[dict[str, Any]], truth: np.ndarray, posterior: np.ndarray, margin: np.ndarray,
                  positions: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    result = {
        "rows": int(len(positions)),
        "expected_posterior": score_diagnostics(posterior[positions], truth[positions]),
        "expected_margin": score_diagnostics(margin[positions], truth[positions]),
    }
    if threshold is not None:
        result["thresholded_expected_margin"] = binary_metrics(
            truth[positions], apply_threshold(margin[positions], threshold)
        )
    return result


def save_evidence(path: Path, rows: list[dict[str, Any]], evidence: dict[str, Any], threshold: float,
                  matched_sources: set[int] | None = None) -> None:
    fields = [
        "source_csv_row", "speaker_id", "audio_path", "utterance_id", "start_time", "end_time",
        "expected_phone_canonical", "relation_origin", "binary_target", "expected_posterior", "expected_logit",
        "best_alternative_phone", "best_alternative_logit", "expected_margin", "frozen_threshold_prediction",
        "is_strict_matched_row",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                "source_csv_row": row["_source_index"] + 2, "speaker_id": row["speaker_id"],
                "audio_path": row["audio_path"], "utterance_id": row["utterance_id"],
                "start_time": row["start_time"], "end_time": row["end_time"],
                "expected_phone_canonical": row["expected_phone_canonical"], "relation_origin": row["relation"],
                "binary_target": row["_target"], "expected_posterior": evidence["expected_posterior"][index],
                "expected_logit": evidence["expected_logit"][index],
                "best_alternative_phone": r3.PHONE_VOCAB[evidence["best_ids"][index]],
                "best_alternative_logit": evidence["best_logits"][index], "expected_margin": evidence["margin"][index],
                "frozen_threshold_prediction": int(evidence["margin"][index] <= threshold),
                "is_strict_matched_row": bool(matched_sources and row["_source_index"] in matched_sources),
            })


def main() -> int:
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite audit directory: {EXPERIMENT_DIR}")
    checkpoint_sha = r3.sha256_file(CHECKPOINT_PATH)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA mismatch: {checkpoint_sha}")
    r3.set_seed(r3.SEED)
    audio_root = r3.require_audio_root()
    split_rows, source_summary, matched_sources = r4.load_and_verify(audio_root)
    train_rows, validation_rows = split_rows["train"], split_rows["validation"]
    # This is deliberately the only audio scope. TEST paths were not resolved by load_and_verify.
    # Use R3's 0.50-second preflight semantics, not R4-1's 0.30-second audit window.
    audio_preflight = r3.preflight_audio(train_rows + validation_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if int(checkpoint["epoch"]) != 47:
        raise RuntimeError(f"Expected frozen epoch 47, got {checkpoint['epoch']}")
    model = r3.SmallPronunciationCNNAttention().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    started = time.perf_counter()
    train_evidence = infer(train_rows, model, device, "R4-2A TRAIN frozen R3")
    validation_evidence = infer(validation_rows, model, device, "R4-2A VALIDATION frozen R3")
    inference_seconds = time.perf_counter() - started

    # Reproduce R3-1D selected validation acoustic metrics on the exact correct+substitution subset.
    phone_positions = np.asarray([i for i, row in enumerate(validation_rows) if row["relation"] != "deletion"])
    phone_truth = [r3.PHONE_TO_ID[validation_rows[i]["observed_phone_canonical"]] for i in phone_positions]
    phone_predicted = validation_evidence["predicted_ids"][phone_positions].tolist()
    reproduced = r3.metric_block(phone_truth, phone_predicted, r3.ALL_LABELS)
    reference = json.loads((R3D_DIR / "selected_validation_report.json").read_text(encoding="utf-8"))["metrics"]
    reproduction_fields = ("accuracy", "macro_f1", "balanced_accuracy", "macro_precision", "macro_recall")
    deltas = {field: abs(float(reproduced[field]) - float(reference[field])) for field in reproduction_fields}
    reproduction_failures = [field for field, delta in deltas.items() if delta > REPRODUCTION_TOLERANCE]
    if reproduction_failures:
        raise RuntimeError(f"Frozen R3 reproduction failed: {deltas}")

    train_truth = np.asarray([row["_target"] for row in train_rows], dtype=np.int64)
    validation_truth = np.asarray([row["_target"] for row in validation_rows], dtype=np.int64)
    threshold = fit_recall_constrained_threshold(train_evidence["margin"], train_truth)
    threshold_value = float(threshold["threshold"])
    full_positions = np.arange(len(validation_rows))
    matched_positions = np.asarray([
        index for index, row in enumerate(validation_rows) if row["_source_index"] in matched_sources
    ])
    if len(matched_positions) != 1_564 or int(np.sum(validation_truth[matched_positions])) != 782:
        raise RuntimeError("R4_1_MATCHED_CONTROL_FAIL after evidence inference")

    full = subset_report(validation_rows, validation_truth, validation_evidence["expected_posterior"],
                         validation_evidence["margin"], full_positions, threshold_value)
    matched = subset_report(validation_rows, validation_truth, validation_evidence["expected_posterior"],
                            validation_evidence["margin"], matched_positions, threshold_value)
    matched_auc = matched["expected_margin"]["roc_auc_deletion"]
    matched_pr = matched["expected_margin"]["pr_auc_deletion"]
    if matched_auc >= 0.70 and matched_pr >= 0.65:
        signal_gate, final_status = "EXPECTED_EVIDENCE_SIGNAL_STRONG", "R4_2A_EXPECTED_EVIDENCE_STRONG"
    elif matched_auc >= 0.60:
        signal_gate, final_status = "EXPECTED_EVIDENCE_SIGNAL_MODERATE", "R4_2A_EXPECTED_EVIDENCE_MODERATE"
    else:
        signal_gate, final_status = "EXPECTED_EVIDENCE_SIGNAL_WEAK", "R4_2A_EXPECTED_EVIDENCE_WEAK"

    validation_prediction = apply_threshold(validation_evidence["margin"], threshold_value)
    relation_origin: dict[str, Any] = {}
    for relation in ("correct", "substitution"):
        positions = np.asarray([i for i, row in enumerate(validation_rows) if row["relation"] == relation])
        relation_origin[relation] = {
            "support": int(len(positions)), "margin": distribution(validation_evidence["margin"][positions]),
            "false_deletion_rate": float(np.mean(validation_prediction[positions] == 1)),
        }
    deletion_positions = np.flatnonzero(validation_truth == 1)
    substitution_positions = np.asarray([i for i, row in enumerate(validation_rows) if row["relation"] == "substitution"])
    deletion_vs_sub_truth = np.r_[np.ones(len(deletion_positions)), np.zeros(len(substitution_positions))]
    deletion_vs_sub_score = np.r_[validation_evidence["margin"][deletion_positions],
                                  validation_evidence["margin"][substitution_positions]]
    deletion_vs_sub_auc = float(roc_auc_score(deletion_vs_sub_truth, -deletion_vs_sub_score))
    deletion_median = float(np.median(validation_evidence["margin"][deletion_positions]))
    correct_median = relation_origin["correct"]["margin"]["median"]
    substitution_median = relation_origin["substitution"]["margin"]["median"]
    if matched_auc >= 0.60 and deletion_vs_sub_auc >= 0.60 and deletion_median < min(correct_median, substitution_median):
        interpretation = "DELETION_SPECIFIC_EVIDENCE_CONFIRMED"
    elif matched_auc >= 0.60 and (deletion_vs_sub_auc < 0.60 or deletion_median >= substitution_median):
        interpretation = "EXPECTED_PHONE_MISMATCH_ONLY"
    else:
        interpretation = "DELETION_EVIDENCE_NOT_CONFIRMED"

    speaker_metrics: dict[str, Any] = {}
    for speaker in r4.VALIDATION_SPEAKERS:
        positions = np.asarray([i for i, row in enumerate(validation_rows) if row["speaker_id"] == speaker])
        speaker_metrics[speaker] = binary_metrics(validation_truth[positions], validation_prediction[positions])

    diagnostic_phones = set(IMPORTANT_PHONES)
    support = Counter(row["expected_phone_canonical"] for row in validation_rows if row["relation"] == "deletion")
    diagnostic_phones.update(phone for phone, count in support.items() if count >= 20)
    phone_metrics: dict[str, Any] = {}
    for phone in sorted(diagnostic_phones):
        positions = np.asarray([i for i, row in enumerate(validation_rows) if row["expected_phone_canonical"] == phone])
        phone_truth = validation_truth[positions]
        phone_score = validation_evidence["margin"][positions]
        item: dict[str, Any] = {
            "deletion_support": int(np.sum(phone_truth)), "non_deletion_support": int(np.sum(1 - phone_truth)),
            "median_margin_non_deletion": float(np.median(phone_score[phone_truth == 0])) if np.any(phone_truth == 0) else None,
            "median_margin_deletion": float(np.median(phone_score[phone_truth == 1])) if np.any(phone_truth == 1) else None,
            "thresholded": binary_metrics(phone_truth, validation_prediction[positions]),
        }
        item["roc_auc_deletion"] = (
            float(roc_auc_score(phone_truth, -phone_score)) if len(np.unique(phone_truth)) == 2 else None
        )
        phone_metrics[phone] = item

    concentration: dict[str, Any] = {}
    for name, predicate in {
        "D_T_R_L": lambda phone: phone in DTRL,
        "other_phones": lambda phone: phone not in DTRL,
    }.items():
        positions = np.asarray([i for i, row in enumerate(validation_rows) if predicate(row["expected_phone_canonical"])])
        concentration[name] = subset_report(
            validation_rows, validation_truth, validation_evidence["expected_posterior"],
            validation_evidence["margin"], positions, threshold_value,
        )

    comparisons = {
        "full_validation": {
            name: {metric: full["thresholded_expected_margin"][metric] - value
                   for metric, value in reference_metrics.items()}
            for name, reference_metrics in FULL_REFERENCES.items()
        },
        "strict_matched": {
            name: {metric: matched["thresholded_expected_margin"][metric] - value
                   for metric, value in reference_metrics.items()}
            for name, reference_metrics in MATCHED_REFERENCES.items()
        },
    }

    EXPERIMENT_DIR.mkdir(parents=True)
    config = {
        "experiment": "R4-2A Frozen R3 Expected-Phone Evidence Audit",
        "markers": ["RESEARCH_ONLY", "AUDIT_ONLY", "NO_TRAINING", "R4_TEST_CLOSED"],
        "checkpoint": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_sha, "selected_epoch": 47,
        "v4_sha256": source_summary["dataset_sha256"],
        "matched_control_sha256": source_summary["matched_control_sha256"],
        "audio": {"sample_rate": r3.SAMPLE_RATE, "window_seconds": r3.WINDOW_SECONDS,
                  "window_samples": r3.WINDOW_SAMPLES, "feature_shape": list(r3.EXPECTED_FEATURE_SHAPE)},
        "score": "expected_logit - max(other 39 logits); lower means deletion evidence",
        "threshold_protocol": threshold["selection"],
        "interpretation_preregistered": {
            "deletion_specific": "matched margin AUC>=0.60 AND deletion-vs-substitution AUC>=0.60 AND deletion median below both non-deletion origins",
            "mismatch_only": "matched margin AUC>=0.60 but deletion-vs-substitution condition fails",
            "not_confirmed": "matched margin AUC<0.60",
        },
        "test_audio_paths_resolved": False, "test_audio_accessed": False, "training_performed": False,
    }
    write_json(EXPERIMENT_DIR / "config.json", config)
    write_json(EXPERIMENT_DIR / "preflight.json", {
        "checkpoint_sha256": checkpoint_sha, "source": source_summary, "audio": audio_preflight,
        "device": str(device), "r3_reproduction": {"status": "PASS", "metrics": reproduced, "deltas": deltas},
        "test_audio_paths_resolved": False, "test_audio_accessed": False,
    })
    write_json(EXPERIMENT_DIR / "train_threshold.json", threshold)
    write_json(EXPERIMENT_DIR / "full_validation_diagnostics.json", full)
    write_json(EXPERIMENT_DIR / "strict_matched_diagnostics.json", matched)
    write_json(EXPERIMENT_DIR / "phone_diagnostics.json", phone_metrics)
    write_json(EXPERIMENT_DIR / "concentration_diagnostics.json", concentration)
    write_json(EXPERIMENT_DIR / "relation_origin_diagnostics.json", {
        **relation_origin, "deletion": {"support": int(len(deletion_positions)),
                                       "margin": distribution(validation_evidence["margin"][deletion_positions])},
        "deletion_vs_substitution_roc_auc": deletion_vs_sub_auc,
    })
    write_json(EXPERIMENT_DIR / "per_speaker_metrics.json", speaker_metrics)
    write_json(EXPERIMENT_DIR / "comparator_deltas.json", {
        "references": {"full": FULL_REFERENCES, "matched": MATCHED_REFERENCES}, "deltas": comparisons,
    })
    save_evidence(EXPERIMENT_DIR / "train_phone_evidence.csv", train_rows, train_evidence, threshold_value)
    save_evidence(EXPERIMENT_DIR / "validation_phone_evidence.csv", validation_rows, validation_evidence,
                  threshold_value, matched_sources)
    final = {
        "status": final_status, "primary_signal_gate": signal_gate,
        "deletion_specific_interpretation": interpretation,
        "strict_matched_margin_roc_auc": matched_auc, "strict_matched_margin_pr_auc": matched_pr,
        "deletion_vs_substitution_roc_auc": deletion_vs_sub_auc,
        "train_threshold": threshold_value, "full_validation_metrics": full["thresholded_expected_margin"],
        "strict_matched_metrics": matched["thresholded_expected_margin"],
        "inference_seconds_including_feature_materialization": inference_seconds,
        "training_performed": False, "r4_test_audio_paths_resolved": False,
        "r4_test_audio_accessed": False, "r4_test_inference": False,
    }
    write_json(EXPERIMENT_DIR / "final_status.json", final)
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
