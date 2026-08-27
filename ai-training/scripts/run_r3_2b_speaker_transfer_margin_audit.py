from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
R3D_CHECKPOINT = (
    REPO_ROOT
    / "ai-training/experiments/r3_1d_observed_phone_seed42_48epochs/"
    "R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E"
R3_2A_DIR = REPO_ROOT / "ai-training/experiments/r3_2a_expected_vs_best_alternative"
EVIDENCE_CSV = R3_2A_DIR / "validation_phone_evidence.csv"
EXPECTED_EVIDENCE_SHA256 = "E0C7DF574EAA61DEAB57C77C7E8459BB4306E938DAF6F7AB00B7D0D68DF0111F"
V4_AUDIT = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4_audit.json"
OUTPUT_DIR = REPO_ROOT / "ai-training/experiments/r3_2b_speaker_transfer_margin"

SPEAKERS = ("ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI")
IMPORTANT_PHONES = ("TH", "DH", "R", "V", "D", "T", "S", "Z")
IMPORTANT_PAIRS = (("TH", "T"), ("DH", "D"), ("R", "L"), ("V", "W"), ("Z", "S"))
REPRODUCTION_TOLERANCE = 1e-9
MIN_CALIBRATION_SUBSTITUTION_RECALL = 0.50
PASS_MACRO_F1_DELTA = 0.05
PASS_SUBSTITUTION_F1_DELTA = 0.03
PASS_OOF_SUBSTITUTION_RECALL = 0.50
PASS_EACH_SPEAKER_SUBSTITUTION_RECALL = 0.35
PASS_SPEAKERS_IMPROVED = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(truth, predicted, labels=[0, 1]).astype(int)
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    correct_precision = tn / (tn + fn) if tn + fn else 0.0
    correct_recall = tn / (tn + fp) if tn + fp else 0.0
    correct_f1 = (
        2 * correct_precision * correct_recall / (correct_precision + correct_recall)
        if correct_precision + correct_recall else 0.0
    )
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


def select_constrained_threshold(score: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    sorted_truth = truth[order]
    unique_values, first_indexes, counts = np.unique(sorted_score, return_index=True, return_counts=True)
    total_positive = int(truth.sum())
    total_negative = int(len(truth) - total_positive)
    cumulative_positive = np.cumsum(sorted_truth)
    cumulative_negative = np.cumsum(1 - sorted_truth)
    best_key: tuple[float, float, float, float] | None = None
    best: dict[str, Any] | None = None
    feasible = 0
    for threshold, start, count in zip(unique_values, first_indexes, counts):
        end = int(start + count - 1)
        tp, fp = int(cumulative_positive[end]), int(cumulative_negative[end])
        fn, tn = total_positive - tp, total_negative - fp
        sub_recall = tp / total_positive
        if sub_recall + 1e-15 < MIN_CALIBRATION_SUBSTITUTION_RECALL:
            continue
        feasible += 1
        correct_precision = tn / (tn + fn) if tn + fn else 0.0
        correct_recall = tn / (tn + fp) if tn + fp else 0.0
        correct_f1 = (
            2 * correct_precision * correct_recall / (correct_precision + correct_recall)
            if correct_precision + correct_recall else 0.0
        )
        sub_precision = tp / (tp + fp) if tp + fp else 0.0
        sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall) if sub_precision + sub_recall else 0.0
        macro_f1 = (correct_f1 + sub_f1) / 2.0
        # Lower threshold makes fewer substitution calls, hence is the conservative tie-break.
        key = (macro_f1, sub_f1, sub_precision, -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": float(threshold), "macro_f1": float(macro_f1),
                "substitution_precision": float(sub_precision), "substitution_recall": float(sub_recall),
                "substitution_f1": float(sub_f1), "confusion_matrix": [[tn, fp], [fn, tp]],
            }
    if best is None:
        raise RuntimeError("No threshold satisfies calibration substitution recall >= 0.50")
    best["unique_thresholds"] = int(len(unique_values))
    best["feasible_thresholds"] = int(feasible)
    return best


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)), "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)), "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)), "max": float(np.max(values)),
    }


def diagnostic_for_subset(indexes: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    local_truth, local_prediction = truth[indexes], prediction[indexes]
    correct_count = int(np.sum(local_truth == 0))
    substitution_count = int(np.sum(local_truth == 1))
    correct_recall = float(np.mean(local_prediction[local_truth == 0] == 0)) if correct_count else None
    substitution_recall = float(np.mean(local_prediction[local_truth == 1] == 1)) if substitution_count else None
    predicted_substitution = int(np.sum(local_prediction == 1))
    true_positive = int(np.sum((local_truth == 1) & (local_prediction == 1)))
    substitution_precision = true_positive / predicted_substitution if predicted_substitution else None
    return {
        "correct_origin_support": correct_count, "substitution_origin_support": substitution_count,
        "correct_recall": correct_recall, "substitution_recall": substitution_recall,
        "substitution_precision": substitution_precision,
    }


def main() -> int:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite existing audit directory: {OUTPUT_DIR}")
    checkpoint_sha = sha256_file(R3D_CHECKPOINT)
    evidence_sha = sha256_file(EVIDENCE_CSV)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA mismatch: {checkpoint_sha}")
    if evidence_sha != EXPECTED_EVIDENCE_SHA256:
        raise RuntimeError(f"R3-2A evidence SHA mismatch: {evidence_sha}")

    with EVIDENCE_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "row_index", "speaker_id", "expected_phone_canonical", "true_observed_phone_canonical",
        "relation_origin", "predicted_top1_phone", "expected_logit", "best_alternative_logit", "expected_margin",
    }
    if not rows or required - set(rows[0]):
        raise RuntimeError(f"Evidence fields missing: {sorted(required - set(rows[0] if rows else {}))}")
    if len(rows) != 28_212:
        raise RuntimeError(f"Expected 28,212 evidence rows, got {len(rows)}")
    if [int(row["row_index"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("Evidence row indexes are not sequential and unique")
    speakers = [row["speaker_id"] for row in rows]
    if set(speakers) != set(SPEAKERS):
        raise RuntimeError(f"Validation speaker mismatch: {sorted(set(speakers))}")
    if any(row["relation_origin"] not in {"correct", "substitution"} for row in rows):
        raise RuntimeError("Unexpected relation origin in evidence")

    truth = np.asarray([0 if row["relation_origin"] == "correct" else 1 for row in rows], dtype=np.int64)
    margin = np.asarray([float(row["expected_margin"]) for row in rows], dtype=np.float64)
    recomputed_margin = np.asarray([
        float(row["expected_logit"]) - float(row["best_alternative_logit"]) for row in rows
    ])
    if not np.isfinite(margin).all() or not np.allclose(margin, recomputed_margin, atol=1e-12, rtol=0):
        raise RuntimeError("Margin values are non-finite or do not equal expected_logit - best_alternative_logit")
    hard_prediction = np.asarray([
        int(row["predicted_top1_phone"] != row["expected_phone_canonical"]) for row in rows
    ], dtype=np.int64)
    hard_metrics = binary_metrics(truth, hard_prediction)

    r3_2a_thresholds = json.loads((R3_2A_DIR / "threshold_results.json").read_text(encoding="utf-8"))
    r3_2a_scores = json.loads((R3_2A_DIR / "score_diagnostics.json").read_text(encoding="utf-8"))
    reference_hard = r3_2a_thresholds["hard_argmax"]
    hard_fields = (
        "accuracy", "balanced_accuracy", "macro_f1", "substitution_precision",
        "substitution_recall", "substitution_f1",
    )
    hard_deltas = {field: abs(hard_metrics[field] - reference_hard[field]) for field in hard_fields}
    margin_auc = float(roc_auc_score(truth, -margin))
    margin_pr_auc = float(average_precision_score(truth, -margin))
    margin_summary = {
        "roc_auc_substitution": margin_auc, "pr_auc_substitution": margin_pr_auc,
        "correct_origin": distribution(margin[truth == 0]),
        "substitution_origin": distribution(margin[truth == 1]),
    }
    reference_margin = r3_2a_scores["expected_margin"]
    margin_deltas = {
        "roc_auc_substitution": abs(margin_auc - reference_margin["roc_auc_substitution"]),
        "pr_auc_substitution": abs(margin_pr_auc - reference_margin["pr_auc_substitution"]),
    }
    for origin in ("correct_origin", "substitution_origin"):
        for field in ("mean", "median", "p10", "p25", "p75", "p90"):
            margin_deltas[f"{origin}.{field}"] = abs(margin_summary[origin][field] - reference_margin[origin][field])
    reproduction_failures = [
        f"hard_argmax.{field}: delta={delta:.12g}" for field, delta in hard_deltas.items()
        if delta > REPRODUCTION_TOLERANCE
    ]
    reproduction_failures.extend(
        f"margin.{field}: delta={delta:.12g}" for field, delta in margin_deltas.items()
        if delta > REPRODUCTION_TOLERANCE
    )

    OUTPUT_DIR.mkdir(parents=True)
    reproduction = {
        "status": "PASS" if not reproduction_failures else "FAIL",
        "checkpoint_sha256": checkpoint_sha, "evidence_sha256": evidence_sha,
        "rows": len(rows), "speakers": list(SPEAKERS),
        "hard_argmax": hard_metrics, "hard_argmax_absolute_deltas": hard_deltas,
        "expected_margin": margin_summary, "margin_absolute_deltas": margin_deltas,
        "failures": reproduction_failures, "test_audio_accessed": False,
    }
    write_json(OUTPUT_DIR / "reproduction.json", reproduction)
    write_json(OUTPUT_DIR / "config.json", {
        "experiment": "R3-2B speaker-transfer expected-margin threshold audit",
        "markers": ["RESEARCH_ONLY", "NO_TRAINING", "VALIDATION_EVIDENCE_ONLY", "TEST_CLOSED"],
        "checkpoint_sha256": checkpoint_sha, "evidence_sha256": evidence_sha,
        "score": "expected_margin only", "rule": "margin <= threshold => substitution",
        "speakers": list(SPEAKERS), "folds": 6,
        "calibration": "five speakers; held-out speaker has no influence on threshold",
        "threshold_constraint": "calibration substitution recall >= 0.50",
        "threshold_selection": "highest Macro-F1; higher substitution F1; higher substitution precision; lower threshold",
        "per_phone_threshold": False, "per_speaker_heldout_tuning": False,
        "pass_gate": {
            "A_macro_f1_delta": PASS_MACRO_F1_DELTA,
            "B_oof_substitution_recall": PASS_OOF_SUBSTITUTION_RECALL,
            "C_substitution_f1_delta": PASS_SUBSTITUTION_F1_DELTA,
            "D_each_speaker_substitution_recall": PASS_EACH_SPEAKER_SUBSTITUTION_RECALL,
            "E_speakers_macro_f1_improved": f">={PASS_SPEAKERS_IMPROVED}/6",
        },
        "test_audio_accessed": False,
    })
    if reproduction_failures:
        write_json(OUTPUT_DIR / "final_status.json", {
            "status": "R3_2B_REPRODUCTION_FAIL", "binary_margin_transfer": "BINARY_MARGIN_TRANSFER_NOT_READY",
            "test_opened": False, "failures": reproduction_failures,
        })
        print("R3_2B_REPRODUCTION_FAIL")
        return 0

    speaker_array = np.asarray(speakers)
    oof_prediction = np.full(len(rows), -1, dtype=np.int64)
    fold_assignment = np.full(len(rows), "", dtype=object)
    folds: dict[str, Any] = {}
    thresholds: list[float] = []
    improved_speakers = 0
    for heldout in SPEAKERS:
        heldout_indexes = np.flatnonzero(speaker_array == heldout)
        calibration_indexes = np.flatnonzero(speaker_array != heldout)
        selected = select_constrained_threshold(margin[calibration_indexes], truth[calibration_indexes])
        threshold = selected["threshold"]
        heldout_prediction = (margin[heldout_indexes] <= threshold).astype(np.int64)
        oof_prediction[heldout_indexes] = heldout_prediction
        fold_assignment[heldout_indexes] = heldout
        heldout_metrics = binary_metrics(truth[heldout_indexes], heldout_prediction)
        heldout_hard_metrics = binary_metrics(truth[heldout_indexes], hard_prediction[heldout_indexes])
        macro_delta = heldout_metrics["macro_f1"] - heldout_hard_metrics["macro_f1"]
        improved = macro_delta > 0
        improved_speakers += int(improved)
        thresholds.append(threshold)
        folds[heldout] = {
            "calibration_speakers": [speaker for speaker in SPEAKERS if speaker != heldout],
            "heldout_speaker": heldout, "calibration_rows": int(len(calibration_indexes)),
            "heldout_rows": int(len(heldout_indexes)), "selected_threshold": threshold,
            "calibration": selected, "heldout": heldout_metrics, "hard_argmax_heldout": heldout_hard_metrics,
            "heldout_macro_f1_delta": float(macro_delta), "heldout_macro_f1_improved": improved,
        }
    if np.any(oof_prediction < 0) or np.any(fold_assignment == ""):
        raise RuntimeError("OOF coverage is incomplete")
    if Counter(fold_assignment) != Counter(speakers):
        raise RuntimeError("OOF fold identity does not cover every row exactly once")

    oof_metrics = binary_metrics(truth, oof_prediction)
    deltas = {
        "accuracy": oof_metrics["accuracy"] - hard_metrics["accuracy"],
        "balanced_accuracy": oof_metrics["balanced_accuracy"] - hard_metrics["balanced_accuracy"],
        "macro_f1": oof_metrics["macro_f1"] - hard_metrics["macro_f1"],
        "substitution_precision": oof_metrics["substitution_precision"] - hard_metrics["substitution_precision"],
        "substitution_recall": oof_metrics["substitution_recall"] - hard_metrics["substitution_recall"],
        "substitution_f1": oof_metrics["substitution_f1"] - hard_metrics["substitution_f1"],
    }
    gate = {
        "A_macro_f1_improvement_at_least_0.05": bool(deltas["macro_f1"] >= PASS_MACRO_F1_DELTA),
        "B_oof_substitution_recall_at_least_0.50": bool(oof_metrics["substitution_recall"] >= PASS_OOF_SUBSTITUTION_RECALL),
        "C_substitution_f1_improvement_at_least_0.03": bool(deltas["substitution_f1"] >= PASS_SUBSTITUTION_F1_DELTA),
        "D_each_heldout_substitution_recall_at_least_0.35": bool(
            all(fold["heldout"]["substitution_recall"] >= PASS_EACH_SPEAKER_SUBSTITUTION_RECALL for fold in folds.values())
        ),
        "E_at_least_4_of_6_speakers_improve_macro_f1": bool(improved_speakers >= PASS_SPEAKERS_IMPROVED),
    }
    gate["all_pass"] = all(gate.values())

    threshold_values = np.asarray(thresholds, dtype=np.float64)
    q25, q75 = np.percentile(threshold_values, [25, 75])
    threshold_stats = {
        "values": {speaker: folds[speaker]["selected_threshold"] for speaker in SPEAKERS},
        "mean": float(np.mean(threshold_values)), "median": float(np.median(threshold_values)),
        "min": float(np.min(threshold_values)), "max": float(np.max(threshold_values)),
        "standard_deviation_population": float(np.std(threshold_values, ddof=0)),
        "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25),
        "assessment": "PENDING_CONTEXTUAL_INTERPRETATION",
    }

    phone_diagnostics: dict[str, Any] = {}
    for phone in IMPORTANT_PHONES:
        indexes = np.asarray([index for index, row in enumerate(rows) if row["expected_phone_canonical"] == phone])
        phone_diagnostics[phone] = diagnostic_for_subset(indexes, truth, oof_prediction)

    pair_counts = Counter(
        (row["expected_phone_canonical"], row["true_observed_phone_canonical"])
        for row in rows if row["relation_origin"] == "substitution"
    )
    top_pairs = [pair for pair, _ in pair_counts.most_common(20)]
    pairs_to_report = list(dict.fromkeys([*IMPORTANT_PAIRS, *top_pairs]))
    pair_diagnostics: dict[str, Any] = {}
    for expected_phone, observed_phone in pairs_to_report:
        indexes = np.asarray([
            index for index, row in enumerate(rows)
            if row["relation_origin"] == "substitution"
            and row["expected_phone_canonical"] == expected_phone
            and row["true_observed_phone_canonical"] == observed_phone
        ])
        detected = int(np.sum(oof_prediction[indexes] == 1)) if len(indexes) else 0
        pair_diagnostics[f"{expected_phone}->{observed_phone}"] = {
            "support": int(len(indexes)), "detection_recall": detected / len(indexes) if len(indexes) else None,
            "false_negative_count": int(len(indexes) - detected),
            "margin_distribution": distribution(margin[indexes]) if len(indexes) else None,
            "requested_pair": (expected_phone, observed_phone) in set(IMPORTANT_PAIRS),
        }

    dh = pair_diagnostics["DH->D"]
    other_large = {
        pair: item for pair, item in pair_diagnostics.items()
        if pair != "DH->D" and item["support"] >= 40
    }
    other_large_recalls = [item["detection_recall"] for item in other_large.values()]
    if (
        dh["margin_distribution"]["median"] >= 0
        and dh["detection_recall"] <= oof_metrics["substitution_recall"] - 0.20
    ):
        dh_assessment = "DH_D_SCORE_FAILURE_MODE"
    elif dh["detection_recall"] >= oof_metrics["substitution_recall"] - 0.10 and dh["margin_distribution"]["median"] < 0:
        dh_assessment = "DH_D_GENERAL_SCORE_COMPATIBLE"
    else:
        dh_assessment = "DH_D_SCORE_WEAK"
    dh_analysis = {
        **dh, "assessment": dh_assessment,
        "oof_overall_substitution_recall": oof_metrics["substitution_recall"],
        "detection_recall_delta_vs_oof": dh["detection_recall"] - oof_metrics["substitution_recall"],
        "other_large_pairs": other_large,
        "other_large_pair_detection_recall_mean": float(np.mean(other_large_recalls)),
        "other_large_pair_detection_recall_median": float(np.median(other_large_recalls)),
    }

    with V4_AUDIT.open("r", encoding="utf-8") as handle:
        quality = json.load(handle)["speaker_quality"]
    speaker_quality: dict[str, Any] = {}
    exclusion_rates, heldout_recalls, heldout_macro_f1 = [], [], []
    for speaker in SPEAKERS:
        item = quality[speaker]
        heldout = folds[speaker]["heldout"]
        speaker_quality[speaker] = {
            "clean_substitution_support": item["clean_substitutions"],
            "excluded_substitutions": item["excluded_substitutions"],
            "retention_percentage": item["substitution_retention_percentage"],
            "exclusion_percentage": item["substitution_exclusion_percentage"],
            "high_exclusion_rate": item["high_substitution_exclusion_rate"],
            "heldout_macro_f1": heldout["macro_f1"],
            "heldout_substitution_recall": heldout["substitution_recall"],
            "heldout_substitution_f1": heldout["substitution_f1"],
        }
        exclusion_rates.append(item["substitution_exclusion_percentage"])
        heldout_recalls.append(heldout["substitution_recall"])
        heldout_macro_f1.append(heldout["macro_f1"])
    speaker_quality_association = {
        "speakers": speaker_quality,
        "correlations_descriptive_only_n_equals_6": {
            "exclusion_vs_substitution_recall_pearson": float(pearsonr(exclusion_rates, heldout_recalls).statistic),
            "exclusion_vs_substitution_recall_spearman": float(spearmanr(exclusion_rates, heldout_recalls).statistic),
            "exclusion_vs_macro_f1_pearson": float(pearsonr(exclusion_rates, heldout_macro_f1).statistic),
            "exclusion_vs_macro_f1_spearman": float(spearmanr(exclusion_rates, heldout_macro_f1).statistic),
        },
        "causality_warning": "Descriptive association only; six speakers cannot establish causality.",
    }

    all_validation_candidate = select_constrained_threshold(margin, truth)
    all_validation_candidate.update({
        "label": "VALIDATION_CALIBRATED_CANDIDATE",
        "markers": ["NOT_TESTED", "NOT_PRODUCTION_CALIBRATED"],
    })

    if gate["all_pass"]:
        status = "R3_2B_TRANSFER_PASS"
        readiness = "BINARY_MARGIN_TRANSFER_READY"
    elif gate["A_macro_f1_improvement_at_least_0.05"] or gate["C_substitution_f1_improvement_at_least_0.03"]:
        status = "R3_2B_TRANSFER_WEAK"
        readiness = "BINARY_MARGIN_TRANSFER_NOT_READY"
    else:
        status = "R3_2B_TRANSFER_FAIL"
        readiness = "BINARY_MARGIN_TRANSFER_NOT_READY"

    oof_fields = [
        "row_index", "speaker_id", "expected_phone_canonical", "true_observed_phone_canonical",
        "relation_origin", "expected_margin", "heldout_fold", "fold_threshold", "oof_prediction",
        "hard_argmax_prediction",
    ]
    with (OUTPUT_DIR / "oof_row_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=oof_fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            speaker = row["speaker_id"]
            writer.writerow({
                "row_index": index, "speaker_id": speaker,
                "expected_phone_canonical": row["expected_phone_canonical"],
                "true_observed_phone_canonical": row["true_observed_phone_canonical"],
                "relation_origin": row["relation_origin"], "expected_margin": margin[index],
                "heldout_fold": fold_assignment[index], "fold_threshold": folds[speaker]["selected_threshold"],
                "oof_prediction": "substitution" if oof_prediction[index] else "correct",
                "hard_argmax_prediction": "substitution" if hard_prediction[index] else "correct",
            })

    write_json(OUTPUT_DIR / "fold_results.json", folds)
    write_json(OUTPUT_DIR / "oof_report.json", {
        "rows": len(rows), "each_row_exactly_once": True, "metrics": oof_metrics,
        "hard_argmax": hard_metrics, "deltas": deltas, "gate": gate,
    })
    write_json(OUTPUT_DIR / "threshold_stability.json", threshold_stats)
    write_json(OUTPUT_DIR / "phone_diagnostics.json", phone_diagnostics)
    write_json(OUTPUT_DIR / "substitution_pair_diagnostics.json", pair_diagnostics)
    write_json(OUTPUT_DIR / "dh_d_failure_analysis.json", dh_analysis)
    write_json(OUTPUT_DIR / "speaker_quality_association.json", speaker_quality_association)
    write_json(OUTPUT_DIR / "all_validation_threshold_candidate.json", all_validation_candidate)
    write_json(OUTPUT_DIR / "final_status.json", {
        "status": status, "binary_margin_transfer": readiness, "gate": gate,
        "threshold_stability_assessment": "PENDING_CONTEXTUAL_INTERPRETATION",
        "dh_d_assessment": dh_assessment, "test_opened": False, "test_audio_accessed": False,
    })
    print(json.dumps({
        "status": status, "readiness": readiness, "reproduction": "PASS", "folds": folds,
        "oof": oof_metrics, "hard_argmax": hard_metrics, "deltas": deltas, "gate": gate,
        "threshold_stability": threshold_stats, "dh_d": dh_analysis,
        "all_validation_candidate": all_validation_candidate, "test_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
