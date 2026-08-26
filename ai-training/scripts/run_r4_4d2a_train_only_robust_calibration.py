#!/usr/bin/env python3
"""R4-4D2A TRAIN-only speaker-robust calibration audit.

Consumes the frozen R4-4D0 TRAIN position-score export. It never loads audio,
VALIDATION rows/scores, or TEST paths. Threshold sweeps are exact over all unique
float64 scores plus both np.nextafter edge candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "ai-training/experiments/r4_4d2a_train_only_robust_calibration"
SCORES = REPO_ROOT / "ai-training/experiments/r4_4d0_ctc_hypothesis_feasibility/train_position_scores.csv"
R4D0_AUC = REPO_ROOT / "ai-training/experiments/r4_4d0_ctc_hypothesis_feasibility/train_auc_metrics.json"
R4D0_DEL_SUB = REPO_ROOT / "ai-training/experiments/r4_4d0_ctc_hypothesis_feasibility/train_deletion_vs_substitution.json"
NUMERICAL = REPO_ROOT / "ai-training/experiments/r4_4d1_numerical_contract/r4_4d1_complete_evaluation_contract.json"
V2_FINAL = REPO_ROOT / "ai-training/experiments/r4_4d1_locked_hypothesis_validation/locked_execution_v2/final_status.json"

TRAIN_SPEAKERS = ["BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA"]
VALIDATION_SPEAKERS = ["ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI"]
TEST_SPEAKERS = ["ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK"]
FAMILIES = ["RAW", "TARGET", "TIME"]
RELATION_TO_ID = {"correct": 0, "substitution": 1, "deletion": 2}
ID_TO_RELATION = {value: key for key, value in RELATION_TO_ID.items()}
SIMPLICITY = {"RAW": 0, "TIME": 1, "TARGET": 2}

SOURCE_IDENTITIES = {
    "v4": ("ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv", "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D"),
    "checkpoint": ("ai-training/experiments/r4_4c2_bigru_ctc_seed42/R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt", "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085"),
    "r4_4d0_manifest": ("ai-training/experiments/r4_4d0_ctc_hypothesis_feasibility/artifact_hashes.json", "A8443C85AA9C03879E3A907CC3AB05CC6154BB365BED70A59B4ED8E9FBA1A920"),
    "r4_4d1_numerical_contract": ("ai-training/experiments/r4_4d1_numerical_contract/r4_4d1_complete_evaluation_contract.json", "5DC07A4B719FD6F38DBD1366CF802787FA3882CA972E5356A8F91DB435443425"),
    "train_position_scores": ("ai-training/experiments/r4_4d0_ctc_hypothesis_feasibility/train_position_scores.csv", "E6361398B61B9CFC03AE695CA72377B5F5DBD8D71B254CB202CBE432B918996A"),
    "train_auc_metrics": ("ai-training/experiments/r4_4d0_ctc_hypothesis_feasibility/train_auc_metrics.json", "057DFF840A747CA955F9141A1CDCD9C31303312688C92FE6F2ADB10D83346A9D"),
    "train_deletion_vs_substitution": ("ai-training/experiments/r4_4d0_ctc_hypothesis_feasibility/train_deletion_vs_substitution.json", "627075FE379A8377DD34D6317D33BFE272FD582BFBD835D9F3907C2A20C8C612"),
    "r4_4d1_v2_manifest": ("ai-training/experiments/r4_4d1_locked_hypothesis_validation/locked_execution_v2/artifact_hashes.json", "447D55708D20FC7506C0A49C9F8909B8E7AFDDEEF46F54FC63F99C322AC42577"),
    "matched_control": ("ai-training/experiments/r4_4a_ctc_sequence_feasibility/validation_word_eligible_matched_control.csv", "D933F674743DA06CC8FAB425CEBF81D9C78505E1BDB4A90204DDB2E1A15B4798"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(native(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def metrics_from_predictions(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth_del = truth == 2; pred_del = prediction == 2
    tp = int(np.sum(truth_del & pred_del)); fn = int(np.sum(truth_del & ~pred_del))
    fp = int(np.sum(~truth_del & pred_del)); tn = int(np.sum(~truth_del & ~pred_del))
    deletion_precision = safe_div(tp, tp + fp); deletion_recall = safe_div(tp, tp + fn)
    deletion_f1 = f1(deletion_precision, deletion_recall)
    nondeletion_precision = safe_div(tn, tn + fn); nondeletion_recall = safe_div(tn, tn + fp)
    nondeletion_f1 = f1(nondeletion_precision, nondeletion_recall)
    confusion = np.zeros((3, 3), dtype=np.int64)
    for expected, predicted in zip(truth, prediction):
        confusion[int(expected), int(predicted)] += 1
    per_class: dict[str, Any] = {}
    class_f1: list[float] = []
    for class_id in range(3):
        class_tp = int(confusion[class_id, class_id])
        class_fp = int(confusion[:, class_id].sum() - class_tp)
        class_fn = int(confusion[class_id, :].sum() - class_tp)
        precision = safe_div(class_tp, class_tp + class_fp); recall = safe_div(class_tp, class_tp + class_fn)
        value_f1 = f1(precision, recall); class_f1.append(value_f1)
        per_class[ID_TO_RELATION[class_id]] = {"precision": precision, "recall": recall,
                                                "f1": value_f1, "support": int(confusion[class_id, :].sum())}
    correct_total = int(np.sum(truth == 0)); sub_total = int(np.sum(truth == 1))
    return {
        "accuracy": safe_div(tp + tn, len(truth)),
        "balanced_accuracy": (deletion_recall + nondeletion_recall) / 2.0,
        "binary_macro_f1": (deletion_f1 + nondeletion_f1) / 2.0,
        "deletion_precision": deletion_precision, "deletion_recall": deletion_recall,
        "deletion_f1": deletion_f1, "nondeletion_precision": nondeletion_precision,
        "nondeletion_recall": nondeletion_recall, "nondeletion_f1": nondeletion_f1,
        "binary_confusion_matrix": [[tn, fp], [fn, tp]],
        "correct_false_deletion": safe_div(int(np.sum((truth == 0) & pred_del)), correct_total),
        "substitution_false_deletion": safe_div(int(np.sum((truth == 1) & pred_del)), sub_total),
        "three_relation_macro_f1": float(np.mean(class_f1)),
        "three_relation_per_class": per_class,
        "three_relation_confusion_matrix": confusion.tolist(),
    }


def base_predictions(frame: pd.DataFrame, family: str) -> np.ndarray:
    prefix = family.lower()
    keep = frame[f"{prefix}_keep"].to_numpy(np.float64)
    sub = frame[f"{prefix}_best_sub"].to_numpy(np.float64)
    return np.where(keep >= sub, 0, 1).astype(np.int8)


def apply_threshold(frame: pd.DataFrame, family: str, threshold: float) -> np.ndarray:
    prediction = base_predictions(frame, family)
    scores = frame[f"{family.lower()}_del_vs_best_nondelete"].to_numpy(np.float64)
    prediction[scores >= np.float64(threshold)] = 2
    return prediction


def candidate_snapshot(confusion: np.ndarray, tp: int, fp: int, deletion_total: int,
                       nondeletion_total: int, sub_pred_del: int, sub_total: int,
                       speaker_pred_del: np.ndarray, speaker_support: np.ndarray,
                       threshold: float) -> dict[str, Any]:
    fn = deletion_total - tp; tn = nondeletion_total - fp
    deletion_precision = safe_div(tp, tp + fp); deletion_recall = safe_div(tp, deletion_total)
    deletion_f1 = f1(deletion_precision, deletion_recall)
    nondeletion_precision = safe_div(tn, tn + fn); nondeletion_recall = safe_div(tn, nondeletion_total)
    nondeletion_f1 = f1(nondeletion_precision, nondeletion_recall)
    class_f1 = []
    for class_id in range(3):
        class_tp = int(confusion[class_id, class_id])
        class_fp = int(confusion[:, class_id].sum() - class_tp)
        class_fn = int(confusion[class_id, :].sum() - class_tp)
        precision = safe_div(class_tp, class_tp + class_fp); recall = safe_div(class_tp, class_tp + class_fn)
        class_f1.append(f1(precision, recall))
    speaker_recall = np.divide(speaker_pred_del, speaker_support,
                               out=np.zeros_like(speaker_pred_del, dtype=np.float64), where=speaker_support > 0)
    speaker_gate = bool(np.all(speaker_recall[speaker_support >= 30] >= 0.25))
    substitution_fd = safe_div(sub_pred_del, sub_total)
    eligible = deletion_recall >= 0.45 and substitution_fd <= 0.25 and speaker_gate
    return {
        "threshold": float(threshold), "eligible": eligible,
        "binary_macro_f1": (deletion_f1 + nondeletion_f1) / 2.0,
        "deletion_precision": deletion_precision, "deletion_recall": deletion_recall,
        "deletion_f1": deletion_f1, "substitution_false_deletion": substitution_fd,
        "three_relation_macro_f1": float(np.mean(class_f1)),
        "speaker_gate": speaker_gate, "speaker_deletion_recall": speaker_recall.tolist(),
    }


def select_exact_threshold(frame: pd.DataFrame, family: str, calibration_speakers: Sequence[str]) -> dict[str, Any]:
    scores = frame[f"{family.lower()}_del_vs_best_nondelete"].to_numpy(np.float64)
    truth = frame["truth_id"].to_numpy(np.int8)
    base = base_predictions(frame, family)
    if not np.isfinite(scores).all():
        raise RuntimeError(f"Non-finite {family} calibration scores")
    speakers = frame["speaker"].to_numpy(str)
    speaker_index = {speaker: index for index, speaker in enumerate(calibration_speakers)}
    row_speaker = np.asarray([speaker_index[speaker] for speaker in speakers], dtype=np.int16)
    speaker_support = np.asarray([np.sum((speakers == speaker) & (truth == 2)) for speaker in calibration_speakers], dtype=np.int64)
    speaker_pred_del = np.zeros(len(calibration_speakers), dtype=np.int64)
    confusion = np.zeros((3, 3), dtype=np.int64)
    for expected, predicted in zip(truth, base):
        confusion[int(expected), int(predicted)] += 1
    deletion_total = int(np.sum(truth == 2)); nondeletion_total = len(truth) - deletion_total
    sub_total = int(np.sum(truth == 1)); tp = fp = sub_pred_del = 0
    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    unique_count = int(np.unique(scores).size)
    candidate_count = unique_count + 2; eligible_count = 0; selected: dict[str, Any] | None = None

    def evaluate(threshold: float) -> None:
        nonlocal eligible_count, selected
        item = candidate_snapshot(confusion, tp, fp, deletion_total, nondeletion_total,
                                  sub_pred_del, sub_total, speaker_pred_del, speaker_support, threshold)
        if item["eligible"]:
            eligible_count += 1
            key = (item["binary_macro_f1"], item["deletion_f1"], item["deletion_precision"],
                   item["three_relation_macro_f1"], item["threshold"])
            if selected is None or key > selected["selection_key"]:
                selected = {**item, "selection_key": key}

    evaluate(float(np.nextafter(np.max(scores), np.inf)))
    cursor = 0
    while cursor < len(order):
        threshold = sorted_scores[cursor]; end = cursor + 1
        while end < len(order) and sorted_scores[end] == threshold:
            end += 1
        for row_index in order[cursor:end]:
            expected = int(truth[row_index]); prior = int(base[row_index])
            confusion[expected, prior] -= 1; confusion[expected, 2] += 1
            if expected == 2:
                tp += 1; speaker_pred_del[int(row_speaker[row_index])] += 1
            else:
                fp += 1
                if expected == 1:
                    sub_pred_del += 1
        evaluate(float(threshold)); cursor = end
    evaluate(float(np.nextafter(np.min(scores), -np.inf)))
    if selected is not None:
        selected = {key: value for key, value in selected.items() if key != "selection_key"}
        selected["calibration_speaker_metrics"] = {
            speaker: {"deletion_support": int(speaker_support[index]),
                      "deletion_recall": float(selected["speaker_deletion_recall"][index]),
                      "gate_applies": bool(speaker_support[index] >= 30),
                      "gate_pass": bool(speaker_support[index] < 30 or selected["speaker_deletion_recall"][index] >= 0.25)}
            for index, speaker in enumerate(calibration_speakers)
        }
        del selected["speaker_deletion_recall"]
    return {"status": "PASS" if selected is not None else "NO_ELIGIBLE_THRESHOLD",
            "candidate_count": candidate_count, "unique_score_count": unique_count,
            "eligible_count": eligible_count, "selected": selected}


def group_deletion_metrics(frame: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    subset_truth = truth[mask]; subset_prediction = prediction[mask]
    metrics = metrics_from_predictions(subset_truth, subset_prediction) if len(subset_truth) else metrics_from_predictions(np.asarray([], dtype=np.int8), np.asarray([], dtype=np.int8))
    return {"rows": int(mask.sum()), "deletion_support": int(np.sum(subset_truth == 2)),
            "deletion_precision": metrics["deletion_precision"], "deletion_recall": metrics["deletion_recall"],
            "deletion_f1": metrics["deletion_f1"]}


def stability(thresholds: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(thresholds, dtype=np.float64)
    median = float(np.median(values)); p25, p75 = np.percentile(values, [25, 75])
    return {"count": len(values), "minimum": float(np.min(values)), "maximum": float(np.max(values)),
            "mean": float(np.mean(values)), "median": median, "standard_deviation": float(np.std(values)),
            "p25": float(p25), "p75": float(p75), "iqr": float(p75 - p25),
            "mad": float(np.median(np.abs(values - median)))}


def best_sub_diagnostic(frame: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    true_sub = truth == 1; predicted_sub = true_sub & (prediction == 1)
    support = int(np.sum(true_sub)); coverage_count = int(np.sum(predicted_sub))
    correct = int(np.sum(frame.loc[predicted_sub, "best_sub_phone"].to_numpy(str) ==
                         frame.loc[predicted_sub, "true_observed_phone"].to_numpy(str)))
    return {"true_substitution_support": support, "predicted_substitution_rows": coverage_count,
            "coverage": safe_div(coverage_count, support), "correct": correct,
            "top1_accuracy": safe_div(correct, coverage_count)}


def verify_and_load() -> tuple[pd.DataFrame, dict[str, Any]]:
    expected: dict[str, str] = {}; actual: dict[str, str] = {}
    for name, (relative, digest) in SOURCE_IDENTITIES.items():
        expected[name] = digest; actual[name] = sha256(REPO_ROOT / relative)
    if actual != expected:
        raise RuntimeError(f"Source verification failed: {[key for key in expected if expected[key] != actual.get(key)]}")
    frame = pd.read_csv(SCORES)
    frame["truth_id"] = frame["true_relation"].map(RELATION_TO_ID).astype(np.int8)
    if len(frame) != 56304 or frame["word_id"].nunique() != 16259:
        raise RuntimeError("TRAIN population mismatch")
    if frame.duplicated(["word_id", "position"]).any():
        raise RuntimeError("Duplicate TRAIN position identity")
    relations = frame["true_relation"].value_counts().to_dict()
    if relations != {"correct": 48893, "substitution": 5867, "deletion": 1544}:
        raise RuntimeError(f"TRAIN relation mismatch: {relations}")
    if sorted(frame["speaker"].unique()) != sorted(TRAIN_SPEAKERS):
        raise RuntimeError("TRAIN speaker mismatch")
    for family in FAMILIES:
        for suffix in ("keep", "delete", "best_sub", "del_vs_best_nondelete"):
            if not np.isfinite(frame[f"{family.lower()}_{suffix}"].to_numpy(np.float64)).all():
                raise RuntimeError(f"Non-finite {family} {suffix}")
    if set(frame["speaker"]) & (set(VALIDATION_SPEAKERS) | set(TEST_SPEAKERS)):
        raise RuntimeError("Non-TRAIN speaker leakage")
    if read_json(V2_FINAL)["status"] != "R4_4D1_HYPOTHESIS_THRESHOLD_TRANSFER_FAIL":
        raise RuntimeError("Unexpected frozen D1 v2 motivation status")
    return frame, {"status": "PASS", "expected": expected, "actual": actual,
                   "train_position_scores_complete": True, "identity_duplicates": 0,
                   "validation_artifacts_used_for_selection": False,
                   "validation_scoring": False, "test_accessed": False}


def future_contract(selected_family: str, robust_theta: float, oof: dict[str, Any],
                    fold_thresholds: dict[str, Any], numerical: dict[str, Any]) -> tuple[dict[str, Any], str]:
    payload = {
        "contract_id": "R4-4D2B_PREREGISTERED_FINAL_DEVELOPMENT_VALIDATION",
        "status": "FROZEN_BEFORE_R4_4D2B",
        "selected_score_family": selected_family,
        "robust_theta": robust_theta,
        "threshold_source": "ordinary float64 median of 12 TRAIN-speaker LOSO thresholds",
        "source_shas": {name: digest for name, (_path, digest) in SOURCE_IDENTITIES.items()},
        "train_only_selection": {"speakers": TRAIN_SPEAKERS, "protocol": "12-fold speaker LOSO",
                                 "oof_metrics": oof, "fold_thresholds": fold_thresholds,
                                 "validation_influence": False},
        "acoustic_checkpoint": SOURCE_IDENTITIES["checkpoint"],
        "hypothesis_and_relation_rule": {
            "inherited": "R4-4D1A",
            "decision_score": "D_i=DELETE_i-max(KEEP_i,BEST_SUB_i) within selected family",
            "decision": "D_i>=ROBUST_THETA => deletion; else KEEP>=BEST_SUB => correct; else substitution",
            "best_sub_tie": "lowest canonical phone index"
        },
        "validation_metrics_and_gates": numerical["validation_hard_gates"],
        "matched_control": numerical["matched_control"],
        "classification": numerical["result_classification"],
        "policy": {
            "one_final_development_validation_run": True,
            "r4_4d1_validation_already_consumed": True,
            "r4_4d2b_not_pristine_first_validation": True,
            "no_further_r4_calibration_if_r4_4d2b_fails": True,
            "test_remains_final_independent_holdout": True,
            "test_speakers": TEST_SPEAKERS,
            "test_closed": True,
            "training": False
        }
    }
    markdown = (
        "# R4-4D2B Preregistered Final Development Validation\n\n"
        f"Selected TRAIN-only score family: **{selected_family}**  \n"
        f"ROBUST_THETA: `{robust_theta}`\n\n"
        "The threshold is the ordinary float64 median of the 12 speaker-LOSO TRAIN thresholds. "
        "The frozen acoustic checkpoint, hypothesis construction, relation rule, validation metrics, "
        "gates, matched control, and TEST closure remain unchanged. R4-4D1 validation has already been "
        "consumed; R4-4D2B is one final iterative development validation, not a pristine first validation. "
        "If it fails, no further R4 calibration cycle is authorized.\n"
    )
    return payload, markdown


def run() -> int:
    if any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {OUTPUT}")
    started = time.perf_counter()
    frame, verification = verify_and_load()
    numerical = read_json(NUMERICAL)
    truth = frame["truth_id"].to_numpy(np.int8)
    full_train_raw_selfcheck = select_exact_threshold(frame, "RAW", TRAIN_SPEAKERS)
    if (full_train_raw_selfcheck["selected"] is None
            or full_train_raw_selfcheck["selected"]["threshold"] != 2.197946548461914):
        raise RuntimeError("Exact sweep failed to reproduce frozen R4-4D1 RAW TRAIN threshold")
    write_json(OUTPUT / "preflight.json", {
        "source_verification": verification,
        "train_population": {"words": 16259, "expected_phone_rows": 56304,
                             "relations": {"correct": 48893, "substitution": 5867, "deletion": 1544},
                             "speakers": TRAIN_SPEAKERS},
        "score_artifact_sha256": SOURCE_IDENTITIES["train_position_scores"][1],
        "score_rows_complete": True, "neural_inference": False,
        "exact_sweep_selfcheck": {
            "family": "RAW", "expected_threshold": 2.197946548461914,
            "reproduced_threshold": full_train_raw_selfcheck["selected"]["threshold"],
            "candidate_count": full_train_raw_selfcheck["candidate_count"],
            "eligible_count": full_train_raw_selfcheck["eligible_count"], "pass": True
        },
        "validation_scoring": False, "validation_threshold_evaluation": False,
        "test_accessed": False, "training": False,
    })
    write_json(OUTPUT / "score_family_contract.json", {
        "families": {
            "RAW": "raw_log_score(H)",
            "TARGET": "raw_log_score(H)/max(target_length(H),1)",
            "TIME": "raw_log_score(H)/input_length"
        },
        "decision_score": "D_i=DELETE_i-max(KEEP_i,BEST_SUB_i) within family",
        "threshold_candidates": "nextafter(min,-inf) + every sorted unique float64 score + nextafter(max,+inf)",
        "eligibility": {"calibration_deletion_recall_min": 0.45,
                        "calibration_substitution_false_deletion_max": 0.25,
                        "each_calibration_speaker_support_min": 30,
                        "each_calibration_speaker_recall_min": 0.25},
        "selection_precedence": ["binary_macro_f1", "deletion_f1", "deletion_precision",
                                 "three_relation_macro_f1", "higher_threshold"],
        "robust_theta": "ordinary numeric median of all 12 fold thresholds",
        "family_selection": ["higher OOF deletion F1", "higher OOF Binary Macro-F1",
                             "higher deletion precision", "lower substitution false-deletion",
                             "threshold MAD not compared across different scales", "RAW then TIME then TARGET"],
        "development_gates": {"binary_macro_f1_min": 0.60, "deletion_f1_min": 0.30,
                              "three_relation_macro_f1_min": 0.40},
        "classification_rule_frozen_before_results": {
            "READY": "at least one family passes eligibility and all development gates",
            "UNSTABLE": "if no READY family and every family has at least two no-eligible folds, or every family fails at least three eligible held-out speaker recall gates",
            "SIGNAL_WEAK": "otherwise, when continuous signal exists but no family passes all gates"
        },
        "validation_influence": False, "duration_feature": False, "test_closed": True
    })

    fold_output: dict[str, Any] = {}; oof_output: dict[str, Any] = {}
    speaker_output: dict[str, Any] = {}; stability_output: dict[str, Any] = {}
    full_train_output: dict[str, Any] = {}; predictions_by_family: dict[str, np.ndarray] = {}
    family_gate_results: dict[str, Any] = {}

    for family in FAMILIES:
        family_started = time.perf_counter(); fold_output[family] = {}; speaker_output[family] = {}
        oof_prediction = np.full(len(frame), -1, dtype=np.int8); thresholds: list[float] = []
        no_eligible = 0
        for held_out in TRAIN_SPEAKERS:
            calibration_speakers = [speaker for speaker in TRAIN_SPEAKERS if speaker != held_out]
            calibration_mask = frame["speaker"].to_numpy(str) != held_out
            evaluation_mask = ~calibration_mask
            result = select_exact_threshold(frame.loc[calibration_mask].reset_index(drop=True), family, calibration_speakers)
            fold_record = {"held_out_speaker": held_out, "calibration_speakers": calibration_speakers,
                           "calibration_rows": int(calibration_mask.sum()), "held_out_rows": int(evaluation_mask.sum()),
                           **result}
            if result["selected"] is None:
                no_eligible += 1; fold_output[family][held_out] = fold_record; continue
            theta = float(result["selected"]["threshold"]); thresholds.append(theta)
            held_prediction = apply_threshold(frame.loc[evaluation_mask], family, theta)
            oof_prediction[evaluation_mask] = held_prediction
            held_truth = truth[evaluation_mask]; held_metrics = metrics_from_predictions(held_truth, held_prediction)
            speaker_output[family][held_out] = {"selected_fold_theta": theta,
                "deletion_support": int(np.sum(held_truth == 2)), **held_metrics}
            fold_record["held_out_metrics"] = held_metrics; fold_output[family][held_out] = fold_record
        complete = bool(np.all(oof_prediction >= 0))
        if complete:
            aggregate = metrics_from_predictions(truth, oof_prediction)
            aggregate["best_sub_phone"] = best_sub_diagnostic(frame, truth, oof_prediction)
        else:
            aggregate = None
        speaker_gate_failures = 0
        if complete:
            for speaker in TRAIN_SPEAKERS:
                item = speaker_output[family][speaker]
                if item["deletion_support"] >= 30 and item["deletion_recall"] < 0.25:
                    speaker_gate_failures += 1
        eligible = bool(complete and no_eligible == 0 and aggregate["deletion_recall"] >= 0.45
                        and aggregate["substitution_false_deletion"] <= 0.25 and speaker_gate_failures == 0)
        development = bool(eligible and aggregate["binary_macro_f1"] >= 0.60
                           and aggregate["deletion_f1"] >= 0.30
                           and aggregate["three_relation_macro_f1"] >= 0.40)
        robust_theta = float(np.median(np.asarray(thresholds, dtype=np.float64))) if len(thresholds) == 12 else None
        stability_output[family] = {**(stability(thresholds) if thresholds else {}),
                                    "folds_no_eligible_threshold": no_eligible,
                                    "raw_scale_only_not_cross_family_comparable": True}
        if robust_theta is not None:
            full_prediction = apply_threshold(frame, family, robust_theta)
            full_metrics = metrics_from_predictions(truth, full_prediction)
            full_metrics["per_speaker_deletion_recall"] = {
                speaker: metrics_from_predictions(truth[frame["speaker"].to_numpy(str) == speaker],
                    full_prediction[frame["speaker"].to_numpy(str) == speaker])["deletion_recall"]
                for speaker in TRAIN_SPEAKERS
            }
            full_train_output[family] = {"robust_theta": robust_theta, **full_metrics}
        else:
            full_train_output[family] = {"robust_theta": None, "status": "UNAVAILABLE"}
        oof_output[family] = {"complete_oof": complete, "rows_predicted_once": int(np.sum(oof_prediction >= 0)),
                              "metrics": aggregate, "seconds": time.perf_counter() - family_started}
        family_gate_results[family] = {"all_12_fold_thresholds": no_eligible == 0,
            "folds_no_eligible_threshold": no_eligible,
            "oof_deletion_recall": None if aggregate is None else aggregate["deletion_recall"],
            "oof_substitution_false_deletion": None if aggregate is None else aggregate["substitution_false_deletion"],
            "speaker_gate_failures": speaker_gate_failures, "family_eligible": eligible,
            "development_binary_macro_f1": False if aggregate is None else aggregate["binary_macro_f1"] >= 0.60,
            "development_deletion_f1": False if aggregate is None else aggregate["deletion_f1"] >= 0.30,
            "development_three_relation_macro_f1": False if aggregate is None else aggregate["three_relation_macro_f1"] >= 0.40,
            "development_pass": development}
        predictions_by_family[family] = oof_prediction

    ready = [family for family in FAMILIES if family_gate_results[family]["development_pass"]]
    if ready:
        selected = max(ready, key=lambda family: (
            oof_output[family]["metrics"]["deletion_f1"],
            oof_output[family]["metrics"]["binary_macro_f1"],
            oof_output[family]["metrics"]["deletion_precision"],
            -oof_output[family]["metrics"]["substitution_false_deletion"],
            -SIMPLICITY[family],
        ))
        status = "R4_4D2A_ROBUST_CALIBRATION_READY"
    else:
        selected = None
        all_multiple_missing = all(family_gate_results[family]["folds_no_eligible_threshold"] >= 2 for family in FAMILIES)
        all_speaker_collapse = all(family_gate_results[family]["speaker_gate_failures"] >= 3 for family in FAMILIES)
        status = "R4_4D2A_CALIBRATION_UNSTABLE" if (all_multiple_missing or all_speaker_collapse) else "R4_4D2A_CALIBRATION_SIGNAL_WEAK"

    eligible_families = [family for family in FAMILIES if family_gate_results[family]["family_eligible"]]
    selected_theta = None if selected is None else full_train_output[selected]["robust_theta"]
    continuous_source = read_json(R4D0_AUC); deletion_sub_source = read_json(R4D0_DEL_SUB)
    continuous = {
        family: {
            "deletion_vs_nondeletion": continuous_source["deletion_vs_nondeletion"][family],
            "deletion_vs_substitution": deletion_sub_source[family]
        } for family in FAMILIES
    }

    phone_output: dict[str, Any] = {}; position_output: dict[str, Any] = {}; multi_output: dict[str, Any] = {}
    report_families = eligible_families if eligible_families else FAMILIES
    for family in report_families:
        prediction = predictions_by_family[family]
        if np.any(prediction < 0):
            continue
        phone_output[family] = {}
        for phone in sorted(frame["expected_phone"].unique()):
            mask = frame["expected_phone"].to_numpy(str) == phone
            phone_output[family][phone] = group_deletion_metrics(frame, truth, prediction, mask)
        group_a = frame["expected_phone"].isin(["D", "T", "R", "L"]).to_numpy(bool)
        phone_output[family]["groups"] = {
            "D_T_R_L": group_deletion_metrics(frame, truth, prediction, group_a),
            "OTHER": group_deletion_metrics(frame, truth, prediction, ~group_a)
        }
        position_output[family] = {
            name: group_deletion_metrics(frame, truth, prediction, frame["position_group"].to_numpy(str) == name)
            for name in ("initial", "medial", "final", "single")
        }
        multi_output[family] = {
            "pure_deletion": group_deletion_metrics(frame, truth, prediction,
                ((frame["word_deletions"] > 0) & (frame["word_substitutions"] == 0)).to_numpy(bool)),
            "substitution_plus_deletion": group_deletion_metrics(frame, truth, prediction,
                ((frame["word_deletions"] > 0) & (frame["word_substitutions"] > 0)).to_numpy(bool)),
            "multiple_deletion": group_deletion_metrics(frame, truth, prediction,
                (frame["word_deletions"] > 1).to_numpy(bool))
        }

    future_info = None
    if status == "R4_4D2A_ROBUST_CALIBRATION_READY":
        future_json = OUTPUT / "r4_4d2b_preregistered_validation_design.json"
        future_md = OUTPUT / "r4_4d2b_preregistered_validation_design.md"
        payload, markdown = future_contract(selected, selected_theta, oof_output[selected]["metrics"],
                                             {speaker: fold_output[selected][speaker]["selected"]["threshold"] for speaker in TRAIN_SPEAKERS}, numerical)
        write_json(future_json, payload); future_md.write_text(markdown, encoding="utf-8")
        future_info = {"json_path": str(future_json.relative_to(REPO_ROOT)), "json_sha256": sha256(future_json),
                       "markdown_path": str(future_md.relative_to(REPO_ROOT)), "markdown_sha256": sha256(future_md)}

    write_json(OUTPUT / "loso_fold_thresholds.json", fold_output)
    write_json(OUTPUT / "loso_metrics_by_family.json", oof_output)
    write_json(OUTPUT / "loso_speaker_metrics.json", speaker_output)
    write_json(OUTPUT / "threshold_stability.json", stability_output)
    write_json(OUTPUT / "full_train_robust_theta_metrics.json", full_train_output)
    write_json(OUTPUT / "phone_metrics.json", phone_output)
    write_json(OUTPUT / "position_metrics.json", position_output)
    write_json(OUTPUT / "multi_error_metrics.json", multi_output)
    write_json(OUTPUT / "continuous_auc_metrics.json", continuous)
    selection = {"status": status, "eligible_families": eligible_families,
                 "development_ready_families": ready, "selected_family": selected,
                 "robust_theta": selected_theta, "family_gate_results": family_gate_results,
                 "selection_precedence": ["OOF deletion F1", "OOF Binary Macro-F1", "deletion precision",
                                          "lower substitution false-deletion", "RAW then TIME then TARGET"],
                 "cross_family_threshold_stability_used": False,
                 "future_validation_contract": future_info,
                 "validation_used_for_selection": False}
    write_json(OUTPUT / "selection_result.json", selection)
    write_json(OUTPUT / "final_status.json", {"status": status, "selected_family": selected,
        "robust_theta": selected_theta, "validation_scoring": False,
        "validation_candidate_metrics": False, "validation_used_for_selection": False,
        "test_accessed": False, "training": False,
        "stop_r4_deletion_calibration_if_not_ready": status != "R4_4D2A_ROBUST_CALIBRATION_READY"})
    selected_line = "none" if selected is None else f"{selected} at ROBUST_THETA={selected_theta}"
    report = (
        f"# R4-4D2A TRAIN-Only Speaker-Robust Calibration Audit\n\nFinal: **{status}**\n\n"
        f"Selected: **{selected_line}**\n\nAll results are 12-fold TRAIN-speaker LOSO. "
        "No VALIDATION scoring, TEST access, acoustic inference, or neural training occurred.\n"
    )
    (OUTPUT / "r4_4d2a_report.md").write_text(report, encoding="utf-8")
    write_json(OUTPUT / "compute_report.json", {"seconds": time.perf_counter() - started,
        "source": "frozen TRAIN score CSV", "acoustic_inference": False,
        "families": FAMILIES, "folds_per_family": 12,
        "validation_scoring": False, "test_accessed": False})
    hashes = {path.name: sha256(path) for path in sorted(OUTPUT.iterdir())
              if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(OUTPUT / "artifact_hashes.json", {"algorithm": "SHA-256", "files": hashes,
                                                  "note": "manifest excludes itself"})
    print(json.dumps({"status": status, "selected_family": selected,
                      "robust_theta": selected_theta, "output": str(OUTPUT)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    raise SystemExit(run())
