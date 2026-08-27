"""One frozen R5-3A TRAIN-only speaker-LOSO development execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning


RESEARCH = Path(r"C:\Users\Admin\Documents\KLTN\pronunciation-assistant-research")
R5_3A = RESEARCH / "ai-training/experiments/r5_3a_evidence_separated_relation_scoring"
R5_1A = RESEARCH / "ai-training/experiments/r5_1a_alignability_safe_addition_scoring"
R5_2B = RESEARCH / "ai-training/experiments/r5_2b_relation_competition"

sys.path.insert(0, str(R5_3A))
import r5_3a_classifier as classifier_module  # noqa: E402
from r5_3a_features import FEATURE_ORDER, construct_feature_matrix  # noqa: E402
from r5_3a_loso import run_loso  # noqa: E402


SPEAKERS = ["BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA"]

EXPECTED = {
    "contract": "A1594B45A8EE9C542FEFF28695DA27D59D07C4A37596EAB1D4308161A7B583B3",
    "preregistration": "2239C047504B4B6CA212A40D0284DD38E626CF6E73D541E8AF036311BC8A6CB1",
    "contract_manifest": "CB3C4EC1E2402F3821BF61CA9302731D56CE6FCED2B025295FF6FC780D521D6D",
    "static_manifest": "1BC0406C37193DF26BDCBA7A4DE0E6CE99D31D14785472E378AD67654042F89C",
    "r5_1a_manifest": "C9343A75CE26C2BEECA388EBA855E91AE0992D22C0C5D785308D0A98B89A3CD6",
    "r5_2b_manifest": "37F3C86FF11526B8AB54D173937A0F488125A1D0B610032CC8A5D47B11602387",
    "r5_2c_manifest": "C18621208A35074B7681EFC95261B73147B4A6E3716F17AF0AB0471F8B328F90",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "checkpoint": "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085",
}

R5_1A_COMPARATOR = {
    "auc_all": 0.7734833025081417,
    "auc_correct": 0.8023528214095370,
    "binary_macro_f1": 0.5551978767901391,
    "addition_f1": 0.1379980563654033,
    "correct_far": 0.03491295938104449,
    "sub_far": 0.05016116035455278,
    "delete_far": 0.03349964362081254,
    "event_f1": 0.04389312977099236,
}

R5_2B_CONTEXT = {
    "auc_all": 0.6878413803490975,
    "auc_correct": 0.669239360205041,
    "binary_macro_f1": 0.5304030323273805,
    "addition_f1": 0.08637873754152824,
    "correct_far": 0.04323017408123791,
    "sub_far": 0.017123287671232876,
    "delete_far": 0.006414825374198147,
    "event_f1": 0.03253796095444685,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("execution artifacts must not contain non-finite JSON numbers")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(clean(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_manifest(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("artifacts", manifest.get("files", []))
    failures: list[str] = []
    for entry in entries:
        rel = entry.get("relative_path", entry.get("path"))
        size = entry.get("byte_size", entry.get("size_bytes"))
        target = root / rel
        if not target.is_file():
            failures.append(f"missing:{rel}")
            continue
        if target.stat().st_size != int(size):
            failures.append(f"size:{rel}")
        if sha256(target) != str(entry["sha256"]).upper():
            failures.append(f"hash:{rel}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "entries": len(entries),
        "failures": failures,
        "pass": not failures,
    }


def index_unique(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    result: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        identity = str(row["source_identity"])
        if identity in result:
            duplicates.append(identity)
        else:
            result[identity] = row
    return result, sorted(set(duplicates))


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


def auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if not np.all(np.isfinite(scores)):
        raise ValueError("continuous probabilities must be finite")
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    positive = labels.astype(bool)
    prediction = predictions.astype(bool)
    negative = ~positive
    tp = int(np.count_nonzero(prediction & positive))
    fp = int(np.count_nonzero(prediction & negative))
    fn = int(np.count_nonzero(~prediction & positive))
    tn = int(np.count_nonzero(~prediction & negative))
    addition_f1 = f1(tp, fp, fn)
    nonaddition_f1 = f1(tn, fn, fp)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    tnr = 0.0 if tn + fp == 0 else tn / (tn + fp)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": (tp + tn) / labels.size,
        "balanced_accuracy": (recall + tnr) / 2.0,
        "binary_macro_f1": (addition_f1 + nonaddition_f1) / 2.0,
        "addition_precision": precision,
        "addition_recall": recall,
        "addition_f1": addition_f1,
        "nonaddition_f1": nonaddition_f1,
    }


def false_addition_rate(predictions: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    support = int(mask.sum())
    count = int(np.count_nonzero(predictions & mask))
    return {
        "support": support,
        "predicted_addition": count,
        "false_addition_rate": count / support if support else None,
    }


def position_from_boundary(boundary: int, expected_length: int) -> str:
    if boundary == 0:
        return "BEFORE_FIRST"
    if boundary == expected_length:
        return "AFTER_FINAL"
    return "BETWEEN"


def counter_event_metrics(true_counter: Counter[tuple[str, int, str]], pred_counter: Counter[tuple[str, int, str]]) -> dict[str, Any]:
    tp = int(sum((true_counter & pred_counter).values()))
    predicted = int(sum(pred_counter.values()))
    true = int(sum(true_counter.values()))
    fp = predicted - tp
    fn = true - tp
    precision = 0.0 if predicted == 0 else tp / predicted
    recall = 0.0 if true == 0 else tp / true
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1(tp, fp, fn),
        "true_events": true,
        "predicted_events": predicted,
    }


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)

    identity_manifests = {
        "contract_manifest": audit_manifest(R5_3A, "R5_3A_CONTRACT_MANIFEST.json"),
        "static_manifest": audit_manifest(R5_3A, "R5_3A_STATIC_MANIFEST.json"),
        "r5_1a_manifest": audit_manifest(R5_1A, "R5_1A_EXECUTION_MANIFEST.json"),
        "r5_2b_manifest": audit_manifest(R5_2B, "R5_2B_TC1_PA1_ENV_EXECUTION_MANIFEST.json"),
        "r5_2c_manifest": audit_manifest(R5_2B, "R5_2C_MANIFEST.json"),
    }
    direct_paths = {
        "contract": R5_3A / "R5_3A_DEVELOPMENT_CONTRACT.json",
        "preregistration": R5_3A / "R5_3A_PREREGISTRATION.md",
        "v4": RESEARCH / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv",
        "checkpoint": RESEARCH / "ai-training/experiments/r4_4c2_bigru_ctc_seed42/R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt",
    }
    direct_checks = {
        name: {
            "path": str(path),
            "expected_sha256": EXPECTED[name],
            "actual_sha256": sha256(path),
            "match": sha256(path) == EXPECTED[name],
            "loaded": False if name == "checkpoint" else None,
        }
        for name, path in direct_paths.items()
    }
    expected_manifest_hashes = {
        "contract_manifest": EXPECTED["contract_manifest"],
        "static_manifest": EXPECTED["static_manifest"],
        "r5_1a_manifest": EXPECTED["r5_1a_manifest"],
        "r5_2b_manifest": EXPECTED["r5_2b_manifest"],
        "r5_2c_manifest": EXPECTED["r5_2c_manifest"],
    }
    identity_pass = all(
        item["pass"] and item["sha256"] == expected_manifest_hashes[name]
        for name, item in identity_manifests.items()
    ) and all(item["match"] for item in direct_checks.values())
    implementation_hashes = {
        name: sha256(R5_3A / name)
        for name in ["r5_3a_features.py", "r5_3a_classifier.py", "r5_3a_loso.py", "r5_3a_threshold.py"]
    }
    static_manifest = json.loads((R5_3A / "R5_3A_STATIC_MANIFEST.json").read_text(encoding="utf-8"))
    static_index = {entry["relative_path"]: entry["sha256"] for entry in static_manifest["artifacts"]}
    implementation_match = all(
        implementation_hashes[name] == static_index[name] for name in implementation_hashes
    )
    identity = {
        "status": "PASS" if identity_pass and implementation_match else "R5_3A_EXECUTION_BLOCKED_IDENTITY",
        "manifests": identity_manifests,
        "direct_checks": direct_checks,
        "implementation_hashes": implementation_hashes,
        "static_implementation_match": implementation_match,
        "sklearn_version": sklearn.__version__,
        "sklearn_version_match": sklearn.__version__ == "1.8.0",
        "checkpoint_loaded": False,
    }
    write_json(output / "r5_3a_execution_identity.json", identity)
    if not identity_pass or not implementation_match or sklearn.__version__ != "1.8.0":
        raise RuntimeError("R5_3A_EXECUTION_BLOCKED_IDENTITY")

    old_rows = read_jsonl(R5_1A / "r5_1a_train_scores.jsonl")
    new_rows = read_jsonl(R5_2B / "r5_2b_tc1_pa1_env_train_scores.jsonl")
    old_index, old_duplicates = index_unique(old_rows)
    new_index, new_duplicates = index_unique(new_rows)
    old_keys = set(old_index)
    new_keys = set(new_index)
    ordered_ids = [str(row["source_identity"]) for row in new_rows]
    metadata_mismatches: list[str] = []
    for identity_key in sorted(old_keys & new_keys):
        old = old_index[identity_key]
        new = new_index[identity_key]
        if (
            old["speaker"] != new["speaker"]
            or old["word"] != new["word"]
            or old["expected_sequence"] != new["expected_sequence"]
            or bool(old["addition_label"]) != bool(new["addition_label"])
            or old["ground_truth_events"] != new["ground_truth_events"]
        ):
            metadata_mismatches.append(identity_key)
    ordered_old = [old_index[identity_key] for identity_key in ordered_ids if identity_key in old_index]
    labels = np.asarray([int(bool(row["addition_label"])) for row in ordered_old], dtype=np.int64)
    observed_speakers = sorted({str(row["speaker"]) for row in ordered_old})
    join_pass = (
        len(old_rows) == len(new_rows) == 16582
        and len(old_keys & new_keys) == 16582
        and not old_duplicates
        and not new_duplicates
        and not (old_keys - new_keys)
        and not (new_keys - old_keys)
        and not metadata_mismatches
        and int(labels.sum()) == 323
        and int((labels == 0).sum()) == 16259
        and observed_speakers == sorted(SPEAKERS)
    )
    join_audit = {
        "status": "PASS" if join_pass else "R5_3A_EXECUTION_BLOCKED_ROW_IDENTITY",
        "matched_rows": len(old_keys & new_keys),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
        "r5_1a_duplicates": len(old_duplicates),
        "r5_2b_duplicates": len(new_duplicates),
        "missing_r5_1a": len(new_keys - old_keys),
        "missing_r5_2b": len(old_keys - new_keys),
        "metadata_mismatches": len(metadata_mismatches),
        "fuzzy_matching": False,
        "excluded_rows": 0,
        "speakers": observed_speakers,
    }
    write_json(output / "r5_3a_train_join_audit.json", join_audit)
    if not join_pass:
        raise RuntimeError("R5_3A_EXECUTION_BLOCKED_ROW_IDENTITY")

    features, feature_ids = construct_feature_matrix(old_rows, new_rows)
    if feature_ids != ordered_ids or features.shape != (16582, 3) or not np.all(np.isfinite(features)):
        raise RuntimeError("R5_3A_EXECUTION_BLOCKED_FEATURE_PROVENANCE")
    feature_rows: list[dict[str, Any]] = []
    loso_rows: list[dict[str, Any]] = []
    for index, identity_key in enumerate(feature_ids):
        old = old_index[identity_key]
        new = new_index[identity_key]
        a_value, s_value, d_value = (float(value) for value in features[index])
        feature_rows.append(
            {
                "source_identity": identity_key,
                "speaker": old["speaker"],
                "addition_label": bool(old["addition_label"]),
                "A": a_value,
                "S": s_value,
                "D": d_value,
            }
        )
        loso_rows.append(
            {
                "row_id": identity_key,
                "speaker": old["speaker"],
                "A": np.float64(a_value),
                "S": np.float64(s_value),
                "D": np.float64(d_value),
                "label": int(bool(old["addition_label"])),
                "correct_only_negative": bool(new["relation_cohorts"]["correct_only"]),
                "best_insert_phone": new["best_insert_phone"],
                "best_insert_boundary": int(new["best_insert_boundary"]),
            }
        )
    write_jsonl(output / "r5_3a_train_features.jsonl", feature_rows)

    captured_models: list[Any] = []
    original_factory = classifier_module.make_classifier

    def capturing_factory() -> Any:
        model = original_factory()
        captured_models.append(model)
        return model

    classifier_module.make_classifier = capturing_factory
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            oof_outputs, fold_audits = run_loso(loso_rows, SPEAKERS)
    finally:
        classifier_module.make_classifier = original_factory
    convergence_warnings = [warning for warning in caught if issubclass(warning.category, ConvergenceWarning)]
    if len(captured_models) != 12 or convergence_warnings:
        raise RuntimeError("R5_3A_EXECUTION_TECHNICAL_FAILURE_CONVERGENCE")
    n_iters = [int(model.n_iter_[0]) for model in captured_models]
    if any(value >= 1000 for value in n_iters):
        raise RuntimeError("R5_3A_EXECUTION_TECHNICAL_FAILURE_CONVERGENCE")

    output_index = {str(row["row_id"]): row for row in oof_outputs}
    if len(output_index) != 16582 or set(output_index) != set(feature_ids):
        raise RuntimeError("R5_3A_EXECUTION_BLOCKED_ROW_IDENTITY")
    probabilities = np.asarray([float(output_index[key]["probability"]) for key in feature_ids], dtype=np.float64)
    thresholds = np.asarray([float(output_index[key]["threshold"]) for key in feature_ids], dtype=np.float64)
    predictions = np.asarray([bool(output_index[key]["predicted_addition"]) for key in feature_ids], dtype=bool)
    if not np.all(np.isfinite(probabilities)) or probabilities.shape != (16582,):
        raise RuntimeError("R5_3A_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION")

    fold_models: list[dict[str, Any]] = []
    fold_thresholds: list[dict[str, Any]] = []
    for speaker, audit, model, n_iter in zip(SPEAKERS, fold_audits, captured_models, n_iters):
        fold_models.append(
            {
                "heldout_speaker": speaker,
                "calibration_rows": len(audit["calibration_row_ids"]),
                "heldout_rows": len(audit["heldout_row_ids"]),
                "scaler_mean": audit["scaler_mean"],
                "scaler_scale": audit["scaler_scale"],
                "scaler_var": audit["scaler_var"],
                "intercept": float(audit["model_intercept"][0]),
                "coefficient_A": float(audit["model_coefficients"][0][0]),
                "coefficient_S": float(audit["model_coefficients"][0][1]),
                "coefficient_D": float(audit["model_coefficients"][0][2]),
                "n_iter": n_iter,
                "converged": True,
            }
        )
        fold_thresholds.append(
            {
                "heldout_speaker": speaker,
                "calibration_rows": len(audit["calibration_row_ids"]),
                "heldout_rows": len(audit["heldout_row_ids"]),
                "candidate_count": int(audit["threshold_candidate_count"]),
                "selected_theta": float(audit["threshold"]),
            }
        )
    write_json(output / "r5_3a_fold_models.json", {"feature_order": list(FEATURE_ORDER), "folds": fold_models})
    write_json(output / "r5_3a_fold_thresholds.json", {"speaker_order": SPEAKERS, "folds": fold_thresholds, "thresholds_in_speaker_order": [item["selected_theta"] for item in fold_thresholds]})

    oof_scores: list[dict[str, Any]] = []
    oof_predictions: list[dict[str, Any]] = []
    for index, identity_key in enumerate(feature_ids):
        old = old_index[identity_key]
        new = new_index[identity_key]
        score_row = {
            "source_identity": identity_key,
            "speaker": old["speaker"],
            "addition_label": bool(labels[index]),
            "addition_probability": float(probabilities[index]),
        }
        oof_scores.append(score_row)
        predicted_event = None
        if bool(predictions[index]):
            predicted_event = {
                "phone": new["best_insert_phone"],
                "phone_index": int(new["best_insert_phone_index"]),
                "boundary": int(new["best_insert_boundary"]),
            }
        oof_predictions.append(
            {
                **score_row,
                "theta": float(thresholds[index]),
                "predicted_addition": bool(predictions[index]),
                "BEST_INSERT_phone": new["best_insert_phone"],
                "BEST_INSERT_phone_index": int(new["best_insert_phone_index"]),
                "BEST_INSERT_boundary": int(new["best_insert_boundary"]),
                "predicted_event": predicted_event,
            }
        )
    write_jsonl(output / "r5_3a_oof_scores.jsonl", oof_scores)
    write_jsonl(output / "r5_3a_oof_predictions.jsonl", oof_predictions)

    correct_mask = np.asarray([bool(new_index[key]["relation_cohorts"]["correct_only"]) for key in feature_ids], dtype=bool)
    sub_mask = np.asarray([bool(new_index[key]["relation_cohorts"]["substitution_negative"]) for key in feature_ids], dtype=bool)
    delete_mask = np.asarray([bool(new_index[key]["relation_cohorts"]["deletion_negative"]) for key in feature_ids], dtype=bool)
    positive_mask = labels.astype(bool)
    continuous = {
        "authoritative_output": "predict_proba(X_scaled) class identity 1",
        "addition_vs_all_negative_roc_auc": auc_mann_whitney(probabilities, positive_mask),
        "addition_vs_correct_only_roc_auc": auc_mann_whitney(probabilities[positive_mask | correct_mask], positive_mask[positive_mask | correct_mask]),
        "addition_vs_substitution_containing_roc_auc": auc_mann_whitney(probabilities[positive_mask | sub_mask], positive_mask[positive_mask | sub_mask]),
        "addition_vs_deletion_containing_roc_auc": auc_mann_whitney(probabilities[positive_mask | delete_mask], positive_mask[positive_mask | delete_mask]),
        "AUC_implementation": "deterministic Mann-Whitney average-rank",
    }
    write_json(output / "r5_3a_continuous_metrics.json", continuous)
    binary = binary_metrics(labels, predictions)
    write_json(output / "r5_3a_binary_metrics.json", binary)

    cohort_metrics = {
        "correct_only": false_addition_rate(predictions, correct_mask),
        "substitution_negative": false_addition_rate(predictions, sub_mask),
        "deletion_negative": false_addition_rate(predictions, delete_mask),
    }
    for name, comparator_key in [("correct_only", "correct_far"), ("substitution_negative", "sub_far"), ("deletion_negative", "delete_far")]:
        value = cohort_metrics[name]["false_addition_rate"]
        cohort_metrics[name]["delta_vs_R5_1A"] = value - R5_1A_COMPARATOR[comparator_key]
        cohort_metrics[name]["descriptive_delta_vs_R5_2B"] = value - R5_2B_CONTEXT[comparator_key]
    write_json(output / "r5_3a_relation_cohort_metrics.json", cohort_metrics)

    true_by_position: dict[str, Counter[tuple[str, int, str]]] = {name: Counter() for name in ["BEFORE_FIRST", "BETWEEN", "AFTER_FINAL"]}
    predicted_by_position: dict[str, Counter[tuple[str, int, str]]] = {name: Counter() for name in ["BEFORE_FIRST", "BETWEEN", "AFTER_FINAL"]}
    for index, identity_key in enumerate(feature_ids):
        old = old_index[identity_key]
        new = new_index[identity_key]
        for event in old["ground_truth_events"]:
            true_by_position[str(event["position"])][(identity_key, int(event["boundary"]), str(event["phone"]))] += 1
        if predictions[index]:
            boundary = int(new["best_insert_boundary"])
            position = position_from_boundary(boundary, len(new["expected_sequence"]))
            predicted_by_position[position][(identity_key, boundary, str(new["best_insert_phone"]))] += 1
    true_all = sum(true_by_position.values(), Counter())
    predicted_all = sum(predicted_by_position.values(), Counter())
    event = {
        "exact_event": counter_event_metrics(true_all, predicted_all),
        "by_position": {
            position: counter_event_metrics(true_by_position[position], predicted_by_position[position])
            for position in ["BEFORE_FIRST", "BETWEEN", "AFTER_FINAL"]
        },
        "BEST_INSERT_changed": False,
        "matching": "Counter intersection on (source_identity,boundary,canonical_phone)",
    }
    write_json(output / "r5_3a_event_metrics.json", event)

    coefficient_values = {
        "A": np.asarray([row["coefficient_A"] for row in fold_models], dtype=np.float64),
        "S": np.asarray([row["coefficient_S"] for row in fold_models], dtype=np.float64),
        "D": np.asarray([row["coefficient_D"] for row in fold_models], dtype=np.float64),
    }
    coefficient_summary = {
        name: {
            "median": float(np.median(values)),
            "positive_folds": int(np.count_nonzero(values > 0.0)),
            "zero_folds": int(np.count_nonzero(values == 0.0)),
            "negative_folds": int(np.count_nonzero(values < 0.0)),
            "sign_consistent": bool(np.all(values > 0.0) or np.all(values < 0.0) or np.all(values == 0.0)),
        }
        for name, values in coefficient_values.items()
    }
    coefficient_diagnostics = {
        "standardized_feature_order": list(FEATURE_ORDER),
        "folds": fold_models,
        "summary": coefficient_summary,
        "descriptive_only": True,
    }
    write_json(output / "r5_3a_coefficient_diagnostics.json", coefficient_diagnostics)

    gate_definitions = {
        "G1": (continuous["addition_vs_all_negative_roc_auc"], ">=", 0.70),
        "G2": (continuous["addition_vs_correct_only_roc_auc"], ">=", 0.70),
        "G3": (binary["binary_macro_f1"], ">", R5_1A_COMPARATOR["binary_macro_f1"]),
        "G4": (binary["addition_f1"], ">", R5_1A_COMPARATOR["addition_f1"]),
        "G5": (cohort_metrics["correct_only"]["false_addition_rate"], "<=", R5_1A_COMPARATOR["correct_far"]),
        "G6": (cohort_metrics["substitution_negative"]["false_addition_rate"], "<", R5_1A_COMPARATOR["sub_far"]),
        "G7": (cohort_metrics["deletion_negative"]["false_addition_rate"], "<", R5_1A_COMPARATOR["delete_far"]),
        "G8": (event["exact_event"]["f1"], ">=", R5_1A_COMPARATOR["event_f1"]),
    }
    gates: dict[str, Any] = {}
    for name, (value, operator, threshold) in gate_definitions.items():
        passed = {
            ">=": value >= threshold,
            ">": value > threshold,
            "<=": value <= threshold,
            "<": value < threshold,
        }[operator]
        gates[name] = {
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "result": "PASS" if passed else "FAIL",
            "full_precision": True,
        }
    passed_count = sum(item["result"] == "PASS" for item in gates.values())
    threshold_values = np.asarray([row["selected_theta"] for row in fold_thresholds], dtype=np.float64)
    robust = {
        "status": "R5_3A_ROBUST_THETA" if passed_count == 8 else "R5_3A_ROBUST_THETA_NOT_AUTHORIZED",
        "thresholds_speaker_order": threshold_values.tolist(),
        "sorted_thresholds": np.sort(threshold_values).tolist(),
        "value": float(np.median(threshold_values)) if passed_count == 8 else None,
    }
    gate_results = {
        "gates": gates,
        "passed_count": passed_count,
        "total": 8,
        "all_pass": passed_count == 8,
        "robust_threshold": robust,
    }
    write_json(output / "r5_3a_gate_results.json", gate_results)

    protocol = {
        "execution_count": 1,
        "neural_training": False,
        "checkpoint_loaded": False,
        "checkpoint_inference": False,
        "audio_paths_resolved": False,
        "audio_accessed": False,
        "new_acoustic_scores_created": False,
        "real_classifier_fitting": True,
        "folds_fitted": 12,
        "real_threshold_selection": True,
        "thresholds_selected": 12,
        "validation_paths_resolved": False,
        "validation_accessed": False,
        "test_paths_resolved": False,
        "test_accessed": False,
        "frozen_implementation_modified": False,
        "frozen_contract_modified": False,
        "rerun_performed": False,
        "hyperparameter_search": False,
        "feature_search": False,
        "heldout_label_leakage": False,
        "convergence_warnings": len(convergence_warnings),
    }
    write_json(output / "r5_3a_protocol_audit.json", protocol)
    final_status = "R5_3A_EVIDENCE_SEPARATION_DEVELOPMENT_PASS" if passed_count == 8 else "R5_3A_EVIDENCE_SEPARATION_DEVELOPMENT_NOT_CONFIRMED"
    final = {
        "status": final_status,
        "gates_passed": passed_count,
        "gates_total": 8,
        "robust_threshold": robust,
        "primary_comparator": "R5-1A",
        "R5_2B_role": "descriptive mechanistic context only",
    }
    write_json(output / "r5_3a_final_status.json", final)

    report = f"""# R5-3A Frozen TRAIN Development Result

## Status

`{final_status}`

## Exact population and method

- Exact joined TRAIN rows: 16,582; positive 323; negative 16,259.
- Feature order: `[A,S,D]`; all finite; no forbidden feature used.
- Exact 12-fold TRAIN speaker LOSO with calibration-only StandardScaler, fixed balanced L2 LogisticRegression, and class-1 `predict_proba`.
- All 12 folds converged within max_iter=1000.

## Continuous metrics

- Addition/all-negative AUC: {continuous['addition_vs_all_negative_roc_auc']:.17g}
- Addition/correct-only AUC: {continuous['addition_vs_correct_only_roc_auc']:.17g}
- Addition/substitution-containing AUC: {continuous['addition_vs_substitution_containing_roc_auc']:.17g}
- Addition/deletion-containing AUC: {continuous['addition_vs_deletion_containing_roc_auc']:.17g}

## OOF decision metrics

- TP/FP/FN/TN: {binary['TP']} / {binary['FP']} / {binary['FN']} / {binary['TN']}
- Binary Macro-F1: {binary['binary_macro_f1']:.17g}
- Addition P/R/F1: {binary['addition_precision']:.17g} / {binary['addition_recall']:.17g} / {binary['addition_f1']:.17g}
- Correct/SUB/DELETE FAR: {cohort_metrics['correct_only']['false_addition_rate']:.17g} / {cohort_metrics['substitution_negative']['false_addition_rate']:.17g} / {cohort_metrics['deletion_negative']['false_addition_rate']:.17g}
- Exact-event F1: {event['exact_event']['f1']:.17g}

## Frozen gates

{passed_count} / 8 PASS.

{chr(10).join(f'- {name}: {item["result"]}' for name, item in gates.items())}

Robust threshold: `{robust['status']}`{f" = {robust['value']:.17g}" if robust['value'] is not None else ""}

## Protocol

This was the single authorized real TRAIN classifier execution. No neural checkpoint was loaded, no audio was read, and no new acoustic score was created. VALIDATION and TEST were not resolved or accessed. The frozen contract and static implementation were unchanged. No rerun or retuning occurred.
"""
    (output / "R5_3A_TRAIN_DEVELOPMENT_RESULT.md").write_text(report, encoding="utf-8")

    artifact_paths: dict[str, Path] = {
        "r5_3a_train_execution.py": Path(__file__).resolve(),
        "r5_3a_execution_identity.json": output / "r5_3a_execution_identity.json",
        "r5_3a_train_join_audit.json": output / "r5_3a_train_join_audit.json",
        "r5_3a_train_features.jsonl": output / "r5_3a_train_features.jsonl",
        "r5_3a_fold_models.json": output / "r5_3a_fold_models.json",
        "r5_3a_fold_thresholds.json": output / "r5_3a_fold_thresholds.json",
        "r5_3a_oof_scores.jsonl": output / "r5_3a_oof_scores.jsonl",
        "r5_3a_oof_predictions.jsonl": output / "r5_3a_oof_predictions.jsonl",
        "r5_3a_continuous_metrics.json": output / "r5_3a_continuous_metrics.json",
        "r5_3a_binary_metrics.json": output / "r5_3a_binary_metrics.json",
        "r5_3a_relation_cohort_metrics.json": output / "r5_3a_relation_cohort_metrics.json",
        "r5_3a_event_metrics.json": output / "r5_3a_event_metrics.json",
        "r5_3a_coefficient_diagnostics.json": output / "r5_3a_coefficient_diagnostics.json",
        "r5_3a_gate_results.json": output / "r5_3a_gate_results.json",
        "r5_3a_protocol_audit.json": output / "r5_3a_protocol_audit.json",
        "r5_3a_final_status.json": output / "r5_3a_final_status.json",
        "R5_3A_TRAIN_DEVELOPMENT_RESULT.md": output / "R5_3A_TRAIN_DEVELOPMENT_RESULT.md",
    }
    artifacts = [
        {
            "relative_path": name,
            "byte_size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in artifact_paths.items()
    ]
    manifest = {
        "stage": "R5-3A one frozen TRAIN-only development execution",
        "status": final_status,
        "manifest_self_excluded": True,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "hash_audit": "HASH_AUDIT_PASS",
    }
    write_json(output / "R5_3A_EXECUTION_MANIFEST.json", manifest)
    failures = []
    for entry in artifacts:
        path = artifact_paths[entry["relative_path"]]
        if path.stat().st_size != entry["byte_size"] or sha256(path) != entry["sha256"]:
            failures.append(entry["relative_path"])
    if failures:
        raise RuntimeError(f"execution hash audit failed: {failures}")
    print(
        json.dumps(
            {
                "status": final_status,
                "gates_passed": passed_count,
                "gates_total": 8,
                "manifest_sha256": sha256(output / "R5_3A_EXECUTION_MANIFEST.json"),
                "artifact_count": len(artifacts),
                "hash_audit": "HASH_AUDIT_PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.output.resolve())
