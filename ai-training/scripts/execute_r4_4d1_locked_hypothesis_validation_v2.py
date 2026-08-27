#!/usr/bin/env python3
"""R4-4D1 validation-only technical re-execution with canonical CSV row identities.

This driver does not expose a TRAIN scoring or threshold-calibration path. It imports
the frozen v1 acoustic/scoring/metric implementation, loads the immutable v1 TRAIN
threshold, normalizes only source_csv_row, and writes to locked_execution_v2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "ai-training/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import execute_r4_4d1_locked_hypothesis_validation as v1


CORRECTION_ROOT = REPO_ROOT / "ai-training/experiments/r4_4d1_identity_correction"
CORRECTION_CONTRACT = CORRECTION_ROOT / "r4_4d1_identity_correction_contract.json"
CORRECTION_CONTRACT_SHA256 = "C8FEDBAA279712B352F048A09A04F343B7BCBA41EA7C03694BD68D9E609B613B"
OUTPUT_ROOT = REPO_ROOT / "ai-training/experiments/r4_4d1_locked_hypothesis_validation"
LOCKED_OUTPUT = OUTPUT_ROOT / "locked_execution_v2"
R4D0_TRAIN_DISTRIBUTIONS = REPO_ROOT / "ai-training/experiments/r4_4d0_ctc_hypothesis_feasibility/train_score_distributions.json"


class IdentityCorrectionError(RuntimeError):
    pass


class FrozenThresholdError(RuntimeError):
    pass


class TrainRecalibrationForbidden(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_correction_contract(verify: bool = True) -> dict[str, Any]:
    if verify and sha256(CORRECTION_CONTRACT) != CORRECTION_CONTRACT_SHA256:
        raise IdentityCorrectionError("Identity-correction contract SHA mismatch")
    contract = read_json(CORRECTION_CONTRACT)
    if contract["contract_id"] != "R4-4D1C_MATCHED_CONTROL_ROW_IDENTITY_CORRECTION":
        raise IdentityCorrectionError("Unexpected identity-correction contract")
    return contract


def verify_sources(correction: dict[str, Any]) -> dict[str, Any]:
    actual: dict[str, str] = {}
    expected: dict[str, str] = {}
    for name, identity in correction["source_identities"].items():
        path = REPO_ROOT / identity["path"]
        expected[name] = identity["sha256"]
        actual[name] = sha256(path)
    if actual != expected:
        mismatches = sorted(name for name in expected if actual.get(name) != expected[name])
        raise IdentityCorrectionError(f"Frozen source SHA mismatch: {mismatches}")
    _original, numerical = v1.load_contracts(True)
    v1_verification = v1.verify_all_sources(numerical)
    return {"status": "PASS", "expected": expected, "actual": actual,
            "v1_source_verification": v1_verification,
            "correction_contract_sha256": sha256(CORRECTION_CONTRACT)}


def canonical_source_csv_row(source_index: int) -> int:
    return int(source_index) + 2


def reject_train_recalibration(requested: bool) -> None:
    if requested:
        raise TrainRecalibrationForbidden("R4-4D1 v2 has no TRAIN calibration path")


def load_frozen_threshold(correction: dict[str, Any], supplied_theta: float | None = None) -> tuple[float, dict[str, Any]]:
    identity = correction["source_identities"]["frozen_threshold"]
    path = REPO_ROOT / identity["path"]
    actual_sha = sha256(path)
    if actual_sha != identity["sha256"]:
        raise FrozenThresholdError("Frozen threshold SHA mismatch")
    payload = read_json(path)
    theta = float(payload["selected_threshold"])
    required = float(correction["threshold"]["theta"])
    if payload.get("score_family") != "RAW" or theta != required:
        raise FrozenThresholdError("Frozen threshold content mismatch")
    if supplied_theta is not None and float(supplied_theta) != required:
        raise FrozenThresholdError("Supplying a different theta is forbidden")
    return theta, {"path": str(path), "sha256": actual_sha, "theta": theta,
                   "score_family": "RAW", "content_verified": True,
                   "train_recalibrated": False}


def canonicalize_scored_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["source_csv_row"] = canonical_source_csv_row(int(row["source_csv_row"]))


def canonical_metadata_ids(words: Sequence[dict[str, Any]]) -> list[int]:
    return [canonical_source_csv_row(int(source["source_index"]))
            for word in words for source in word["clean_rows"]]


def identity_mapping_audit(required_ids: Sequence[int], generated_ids: Sequence[int], expected_count: int = 1434) -> dict[str, Any]:
    required = list(map(int, required_ids)); generated = list(map(int, generated_ids))
    required_set = set(required); generated_set = set(generated)
    missing = sorted(required_set - generated_set)
    duplicate_required = len(required) - len(required_set)
    duplicate_generated = len(generated) - len(generated_set)
    mapped = len(required) - len(missing)
    if (len(required), mapped, len(missing), duplicate_required, duplicate_generated) != (
        expected_count, expected_count, 0, 0, 0
    ):
        raise IdentityCorrectionError(
            f"Canonical matched identity failure: required={len(required)} mapped={mapped} "
            f"missing={len(missing)} required_duplicates={duplicate_required} "
            f"generated_duplicates={duplicate_generated}"
        )
    return {"required_rows": len(required), "mapped_rows": mapped, "missing_rows": 0,
            "duplicate_required_identities": 0, "duplicate_generated_identities": 0,
            "unexpected_ambiguities": 0, "canonical_rule": "int(source_index)+2"}


def frozen_metadata_mapping_audit(correction: dict[str, Any]) -> dict[str, Any]:
    matched_path = REPO_ROOT / correction["source_identities"]["matched_control"]["path"]
    prediction_path = REPO_ROOT / "ai-training/experiments/r4_4c2_bigru_ctc_seed42/validation_phone_predictions.csv"
    with matched_path.open(newline="", encoding="utf-8") as handle:
        matched = list(csv.DictReader(handle))
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        prior = list(csv.DictReader(handle))
    required = [int(row["source_csv_row"]) for row in matched]
    raw_source_indices = [int(row["source_csv_row"]) - 2 for row in prior]
    canonical = [canonical_source_csv_row(value) for value in raw_source_indices]
    old_missing = len(set(required) - set(raw_source_indices))
    audit = identity_mapping_audit(required, canonical, 1434)
    supports = {
        "deletion": sum(row["role"] == "deletion" for row in matched),
        "nondeletion": sum(row["role"] == "non_deletion" for row in matched),
        "phones": len({row["expected_phone"] for row in matched}),
        "speakers": len({row["speaker_id"] for row in matched}),
    }
    return {**audit, "old_mapping": {"mapped_rows": len(required) - old_missing,
                                      "missing_rows": old_missing},
            "canonical_mapping": {"mapped_rows": audit["mapped_rows"], "missing_rows": 0},
            "support": supports, "metadata_only": True, "acoustic_inference": False}


def substitution_phone_diagnostic(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    truth = [row for row in rows if row["true_relation"] == "substitution"]
    predicted = [row for row in truth if row["predicted_relation"] == "substitution"]
    correct = sum(row["best_sub_phone"] == row["true_observed_phone"] for row in predicted)
    return {"true_substitution_rows": len(truth), "predicted_substitution_rows": len(predicted),
            "coverage": v1.safe_div(len(predicted), len(truth)),
            "best_sub_phone_top1_accuracy": v1.safe_div(correct, len(predicted))}


def execute_validation_only() -> int:
    correction = load_correction_contract(True)
    verification = verify_sources(correction)
    reject_train_recalibration(False)
    _original, numerical = v1.load_contracts(True)
    validation_speakers = numerical["populations"]["validation"]["speakers"]

    # This guard runs before audio-root resolution, reconstruction, or path construction.
    v1.guard_test_speakers_before_resolution(validation_speakers)
    if LOCKED_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite locked execution: {LOCKED_OUTPUT}")

    theta, threshold_identity = load_frozen_threshold(correction)
    matched_path = REPO_ROOT / correction["source_identities"]["matched_control"]["path"]
    required_ids = v1.read_matched_ids(matched_path, numerical)

    import run_r4_4b_ctc_sequence as r4b
    audio_root = r4b.r3.require_audio_root()
    validation_words, validation_population = v1.load_population(
        "validation", validation_speakers, numerical, audio_root
    )
    mapping = identity_mapping_audit(required_ids, canonical_metadata_ids(validation_words), 1434)

    LOCKED_OUTPUT.mkdir(parents=True, exist_ok=False)
    v1.LOCKED_OUTPUT = LOCKED_OUTPUT
    write_json(LOCKED_OUTPUT / "execution_preflight.json", {
        "source_verification": verification, "threshold_identity": threshold_identity,
        "identity_mapping": mapping, "validation_only": True, "train_scoring": False,
        "train_recalibration": False, "validation_scoring_started": False,
        "training": False, "test_accessed": False, "preserved_v1": True,
    })
    write_json(LOCKED_OUTPUT / "threshold_identity.json", threshold_identity)
    write_json(LOCKED_OUTPUT / "identity_mapping_audit.json", mapping)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    validation_rows, validation_compute = v1.score_population(
        validation_words, device,
        REPO_ROOT / numerical["source_identities"]["r4_4c2_checkpoint"]["path"], "validation"
    )
    canonicalize_scored_rows(validation_rows)
    v1.apply_threshold(validation_rows, theta)
    v1.matched_control_guard(required_ids, [row["source_csv_row"] for row in validation_rows], 1434)
    diagnostics = v1.validation_diagnostics(validation_rows, required_ids, numerical)
    status = v1.classify_result(
        True, True, diagnostics["binary"], diagnostics["false_deletion"], diagnostics["matched"],
        diagnostics["three"], diagnostics["speakers"], diagnostics["continuous"], numerical
    )

    write_json(LOCKED_OUTPUT / "validation_binary_metrics.json", {**diagnostics["binary"], **diagnostics["false_deletion"]})
    three = {**diagnostics["three"], "substitution_phone_diagnostic": substitution_phone_diagnostic(validation_rows)}
    write_json(LOCKED_OUTPUT / "validation_3class_metrics.json", three)
    write_json(LOCKED_OUTPUT / "matched_control_metrics.json", diagnostics["matched"])
    write_json(LOCKED_OUTPUT / "continuous_auc_metrics.json", diagnostics["continuous"])
    write_json(LOCKED_OUTPUT / "speaker_metrics.json", diagnostics["speakers"])

    phone_output: dict[str, Any] = {}
    for phone in sorted({row["expected_phone"] for row in validation_rows}):
        phone_output[phone] = v1.grouped_deletion_metrics(
            [row for row in validation_rows if row["expected_phone"] == phone]
        )
    phone_output["groups"] = {
        "D_T_R_L": v1.grouped_deletion_metrics([row for row in validation_rows if row["expected_phone"] in {"D", "T", "R", "L"}]),
        "OTHER": v1.grouped_deletion_metrics([row for row in validation_rows if row["expected_phone"] not in {"D", "T", "R", "L"}]),
    }
    write_json(LOCKED_OUTPUT / "phone_metrics.json", phone_output)
    write_json(LOCKED_OUTPUT / "position_metrics.json", {
        name: v1.grouped_deletion_metrics([row for row in validation_rows if row["position_group"] == name])
        for name in ("initial", "medial", "final", "single")
    })
    write_json(LOCKED_OUTPUT / "multi_error_metrics.json", {
        "pure_deletion": v1.grouped_deletion_metrics([row for row in validation_rows if row["word_deletions"] > 0 and row["word_substitutions"] == 0]),
        "substitution_plus_deletion": v1.grouped_deletion_metrics([row for row in validation_rows if row["word_deletions"] > 0 and row["word_substitutions"] > 0]),
        "multiple_deletion": v1.grouped_deletion_metrics([row for row in validation_rows if row["word_deletions"] > 1]),
    })
    length_buckets = {"1": lambda n: n == 1, "2": lambda n: n == 2, "3": lambda n: n == 3,
                      "4": lambda n: n == 4, "5": lambda n: n == 5, "6+": lambda n: n >= 6}
    write_json(LOCKED_OUTPUT / "length_metrics.json", {
        name: v1.grouped_deletion_metrics([row for row in validation_rows if predicate(row["word_expected_length"])])
        for name, predicate in length_buckets.items()
    })
    duration_buckets = {"lt_150ms": lambda d: d < .150, "150_250ms": lambda d: .150 <= d < .250,
                        "250_400ms": lambda d: .250 <= d < .400, "400_600ms": lambda d: .400 <= d < .600,
                        "ge_600ms": lambda d: d >= .600}
    write_json(LOCKED_OUTPUT / "duration_metrics.json", {
        name: v1.grouped_deletion_metrics([row for row in validation_rows if predicate(row["word_duration"])])
        for name, predicate in duration_buckets.items()
    })

    greedy_rows = [row for row in validation_rows if row["greedy_relation"] == "deletion"]
    greedy_true = [row for row in greedy_rows if row["true_relation"] == "deletion"]
    greedy_false = [row for row in greedy_rows if row["true_relation"] != "deletion"]
    write_json(LOCKED_OUTPUT / "greedy_rescue_metrics.json", {
        "greedy_true_deletions": len(greedy_true),
        "true_deletion_retention_rate": v1.safe_div(sum(row["predicted_relation"] == "deletion" for row in greedy_true), len(greedy_true)),
        "greedy_false_deletions": len(greedy_false),
        "rescued_to_correct": sum(row["predicted_relation"] == "correct" for row in greedy_false),
        "rescued_to_substitution": sum(row["predicted_relation"] == "substitution" for row in greedy_false),
        "false_deletion_rescue_rate": v1.safe_div(sum(row["predicted_relation"] != "deletion" for row in greedy_false), len(greedy_false)),
    })
    score_distributions = {
        relation: v1.distribution(row["score"] for row in validation_rows if row["true_relation"] == relation)
        for relation in v1.RELATIONS
    }
    write_json(LOCKED_OUTPUT / "score_distributions.json", score_distributions)
    train_distributions = read_json(R4D0_TRAIN_DISTRIBUTIONS)["RAW"]
    train_medians = {relation: float(train_distributions[relation]["median"]) for relation in v1.RELATIONS}
    validation_medians = {relation: float(score_distributions[relation]["median"]) for relation in v1.RELATIONS}
    write_json(LOCKED_OUTPUT / "threshold_transfer.json", {
        "threshold": theta, "train_medians": train_medians, "validation_medians": validation_medians,
        "absolute_median_shift": {name: abs(validation_medians[name] - train_medians[name]) for name in v1.RELATIONS},
        "continuous": diagnostics["continuous"], "threshold_adjusted": False,
        "train_scores_recomputed": False,
    })
    v1.write_prediction_exports(validation_rows, validation_words)
    write_json(LOCKED_OUTPUT / "compute_report.json", {
        "device": str(device), "validation": validation_compute,
        "validation_seconds": time.perf_counter() - started,
        "validation_population": validation_population, "train_scoring": False,
        "train_recalibration": False, "locked_execution": "v2",
    })
    write_json(LOCKED_OUTPUT / "final_status.json", {
        "status": status, "threshold_sha256": threshold_identity["sha256"],
        "threshold": theta, "threshold_modified": False, "train_recalibrated": False,
        "validation_only": True, "training": False, "test_accessed": False,
    })
    (LOCKED_OUTPUT / "r4_4d1_locked_report.md").write_text(
        f"# R4-4D1 Locked Hypothesis Validation v2\n\nFinal: **{status}**\n\n"
        f"Threshold: `{theta}`\n\nThreshold SHA: `{threshold_identity['sha256']}`\n",
        encoding="utf-8",
    )
    hashes = {path.name: sha256(path) for path in sorted(LOCKED_OUTPUT.iterdir())
              if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(LOCKED_OUTPUT / "artifact_hashes.json", {"algorithm": "SHA-256", "files": hashes,
                                                         "note": "manifest excludes itself"})
    print(json.dumps({"status": status, "threshold": theta, "output": str(LOCKED_OUTPUT)}, indent=2))
    return 0


def run_synthetic_tests() -> dict[str, Any]:
    correction = load_correction_contract(True)
    tests: dict[str, bool] = {}
    tests["A_source_index_0_maps_to_2"] = canonical_source_csv_row(0) == 2
    tests["B_source_index_100_maps_to_102"] = canonical_source_csv_row(100) == 102
    mapping = frozen_metadata_mapping_audit(correction)
    tests["C_frozen_mapping_1434_of_1434"] = (
        mapping["canonical_mapping"] == {"mapped_rows": 1434, "missing_rows": 0}
        and mapping["duplicate_required_identities"] == 0
        and mapping["duplicate_generated_identities"] == 0
    )
    tests["D_old_mapping_reproduces_233_missing"] = mapping["old_mapping"] == {"mapped_rows": 1201, "missing_rows": 233}
    theta, identity = load_frozen_threshold(correction)
    tests["E_threshold_sha_and_theta"] = (
        theta == 2.197946548461914
        and identity["sha256"] == "36F6FD5AB6B7E98A607D499445E455DCAB8C3DD4ACDD19F252DC472FCDD07E94"
    )
    try:
        load_frozen_threshold(correction, supplied_theta=theta + 1.0)
        tests["F_different_theta_fatal"] = False
    except FrozenThresholdError:
        tests["F_different_theta_fatal"] = True
    try:
        reject_train_recalibration(True)
        tests["G_train_recalibration_fatal"] = False
    except TrainRecalibrationForbidden:
        tests["G_train_recalibration_fatal"] = True
    resolver_called = False
    def resolver() -> None:
        nonlocal resolver_called
        resolver_called = True
    try:
        v1.guard_test_speakers_before_resolution(["ASI"], resolver)
        tests["H_test_guard_before_resolution"] = False
    except v1.TestSpeakerGuardError:
        tests["H_test_guard_before_resolution"] = not resolver_called
    try:
        identity_mapping_audit(list(range(1434)), list(range(1433)), 1434)
        tests["I_missing_identity_fatal"] = False
    except IdentityCorrectionError:
        tests["I_missing_identity_fatal"] = True
    _original, numerical = v1.load_contracts(True)
    frozen_gates = numerical["validation_hard_gates"]
    correction_gates = correction["metrics_gates_and_classification"]["confirmation_gates"]
    expected_gates = {
        "binary_macro_f1_min": frozen_gates["binary_macro_f1_min"],
        "deletion_recall_min": frozen_gates["deletion_recall_min"],
        "deletion_f1_min": frozen_gates["deletion_f1_min"],
        "substitution_false_deletion_max": frozen_gates["substitution_false_deletion_max"],
        "matched_macro_f1_min": frozen_gates["matched_binary_macro_f1_min"],
        "matched_deletion_f1_min": frozen_gates["matched_deletion_f1_min"],
        "speaker_deletion_recall_min": frozen_gates["speaker_deletion_recall"]["minimum"],
        "speaker_gate_support_min": frozen_gates["speaker_deletion_recall"]["apply_when_support_at_least"],
        "three_relation_macro_f1_min": frozen_gates["three_relation_macro_f1_min"],
    }
    tests["J_metrics_gates_classification_match"] = (
        correction_gates == expected_gates
        and correction["metrics_gates_and_classification"]["strong_partial_and_threshold_transfer_fail"] == "unchanged"
        and correction["metrics_gates_and_classification"]["classification_precedence"] == "unchanged"
    )
    return {"tests": tests, "passed": sum(tests.values()), "total": len(tests),
            "all_passed": all(tests.values()), "mapping_audit": mapping,
            "validation_inference": False, "validation_hypothesis_scoring": False,
            "validation_metrics": False, "train_recalibration": False,
            "training": False, "test_accessed": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true", help="Run one future locked VALIDATION-only execution")
    mode.add_argument("--synthetic-tests", action="store_true", help="Run metadata/static tests only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.synthetic_tests:
        report = run_synthetic_tests()
        print(json.dumps(report, indent=2))
        return 0 if report["all_passed"] else 1
    return execute_validation_only()


if __name__ == "__main__":
    raise SystemExit(main())
