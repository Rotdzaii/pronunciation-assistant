#!/usr/bin/env python3
"""Single-use locked R4-4D2B TARGET-normalized VALIDATION runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "ai-training/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import execute_r4_4d1_locked_hypothesis_validation as v1


DESIGN_ROOT = REPO_ROOT / "ai-training/experiments/r4_4d2a_train_only_robust_calibration"
DESIGN_JSON = DESIGN_ROOT / "r4_4d2b_preregistered_validation_design.json"
DESIGN_MD = DESIGN_ROOT / "r4_4d2b_preregistered_validation_design.md"
D2A_MANIFEST = DESIGN_ROOT / "artifact_hashes.json"
OUTPUT = REPO_ROOT / "ai-training/experiments/r4_4d2b_final_target_validation"
TRAIN_DISTRIBUTIONS = REPO_ROOT / "ai-training/experiments/r4_4d0_ctc_hypothesis_feasibility/train_score_distributions.json"
THETA = np.float64(0.16184102947061696)
EXPECTED = {
    "design_json": "F0AC6874C1330DBFA2A8D99C88BE5167DCBED31122B40CE1F427FC0938DFA8AA",
    "design_md": "71BEB2A97331455863CCE203FD886F6831697D69F857B2AC1D5C35EAA5DB2D6B",
    "d2a_manifest": "F27A47A0E00398FF3C00CB4EB4F6997210D13D3B2B6BE0A72AB0F264A5D76B50",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "checkpoint": "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085",
    "matched": "D933F674743DA06CC8FAB425CEBF81D9C78505E1BDB4A90204DDB2E1A15B4798",
    "numerical": "5DC07A4B719FD6F38DBD1366CF802787FA3882CA972E5356A8F91DB435443425",
    "v1_driver": "5F12FB5E6B0A4765107DCAD3C822F32E6909940E9FAE265F5C3E3551ACB6AE22",
}


class LockedExecutionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, payload: Any) -> None:
    v1.write_json(OUTPUT / name, payload)


def verify_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    _original, numerical = v1.load_contracts(True)
    v1_sources = v1.verify_all_sources(numerical)
    paths = {
        "design_json": DESIGN_JSON,
        "design_md": DESIGN_MD,
        "d2a_manifest": D2A_MANIFEST,
        "v4": REPO_ROOT / numerical["source_identities"]["v4"]["path"],
        "checkpoint": REPO_ROOT / numerical["source_identities"]["r4_4c2_checkpoint"]["path"],
        "matched": REPO_ROOT / numerical["matched_control"]["path"],
        "numerical": v1.NUMERICAL_JSON,
        "v1_driver": Path(v1.__file__),
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    mismatches = {name: {"expected": EXPECTED[name], "actual": value}
                  for name, value in actual.items() if value != EXPECTED[name]}
    if mismatches:
        raise LockedExecutionError(f"R4_4D2B_SOURCE_VERIFICATION_FAIL: {mismatches}")
    design = read_json(DESIGN_JSON)
    checks = {
        "score_family_TARGET": bool(design.get("selected_score_family") == "TARGET"),
        "theta_exact": bool(np.float64(design.get("robust_theta")) == THETA),
        "one_validation": bool(design["policy"]["one_final_development_validation_run"] is True),
        "test_closed": bool(design["policy"]["test_closed"] is True),
        "training_false": bool(design["policy"]["training"] is False),
    }
    if not all(checks.values()):
        raise LockedExecutionError(f"Frozen design semantic mismatch: {checks}")
    return numerical, {"status": "PASS", "actual": actual, "expected": EXPECTED,
                       "design_checks": checks, "inherited_v1_sources": v1_sources}


def canonical_source_csv_row(source_index: int) -> int:
    return int(source_index) + 2


def mapping_audit(required: Sequence[int], generated: Sequence[int]) -> dict[str, Any]:
    required = [int(value) for value in required]
    generated = [int(value) for value in generated]
    required_set, generated_set = set(required), set(generated)
    missing = required_set - generated_set
    duplicates = len(generated) - len(generated_set)
    required_duplicates = len(required) - len(required_set)
    mapped = len(required) - len(missing)
    if (len(required), mapped, len(missing), duplicates, required_duplicates) != (1434, 1434, 0, 0, 0):
        raise LockedExecutionError("Frozen matched-control identity mapping failed")
    return {"canonical_rule": "source_csv_row=int(source_index)+2", "required_rows": 1434,
            "mapped_rows": 1434, "missing_rows": 0, "duplicate_collisions": 0,
            "required_duplicate_identities": 0, "unexpected_ambiguities": 0,
            "support": {"deletion": 717, "nondeletion": 717, "phones": 32,
                        "validation_speakers": 6}, "metadata_only": True}


def substitute_diagnostic(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    truth = [row for row in rows if row["true_relation"] == "substitution"]
    predicted = [row for row in truth if row["predicted_relation"] == "substitution"]
    correct = sum(row["best_sub_phone"] == row["true_observed_phone"] for row in predicted)
    return {"true_substitution_support": len(truth), "predicted_substitution_rows": len(predicted),
            "coverage": v1.safe_div(len(predicted), len(truth)),
            "best_sub_phone_top1_accuracy": v1.safe_div(correct, len(predicted))}


def convert_to_target(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["source_csv_row"] = canonical_source_csv_row(int(row["source_csv_row"]))
        row["score"] = np.float64(row["target_del_vs_best_nondelete"])
        row["keep_score"] = np.float64(row["target_keep"])
        row["delete_score"] = np.float64(row["target_delete"])
        row["best_sub_score"] = np.float64(row["target_best_sub"])
        row["score_family"] = "TARGET"


def confirmation_gates(diagnostics: dict[str, Any]) -> dict[str, Any]:
    binary, fd, matched, three, speakers = (diagnostics[key] for key in
        ("binary", "false_deletion", "matched", "three", "speakers"))
    speaker_detail = {name: {"support": metrics["deletion_support"],
                             "recall": metrics["deletion_recall"],
                             "passes": metrics["deletion_support"] < 30 or metrics["deletion_recall"] >= .25}
                      for name, metrics in speakers.items()}
    gates = {
        "A_binary_macro_f1_ge_0_70": binary["binary_macro_f1"] >= .70,
        "B_deletion_recall_ge_0_45": binary["deletion_recall"] >= .45,
        "C_deletion_f1_ge_0_40": binary["deletion_f1"] >= .40,
        "D_substitution_false_deletion_le_0_25": fd["substitution_false_deletion"] <= .25,
        "E_matched_macro_f1_ge_0_60": matched["binary_macro_f1"] >= .60,
        "F_matched_deletion_f1_ge_0_55": matched["deletion_f1"] >= .55,
        "G_all_supported_speakers_recall_ge_0_25": all(item["passes"] for item in speaker_detail.values()),
        "H_three_relation_macro_f1_ge_0_40": three["macro_f1"] >= .40,
    }
    return {"gates": gates, "speaker_gate_detail": speaker_detail,
            "passed": sum(gates.values()), "total": 8, "all_passed": all(gates.values())}


def execute() -> int:
    recovery_marker = OUTPUT / "pre_execution_technical_stop.json"
    if OUTPUT.exists() and set(path.name for path in OUTPUT.iterdir()) != {recovery_marker.name}:
        raise LockedExecutionError(f"Refusing overwrite/rerun: {OUTPUT}")
    numerical, verification = verify_sources()
    validation_speakers = numerical["populations"]["validation"]["speakers"]
    # This guard occurs before audio-root resolution or any dataset path construction.
    v1.guard_test_speakers_before_resolution(validation_speakers)

    required_ids = v1.read_matched_ids(REPO_ROOT / numerical["matched_control"]["path"], numerical)
    import run_r4_4b_ctc_sequence as r4b
    audio_root = r4b.r3.require_audio_root()
    validation_words, population = v1.load_population("validation", validation_speakers, numerical, audio_root)
    expected_population = {"words": 7728, "relations": {"correct": 22759, "substitution": 2664, "deletion": 914}}
    if population["words"] != expected_population["words"] or population["relations"] != expected_population["relations"]:
        raise LockedExecutionError(f"Validation population mismatch: {population}")
    generated_ids = [canonical_source_csv_row(int(source["source_index"]))
                     for word in validation_words for source in word["clean_rows"]]
    mapping = mapping_audit(required_ids, generated_ids)

    OUTPUT.mkdir(parents=True, exist_ok=OUTPUT.exists())
    v1.LOCKED_OUTPUT = OUTPUT
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    command = f'"{sys.executable}" "{Path(__file__).resolve()}" --execute'
    driver_sha = sha256(Path(__file__))
    write_json("execution_preflight.json", {
        "status": "PASS", "source_verification": verification, "score_family": "TARGET",
        "theta": float(THETA), "theta_source": "TRAIN-only 12-fold speaker-LOSO median",
        "threshold_recalculated": False, "checkpoint_sha256": EXPECTED["checkpoint"],
        "contract_shas": {"json": EXPECTED["design_json"], "markdown": EXPECTED["design_md"],
                          "numerical": EXPECTED["numerical"]},
        "matched_control_sha256": EXPECTED["matched"], "test_guard_active": True,
        "test_closed": True, "execution_count": 1, "validation_execution_started": False,
        "training": False, "command": command, "start_time": started_wall,
        "preflight_process_attempts": 2, "pre_execution_stop_preserved": recovery_marker.exists(),
        "driver_sha256": driver_sha,
    })
    write_json("threshold_identity.json", {"score_family": "TARGET", "theta": float(THETA),
        "source": str(DESIGN_JSON), "source_sha256": EXPECTED["design_json"],
        "ordinary_float64_median_of_12_train_loso_thresholds": True,
        "recalculated": False, "modified": False})
    write_json("score_contract_identity.json", {"family": "TARGET",
        "raw": "-CTCLoss(log_softmax(logits),H,blank=40,reduction=none,zero_infinity=true)",
        "target": "RAW_SCORE(H)/max(len(H),1)",
        "decision_score": "DELETE_i-max(KEEP_i,BEST_SUB_i)", "substitution_alternatives": 39,
        "best_sub_tie": "lowest canonical phone index", "dtype_for_decision": "float64",
        "threshold": float(THETA), "threshold_equality": "deletion",
        "keep_sub_equality": "correct", "alternative_families_evaluated": False})
    write_json("matched_mapping_verification.json", mapping)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    rows, compute = v1.score_population(validation_words, device,
        REPO_ROOT / numerical["source_identities"]["r4_4c2_checkpoint"]["path"], "validation")
    convert_to_target(rows)
    v1.apply_threshold(rows, float(THETA))
    v1.matched_control_guard(required_ids, [row["source_csv_row"] for row in rows], 1434)
    diagnostics = v1.validation_diagnostics(rows, required_ids, numerical)
    gate_result = confirmation_gates(diagnostics)
    status = ("R4_4D2B_DELETION_VALIDATION_CONFIRMED" if gate_result["all_passed"]
              else "R4_4D2B_DELETION_VALIDATION_NOT_CONFIRMED")

    write_json("validation_binary_metrics.json", {**diagnostics["binary"],
        **diagnostics["false_deletion"], "confirmation_gates": gate_result})
    sub_diag = substitute_diagnostic(rows)
    write_json("validation_3class_metrics.json", {**diagnostics["three"],
        "substitution_phone_diagnostic": sub_diag})
    write_json("matched_control_metrics.json", diagnostics["matched"])
    write_json("continuous_auc_metrics.json", diagnostics["continuous"])
    distributions = {relation: v1.distribution(row["score"] for row in rows if row["true_relation"] == relation)
                     for relation in v1.RELATIONS}
    write_json("score_distributions.json", distributions)
    train_distributions = read_json(TRAIN_DISTRIBUTIONS)["TARGET"]
    train_medians = {name: float(train_distributions[name]["median"]) for name in v1.RELATIONS}
    validation_medians = {name: float(distributions[name]["median"]) for name in v1.RELATIONS}
    shifts = {name: abs(validation_medians[name] - train_medians[name]) for name in v1.RELATIONS}
    write_json("threshold_transfer.json", {"theta": float(THETA), "train_medians": train_medians,
        "validation_medians": validation_medians, "absolute_median_shifts": shifts,
        "assessment": "descriptive_shift_only", "threshold_recalculated": False,
        "threshold_modified": False, "continuous_target_signal": diagnostics["continuous"]})
    write_json("speaker_metrics.json", diagnostics["speakers"])

    phone = {name: v1.grouped_deletion_metrics([row for row in rows if row["expected_phone"] == name])
             for name in sorted({row["expected_phone"] for row in rows})}
    phone["groups"] = {
        "D_T_R_L": v1.grouped_deletion_metrics([row for row in rows if row["expected_phone"] in {"D", "T", "R", "L"}]),
        "OTHER": v1.grouped_deletion_metrics([row for row in rows if row["expected_phone"] not in {"D", "T", "R", "L"}]),
    }
    write_json("phone_metrics.json", phone)
    write_json("position_metrics.json", {
        label: v1.grouped_deletion_metrics([row for row in rows if row["position_group"] == source])
        for label, source in (("initial", "initial"), ("medial", "medial"),
                              ("final", "final"), ("single_phone", "single"))})
    write_json("multi_error_metrics.json", {
        "pure_deletion": v1.grouped_deletion_metrics([row for row in rows if row["word_deletions"] > 0 and row["word_substitutions"] == 0]),
        "substitution_plus_deletion": v1.grouped_deletion_metrics([row for row in rows if row["word_deletions"] > 0 and row["word_substitutions"] > 0]),
        "multiple_deletion": v1.grouped_deletion_metrics([row for row in rows if row["word_deletions"] > 1]),
    })
    length_buckets = {"1": lambda n: n == 1, "2": lambda n: n == 2, "3": lambda n: n == 3,
                      "4": lambda n: n == 4, "5": lambda n: n == 5, "6+": lambda n: n >= 6}
    write_json("length_metrics.json", {name: v1.grouped_deletion_metrics(
        [row for row in rows if predicate(row["word_expected_length"])]) for name, predicate in length_buckets.items()})
    duration_buckets = {"lt_150ms": lambda d: d < .150, "150_250ms": lambda d: .150 <= d < .250,
                        "250_400ms": lambda d: .250 <= d < .400, "400_600ms": lambda d: .400 <= d < .600,
                        "ge_600ms": lambda d: d >= .600}
    write_json("duration_metrics.json", {name: v1.grouped_deletion_metrics(
        [row for row in rows if predicate(row["word_duration"])]) for name, predicate in duration_buckets.items()})

    d1 = {"binary_macro_f1": .643750, "balanced_accuracy": .696587,
          "deletion_precision": .253149, "deletion_recall": .439825, "deletion_f1": .321343,
          "matched_macro_f1": .613344, "matched_deletion_f1": .533802,
          "three_relation_macro_f1": .462263}
    oof = {"binary_macro_f1": .649786, "deletion_precision": .253866,
           "deletion_recall": .457254, "deletion_f1": .326474,
           "correct_false_deletion": .037490, "substitution_false_deletion": .041248,
           "three_relation_macro_f1": .497274}
    actual = {"binary_macro_f1": diagnostics["binary"]["binary_macro_f1"],
              "balanced_accuracy": diagnostics["binary"]["balanced_accuracy"],
              "deletion_precision": diagnostics["binary"]["deletion_precision"],
              "deletion_recall": diagnostics["binary"]["deletion_recall"],
              "deletion_f1": diagnostics["binary"]["deletion_f1"],
              "matched_macro_f1": diagnostics["matched"]["binary_macro_f1"],
              "matched_deletion_f1": diagnostics["matched"]["deletion_f1"],
              "three_relation_macro_f1": diagnostics["three"]["macro_f1"]}
    write_json("comparison_metrics.json", {"r4_4d1_raw": d1, "target_train_oof": oof,
        "r4_4d2b_target": actual,
        "delta_vs_r4_4d1_raw": {key: actual[key] - value for key, value in d1.items()},
        "target_oof_vs_validation": {key: actual[key] - value for key, value in oof.items() if key in actual},
        "historical": {
            "duration_only": {"binary_macro_f1": .668146, "deletion_f1": .364164},
            "r4_1": {"binary_macro_f1": .657336, "deletion_f1": .341612},
            "r4_2a": {"binary_macro_f1": .566997, "deletion_f1": .197525},
            "r4_3b": {"binary_macro_f1": .503101, "deletion_f1": .025465},
            "r4_4c2_greedy": {"binary_macro_f1": .555712, "deletion_f1": .185464},
            "r4_4d1_raw": {"binary_macro_f1": .643750, "deletion_f1": .321343},
            "r4_4d2b_target": {"binary_macro_f1": actual["binary_macro_f1"], "deletion_f1": actual["deletion_f1"]},
        }})
    v1.write_prediction_exports(rows, validation_words)
    elapsed = time.perf_counter() - started
    ended_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json("compute_report.json", {"device": str(device), "python": platform.python_version(),
        "pytorch": torch.__version__, "cuda": torch.version.cuda, "validation": compute,
        "validation_seconds": elapsed, "average_seconds_per_word": elapsed / 7728,
        "start_time": started_wall, "end_time": ended_wall, "execution_count": 1,
        "training": False, "threshold_calibration": False, "test_accessed": False})
    write_json("final_status.json", {"status": status, "confirmation_gates": gate_result,
        "score_family": "TARGET", "theta": float(THETA), "threshold_changed": False,
        "threshold_recalculated": False, "validation_execution_count": 1,
        "neural_training": False, "r4_test_accessed": False,
        "scientific_status": "iterative development VALIDATION; not an untouched final holdout",
        "stop_current_r4_deletion_research": status.endswith("NOT_CONFIRMED")})
    report = (
        "# R4-4D2B Final Locked TARGET Calibration Validation\n\n"
        f"Final: **{status}**\n\n"
        "This was an iterative development VALIDATION; prior R4 validation results were already observed. "
        "The six-speaker R4 TEST remains the untouched independent holdout.\n\n"
        f"Score family: `TARGET`; theta: `{float(THETA)}`. Threshold recalculated: **NO**.\n\n"
        f"Binary Macro-F1: `{actual['binary_macro_f1']:.6f}`; deletion F1: `{actual['deletion_f1']:.6f}`.\n\n"
        f"Frozen gates passed: `{gate_result['passed']}/8`.\n\n"
        f"R4 TEST accessed: **NO**. Neural training: **NO**.\n"
    )
    if status.endswith("NOT_CONFIRMED"):
        report += "\nThe frozen decision criteria were not all met. Current R4 deletion research stops here; deletion detection remains a Phoenix research limitation.\n"
    (OUTPUT / "r4_4d2b_report.md").write_text(report, encoding="utf-8")
    hashes = {path.name: sha256(path) for path in sorted(OUTPUT.iterdir())
              if path.is_file() and path.name != "artifact_hashes.json"}
    write_json("artifact_hashes.json", {"algorithm": "SHA-256", "files": hashes,
        "manifest_excludes_itself": True, "driver_sha256": driver_sha})
    print(json.dumps({"status": status, "output": str(OUTPUT), "theta": float(THETA),
                      "gates_passed": gate_result["passed"]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    if not args.execute:
        raise LockedExecutionError("Only the locked --execute mode exists")
    return execute()


if __name__ == "__main__":
    raise SystemExit(main())
