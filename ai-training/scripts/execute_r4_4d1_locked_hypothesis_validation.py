"""Frozen R4-4D1 TRAIN-calibrated CTC hypothesis validation driver.

`--self-test` is synthetic/static only. `--execute` is the future locked run:
TRAIN scoring -> threshold artifact write/hash/re-open verification -> exactly
one VALIDATION scoring pass.  The execution mode is intentionally not invoked
by the R4-4D1B driver-freeze task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_PREREG = REPO_ROOT / "ai-training/experiments/r4_4d0_ctc_hypothesis_feasibility/r4_4d1_preregistered_score_design.json"
NUMERICAL_DIR = REPO_ROOT / "ai-training/experiments/r4_4d1_numerical_contract"
NUMERICAL_JSON = NUMERICAL_DIR / "r4_4d1_complete_evaluation_contract.json"
NUMERICAL_MD = NUMERICAL_DIR / "r4_4d1_complete_evaluation_contract.md"
OUTPUT_ROOT = REPO_ROOT / "ai-training/experiments/r4_4d1_locked_hypothesis_validation"
LOCKED_OUTPUT = OUTPUT_ROOT / "locked_execution_v1"
EXPECTED_TRUST_ANCHORS = {
    "original_preregistration": "7FD870DF2321465D716EAFB1B66E7D6116C153E3D8BCD494494F3FAE44ACC784",
    "numerical_json": "5DC07A4B719FD6F38DBD1366CF802787FA3882CA972E5356A8F91DB435443425",
    "numerical_markdown": "F00AE59A7EC1B44746BE39439C28C3EAFF5294C57C31D3088C4610062BF913C4",
}
RELATIONS = ("correct", "substitution", "deletion")
RELATION_INDEX = {name: index for index, name in enumerate(RELATIONS)}
TEST_SPEAKERS = frozenset(("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK"))


class SourceVerificationError(RuntimeError):
    pass


class TestSpeakerGuardError(RuntimeError):
    pass


class MatchedControlError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_contracts(verify_hashes: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    if verify_hashes:
        actual = {
            "original_preregistration": sha256(ORIGINAL_PREREG),
            "numerical_json": sha256(NUMERICAL_JSON),
            "numerical_markdown": sha256(NUMERICAL_MD),
        }
        mismatch = {key: {"expected": EXPECTED_TRUST_ANCHORS[key], "actual": value}
                    for key, value in actual.items() if value != EXPECTED_TRUST_ANCHORS[key]}
        if mismatch:
            raise SourceVerificationError(f"R4_4D1_SOURCE_VERIFICATION_FAIL: {mismatch}")
    original = json.loads(ORIGINAL_PREREG.read_text(encoding="utf-8"))
    contract = json.loads(NUMERICAL_JSON.read_text(encoding="utf-8"))
    validate_contract_schema(original, contract)
    return original, contract


def validate_contract_schema(original: dict[str, Any], contract: dict[str, Any]) -> None:
    checks = {
        "original_raw": original.get("score_family") == "RAW",
        "additive_original_sha": contract["additive_contract_policy"]["original_preregistration_sha256"] == EXPECTED_TRUST_ANCHORS["original_preregistration"],
        "contract_raw": contract["ctc_score"]["selected_family"] == "RAW",
        "blank_40": contract["ctc_score"]["parameters"]["blank"] == 40,
        "reduction_none": contract["ctc_score"]["parameters"]["reduction"] == "none",
        "zero_infinity": contract["ctc_score"]["parameters"]["zero_infinity"] is True,
        "float64": contract["decision_score"]["dtype"] == "float64",
        "threshold_float64": contract["train_threshold_calibration"]["dtype"] == "float64",
        "relation_order": tuple(contract["metrics"]["three_relation"]["class_order"]) == RELATIONS,
        "matched_1434": contract["matched_control"]["total_rows"] == 1434,
        "test_exclusion": set(contract["populations"]["test"]["speakers"]) == TEST_SPEAKERS,
        "train_speakers": len(contract["populations"]["train"]["speakers"]) == 12,
        "validation_speakers": len(contract["populations"]["validation"]["speakers"]) == 6,
        "classification_precedence": len(contract["result_classification"]["precedence"]) == 6,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SourceVerificationError(f"Contract schema mismatch: {failed}")


def verify_all_sources(contract: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "original_preregistration": ORIGINAL_PREREG,
        "numerical_json": NUMERICAL_JSON,
        "numerical_markdown": NUMERICAL_MD,
        "v4": REPO_ROOT / contract["source_identities"]["v4"]["path"],
        "r4_4c2_checkpoint": REPO_ROOT / contract["source_identities"]["r4_4c2_checkpoint"]["path"],
        "r4_4c2_manifest": REPO_ROOT / contract["source_identities"]["r4_4c2_manifest"]["path"],
        "r4_4d0_manifest": REPO_ROOT / contract["source_identities"]["r4_4d0_manifest"]["path"],
        "r4_4d0_scoring_implementation": REPO_ROOT / contract["source_identities"]["r4_4d0_scoring_implementation"]["path"],
        "matched_control": REPO_ROOT / contract["matched_control"]["path"],
    }
    expected = {
        **EXPECTED_TRUST_ANCHORS,
        "v4": contract["source_identities"]["v4"]["sha256"],
        "r4_4c2_checkpoint": contract["source_identities"]["r4_4c2_checkpoint"]["sha256"],
        "r4_4c2_manifest": contract["source_identities"]["r4_4c2_manifest"]["sha256"],
        "r4_4d0_manifest": contract["source_identities"]["r4_4d0_manifest"]["sha256"],
        "r4_4d0_scoring_implementation": contract["source_identities"]["r4_4d0_scoring_implementation"]["sha256"],
        "matched_control": contract["matched_control"]["sha256"],
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    mismatch = {name: {"expected": expected[name], "actual": value}
                for name, value in actual.items() if value != expected[name]}
    if mismatch:
        raise SourceVerificationError(f"R4_4D1_SOURCE_VERIFICATION_FAIL: {mismatch}")
    return {"status": "PASS", "expected": expected, "actual": actual}


def guard_test_speakers_before_resolution(
    speakers: Iterable[str], resolver: Callable[[], Any] | None = None
) -> Any:
    requested = frozenset(speakers)
    leaked = sorted(requested & TEST_SPEAKERS)
    if leaked:
        raise TestSpeakerGuardError(f"TEST speakers rejected before path resolution: {leaked}")
    return resolver() if resolver is not None else None


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def f1_score_value(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def binary_metrics(true_relations: Sequence[str], predicted_relations: Sequence[str]) -> dict[str, Any]:
    if len(true_relations) != len(predicted_relations):
        raise ValueError("Truth/prediction length mismatch")
    tn = fp = fn = tp = 0
    for truth, prediction in zip(true_relations, predicted_relations):
        true_delete = truth == "deletion"; pred_delete = prediction == "deletion"
        if true_delete and pred_delete: tp += 1
        elif true_delete: fn += 1
        elif pred_delete: fp += 1
        else: tn += 1
    deletion_precision = safe_div(tp, tp + fp); deletion_recall = safe_div(tp, tp + fn)
    nondeletion_precision = safe_div(tn, tn + fn); nondeletion_recall = safe_div(tn, tn + fp)
    deletion_f1 = f1_score_value(deletion_precision, deletion_recall)
    nondeletion_f1 = f1_score_value(nondeletion_precision, nondeletion_recall)
    return {
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": (nondeletion_recall + deletion_recall) / 2.0,
        "binary_macro_f1": (nondeletion_f1 + deletion_f1) / 2.0,
        "deletion_precision": deletion_precision, "deletion_recall": deletion_recall,
        "deletion_f1": deletion_f1, "nondeletion_precision": nondeletion_precision,
        "nondeletion_recall": nondeletion_recall, "nondeletion_f1": nondeletion_f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def three_class_metrics(true_relations: Sequence[str], predicted_relations: Sequence[str]) -> dict[str, Any]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for truth, prediction in zip(true_relations, predicted_relations):
        matrix[RELATION_INDEX[truth], RELATION_INDEX[prediction]] += 1
    per_class: dict[str, Any] = {}; f1_values = []
    for index, name in enumerate(RELATIONS):
        tp = int(matrix[index, index]); fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp); support = int(matrix[index, :].sum())
        precision = safe_div(tp, tp + fp); recall = safe_div(tp, tp + fn)
        f1 = f1_score_value(precision, recall); f1_values.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {"class_order": list(RELATIONS), "macro_f1": float(np.mean(f1_values)),
            "per_class": per_class, "confusion_matrix": matrix.tolist()}


def false_deletion_metrics(true_relations: Sequence[str], predicted_relations: Sequence[str]) -> dict[str, float]:
    output = {}
    for relation in ("correct", "substitution"):
        indexes = [index for index, value in enumerate(true_relations) if value == relation]
        output[f"{relation}_false_deletion"] = safe_div(
            sum(predicted_relations[index] == "deletion" for index in indexes), len(indexes)
        )
    return output


def raw_ctc_score(log_probs: torch.Tensor, target: Sequence[int], blank: int = 40) -> float:
    if log_probs.ndim != 2 or log_probs.shape[1] != 41:
        raise ValueError("Expected [T,41] log probabilities")
    criterion = torch.nn.CTCLoss(blank=blank, reduction="none", zero_infinity=True)
    values = torch.tensor(list(target), dtype=torch.long, device=log_probs.device)
    nll = criterion(
        log_probs.unsqueeze(1), values,
        torch.tensor([log_probs.shape[0]], dtype=torch.long, device=log_probs.device),
        torch.tensor([len(target)], dtype=torch.long, device=log_probs.device),
    )[0]
    score = -float(nll.item())
    if not math.isfinite(score):
        raise RuntimeError("Non-finite RAW CTC score")
    return score


def best_substitution(candidate_scores: dict[int, float]) -> tuple[int, float]:
    if not candidate_scores:
        raise ValueError("No substitution candidates")
    best_score = max(candidate_scores.values())
    phone = min(index for index, score in candidate_scores.items() if score == best_score)
    return phone, float(best_score)


def decision_score(delete_score: float, keep_score: float, best_sub_score: float) -> np.float64:
    return np.float64(delete_score) - max(np.float64(keep_score), np.float64(best_sub_score))


def relation_decision(
    score: float, threshold: float, keep_score: float, best_sub_score: float, best_sub_phone: int
) -> tuple[str, int | None]:
    if np.float64(score) >= np.float64(threshold):
        return "deletion", None
    if np.float64(keep_score) >= np.float64(best_sub_score):
        return "correct", None
    return "substitution", int(best_sub_phone)


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("TRAIN D_i scores must be non-empty and finite")
    unique = np.unique(values)
    return np.concatenate((
        np.asarray([np.nextafter(unique[0], -np.inf)], dtype=np.float64),
        unique,
        np.asarray([np.nextafter(unique[-1], np.inf)], dtype=np.float64),
    ))


def _prediction_base(rows: Sequence[dict[str, Any]]) -> list[str]:
    return ["correct" if np.float64(row["keep_score"]) >= np.float64(row["best_sub_score"])
            else "substitution" for row in rows]


def threshold_eligibility(
    true_relations: Sequence[str], predicted_relations: Sequence[str], speakers: Sequence[str],
    contract: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    binary = binary_metrics(true_relations, predicted_relations)
    false_deletion = false_deletion_metrics(true_relations, predicted_relations)
    rule = contract["train_threshold_calibration"]["eligibility"]
    speaker_results = {}; speaker_pass = True
    for speaker in sorted(set(speakers)):
        indexes = [index for index, value in enumerate(speakers) if value == speaker]
        support = sum(true_relations[index] == "deletion" for index in indexes)
        recall = safe_div(sum(true_relations[index] == "deletion" and predicted_relations[index] == "deletion"
                              for index in indexes), support)
        eligible = support >= rule["speaker_rule"]["apply_when_true_deletion_support_at_least"]
        passed = not eligible or recall >= rule["speaker_rule"]["deletion_recall_min"]
        speaker_pass &= passed
        speaker_results[speaker] = {"deletion_support": support, "deletion_recall": recall,
                                    "eligible": eligible, "pass": passed}
    passed = (
        binary["deletion_recall"] >= rule["deletion_recall_min"]
        and false_deletion["substitution_false_deletion"] <= rule["substitution_false_deletion_max"]
        and speaker_pass
    )
    return passed, {"binary": binary, "false_deletion": false_deletion,
                    "speaker": speaker_results, "speaker_gate": speaker_pass}


def select_train_threshold(rows: Sequence[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    candidates = threshold_candidates(scores)
    truth = [row["true_relation"] for row in rows]
    speakers = [row["speaker"] for row in rows]
    base = _prediction_base(rows)
    order = np.argsort(scores, kind="stable")[::-1]
    grouped: dict[float, list[int]] = defaultdict(list)
    for index in order.tolist():
        grouped[float(scores[index])].append(index)
    predicted = list(base)
    candidate_metrics: dict[float, dict[str, Any]] = {}

    def evaluate(threshold: float) -> None:
        eligible, eligibility = threshold_eligibility(truth, predicted, speakers, contract)
        three = three_class_metrics(truth, predicted)
        candidate_metrics[float(threshold)] = {
            "threshold": float(threshold), "eligible": eligible,
            "binary": eligibility["binary"], "false_deletion": eligibility["false_deletion"],
            "speaker": eligibility["speaker"], "speaker_gate": eligibility["speaker_gate"],
            "three_relation_macro_f1": three["macro_f1"],
        }

    evaluate(float(candidates[-1]))  # nextafter(max,+inf): no deletion
    for threshold in sorted(grouped, reverse=True):
        for index in grouped[threshold]:
            predicted[index] = "deletion"
        evaluate(threshold)
    evaluate(float(candidates[0]))  # nextafter(min,-inf): all deletion

    ordered_metrics = [candidate_metrics[float(value)] for value in candidates]
    eligible_metrics = [item for item in ordered_metrics if item["eligible"]]
    if not eligible_metrics:
        return {"status": "R4_4D1_TRAIN_CALIBRATION_NO_ELIGIBLE_THRESHOLD",
                "candidate_count": len(candidates), "eligible_count": 0,
                "candidate_metrics": ordered_metrics, "selected": None}
    selected = max(eligible_metrics, key=lambda item: (
        item["binary"]["binary_macro_f1"], item["binary"]["deletion_f1"],
        item["binary"]["deletion_precision"], item["three_relation_macro_f1"], item["threshold"],
    ))
    return {"status": "PASS", "candidate_count": len(candidates),
            "eligible_count": len(eligible_metrics), "candidate_metrics": ordered_metrics,
            "selected": selected}


def matched_control_guard(
    required_ids: Sequence[int], prediction_ids: Sequence[int], expected_count: int = 1434
) -> list[int]:
    if len(required_ids) != expected_count or len(set(required_ids)) != expected_count:
        raise MatchedControlError("Frozen matched identities are not exact/unique")
    counts = Counter(prediction_ids)
    missing = [identity for identity in required_ids if counts[identity] != 1]
    if missing:
        raise MatchedControlError(f"Matched identity mapping failure: {len(missing)} rows")
    return list(required_ids)


def auc_metrics(true_relations: Sequence[str], scores: Sequence[float], negative_relation: str | None = None) -> dict[str, Any]:
    indexes = [index for index, relation in enumerate(true_relations)
               if negative_relation is None or relation in {"deletion", negative_relation}]
    labels = np.asarray([true_relations[index] == "deletion" for index in indexes], dtype=np.int8)
    values = np.asarray([scores[index] for index in indexes], dtype=np.float64)
    if len(np.unique(labels)) != 2:
        return {"roc_auc": 0.0, "pr_auc": 0.0, "count": len(indexes)}
    return {"roc_auc": float(roc_auc_score(labels, values)),
            "pr_auc": float(average_precision_score(labels, values)), "count": len(indexes)}


def speaker_gate(speaker_metrics: dict[str, dict[str, Any]], minimum: float, support: int = 30) -> bool:
    return all(item["deletion_support"] < support or item["deletion_recall"] >= minimum
               for item in speaker_metrics.values())


def classify_result(
    source_verified: bool, calibration_available: bool, binary: dict[str, Any] | None,
    false_deletion: dict[str, float] | None, matched: dict[str, Any] | None,
    three: dict[str, Any] | None, speakers: dict[str, dict[str, Any]] | None,
    continuous: dict[str, dict[str, float]] | None, contract: dict[str, Any],
) -> str:
    if not source_verified:
        return "R4_4D1_SOURCE_VERIFICATION_FAIL"
    if not calibration_available:
        return "R4_4D1_TRAIN_CALIBRATION_NO_ELIGIBLE_THRESHOLD"
    assert binary is not None and false_deletion is not None and matched is not None
    assert three is not None and speakers is not None and continuous is not None
    gates = contract["validation_hard_gates"]
    confirmed = (
        binary["binary_macro_f1"] >= gates["binary_macro_f1_min"]
        and binary["deletion_recall"] >= gates["deletion_recall_min"]
        and binary["deletion_f1"] >= gates["deletion_f1_min"]
        and false_deletion["substitution_false_deletion"] <= gates["substitution_false_deletion_max"]
        and matched["binary_macro_f1"] >= gates["matched_binary_macro_f1_min"]
        and matched["deletion_f1"] >= gates["matched_deletion_f1_min"]
        and speaker_gate(speakers, gates["speaker_deletion_recall"]["minimum"],
                         gates["speaker_deletion_recall"]["apply_when_support_at_least"])
        and three["macro_f1"] >= gates["three_relation_macro_f1_min"]
    )
    if confirmed:
        return "R4_4D1_HYPOTHESIS_DELETION_CONFIRMED"
    partial = contract["result_classification"]["strong_partial"]
    strong_partial = (
        binary["binary_macro_f1"] >= partial["binary_macro_f1_min"]
        and binary["deletion_f1"] >= partial["deletion_f1_min"]
        and binary["deletion_recall"] >= partial["deletion_recall_min"]
        and false_deletion["substitution_false_deletion"] <= partial["substitution_false_deletion_max"]
        and matched["binary_macro_f1"] >= partial["matched_binary_macro_f1_min"]
        and matched["deletion_f1"] >= partial["matched_deletion_f1_min"]
        and three["macro_f1"] >= partial["three_relation_macro_f1_min"]
        and speaker_gate(speakers, partial["speaker_deletion_recall"]["minimum"],
                         partial["speaker_deletion_recall"]["apply_when_support_at_least"])
    )
    if strong_partial:
        return "R4_4D1_HYPOTHESIS_DELETION_STRONG_PARTIAL"
    transfer = contract["result_classification"]["threshold_transfer_fail"]
    if (continuous["deletion_vs_nondeletion"]["roc_auc"] >= transfer["validation_deletion_vs_nondeletion_roc_auc_min"]
            and continuous["deletion_vs_substitution"]["roc_auc"] >= transfer["validation_deletion_vs_substitution_roc_auc_min"]):
        return "R4_4D1_HYPOTHESIS_THRESHOLD_TRANSFER_FAIL"
    return "R4_4D1_HYPOTHESIS_SIGNAL_NOT_CONFIRMED"


def freeze_threshold_artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen threshold: {path}")
    write_json(path, payload)  # closes file before hashing
    digest = sha256(path)
    reopened = json.loads(path.read_text(encoding="utf-8"))
    if reopened != payload or sha256(path) != digest:
        raise RuntimeError("Threshold artifact re-open/content/hash verification failed")
    return {"path": str(path), "sha256": digest, "content_verified": True, "closed_before_validation": True}


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {"count": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)), "p25": float(np.percentile(array, 25)),
            "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90))}


def apply_threshold(rows: list[dict[str, Any]], threshold: float) -> None:
    for row in rows:
        prediction, phone_index = relation_decision(
            row["score"], threshold, row["keep_score"], row["best_sub_score"], row["best_sub_phone_index"]
        )
        row["predicted_relation"] = prediction
        row["predicted_observed_phone"] = (
            row["phone_vocabulary"][phone_index] if phone_index is not None else ""
        )


def row_metrics(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    truth = [row["true_relation"] for row in rows]; predicted = [row["predicted_relation"] for row in rows]
    return binary_metrics(truth, predicted), false_deletion_metrics(truth, predicted), three_class_metrics(truth, predicted)


def grouped_deletion_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    binary, _, _ = row_metrics(rows)
    return {"rows": len(rows), "deletion_support": sum(row["true_relation"] == "deletion" for row in rows),
            "deletion_precision": binary["deletion_precision"], "deletion_recall": binary["deletion_recall"],
            "deletion_f1": binary["deletion_f1"],
            "false_deletion_rate": safe_div(sum(row["true_relation"] != "deletion" and row["predicted_relation"] == "deletion" for row in rows),
                                             sum(row["true_relation"] != "deletion" for row in rows))}


def load_population(split: str, speakers: Sequence[str], contract: dict[str, Any], audio_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Lazy imports happen only after guard_test_speakers_before_resolution.
    scripts = REPO_ROOT / "ai-training/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_r4_3a_word_sequence_design_audit as r43a
    import run_r4_4b_ctc_sequence as r4b
    r43a.AUDIT_SPEAKERS = frozenset(speakers)
    records, reconstruction = r43a.build_word_records(audio_root)
    words = [word for word in records if word["usable"] and word["split"] == split]
    expected = contract["populations"][split]
    if len(words) != expected["usable_words"]:
        raise RuntimeError(f"{split} word mismatch")
    for word in words:
        word["target_ids"] = [r4b.PHONE_TO_ID[phone] for phone in word["observed"]]
        word["expected_ids"] = [r4b.PHONE_TO_ID[phone] for phone in word["expected"]]
    relation_counts = Counter(row["relation"] for word in words for row in word["clean_rows"])
    if {name: relation_counts[name] for name in RELATIONS} != expected["relations"]:
        raise RuntimeError(f"{split} relation mismatch")
    return words, {"words": len(words), "relations": dict(relation_counts), "reconstruction": reconstruction}


def score_population(words: list[dict[str, Any]], device: torch.device, checkpoint: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scripts = REPO_ROOT / "ai-training/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_r4_4b_ctc_sequence as r4b
    import run_r4_4c2_bigru_ctc_sequence as model_source
    import run_r4_4d0_ctc_hypothesis_feasibility as r4d0
    started = time.perf_counter()
    features, feature_report = r4b.materialize_features(words, device, f"r4_4d1_{split}")
    model = model_source.WordBiGRUCTCModel().to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    log_probs, decoded, inference_report = r4d0.inference(model, words, features, device)
    offsets = []; cursor = 0
    for word in words:
        offsets.append(cursor); cursor += len(word["expected_ids"])
    keep, delete, sub, sub_phone, scoring_report = r4d0.score_hypotheses(words, log_probs, device, offsets)
    raw_rows = r4d0.build_position_rows(words, log_probs, decoded, keep, delete, sub, sub_phone, offsets)
    rows = []
    for raw in raw_rows:
        word = words[raw["word_index"]]; source = word["clean_rows"][raw["position"]]
        rows.append({
            **raw, "source_csv_row": int(source["source_index"]),
            "word_start": float(word["mfa_start"]), "word_end": float(word["mfa_end"]),
            "score": np.float64(raw["raw_del_vs_best_nondelete"]),
            "keep_score": np.float64(raw["raw_keep"]), "delete_score": np.float64(raw["raw_delete"]),
            "best_sub_score": np.float64(raw["raw_best_sub"]),
            "best_sub_phone_index": int(r4b.PHONE_TO_ID[raw["best_sub_phone"]]),
            "phone_vocabulary": r4b.PHONE_VOCAB,
        })
    return rows, {"features": feature_report, "inference": inference_report,
                  "hypothesis_scoring": scoring_report, "seconds": time.perf_counter() - started}


def validation_diagnostics(rows: list[dict[str, Any]], matched_ids: list[int], contract: dict[str, Any]) -> dict[str, Any]:
    binary, false_deletion, three = row_metrics(rows)
    by_source = {row["source_csv_row"]: row for row in rows}
    matched_rows = [by_source[identity] for identity in matched_ids]
    matched_binary, _, _ = row_metrics(matched_rows)
    speakers = {}
    for speaker in contract["populations"]["validation"]["speakers"]:
        subset = [row for row in rows if row["speaker"] == speaker]
        metrics, fd, _ = row_metrics(subset)
        speakers[speaker] = {"word_count": len({row["word_id"] for row in subset}),
                             "expected_phone_rows": len(subset),
                             "deletion_support": sum(row["true_relation"] == "deletion" for row in subset),
                             "binary_macro_f1": metrics["binary_macro_f1"],
                             "balanced_accuracy": metrics["balanced_accuracy"],
                             "deletion_precision": metrics["deletion_precision"],
                             "deletion_recall": metrics["deletion_recall"],
                             "deletion_f1": metrics["deletion_f1"], **fd}
    continuous = {
        "deletion_vs_nondeletion": auc_metrics([row["true_relation"] for row in rows], [row["score"] for row in rows]),
        "deletion_vs_substitution": auc_metrics([row["true_relation"] for row in rows], [row["score"] for row in rows], "substitution"),
    }
    return {"binary": binary, "false_deletion": false_deletion, "three": three,
            "matched": matched_binary, "speakers": speakers, "continuous": continuous}


def read_matched_ids(path: Path, contract: dict[str, Any]) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    deletion = sum(row["role"] == "deletion" for row in records)
    nondeletion = sum(row["role"] == "non_deletion" for row in records)
    if (len(records), deletion, nondeletion) != (
        contract["matched_control"]["total_rows"], contract["matched_control"]["deletion_rows"],
        contract["matched_control"]["nondeletion_rows"],
    ):
        raise MatchedControlError("Matched-control support mismatch")
    return [int(row["source_csv_row"]) for row in records]


def write_prediction_exports(rows: list[dict[str, Any]], words: list[dict[str, Any]]) -> None:
    excluded = {"phone_vocabulary"}
    fields = [key for key in rows[0] if key not in excluded]
    with (LOCKED_OUTPUT / "validation_phone_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    grouped = defaultdict(list)
    for row in rows: grouped[row["word_index"]].append(row)
    with (LOCKED_OUTPUT / "validation_word_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, word in enumerate(words):
            word_rows = sorted(grouped[index], key=lambda row: row["position"])
            payload = {"word_id": word["word_id"], "speaker": word["speaker_id"],
                       "utterance": word["utterance_id"], "word": word["word"],
                       "expected_sequence": word["expected"], "manual_observed_sequence": word["observed"],
                       "true_relations": [row["true_relation"] for row in word_rows],
                       "predicted_relations": [row["predicted_relation"] for row in word_rows],
                       "scores": [float(row["score"]) for row in word_rows]}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def execute_locked() -> int:
    original, contract = load_contracts(True)
    verification = verify_all_sources(contract)
    train_speakers = contract["populations"]["train"]["speakers"]
    validation_speakers = contract["populations"]["validation"]["speakers"]
    # Guards run before importing reconstruction modules or resolving audio paths.
    guard_test_speakers_before_resolution(train_speakers)
    guard_test_speakers_before_resolution(validation_speakers)
    if LOCKED_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite locked execution: {LOCKED_OUTPUT}")
    LOCKED_OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(LOCKED_OUTPUT / "preflight.json", {
        "source_verification": verification, "contracts_loaded": True,
        "preserved_prior_stop_directory": str(OUTPUT_ROOT), "locked_subdirectory": str(LOCKED_OUTPUT),
        "train_threshold_calculated": False, "validation_scoring_started": False,
        "training": False, "test_accessed": False,
    })
    scripts = REPO_ROOT / "ai-training/scripts"
    if str(scripts) not in sys.path: sys.path.insert(0, str(scripts))
    import run_r4_4b_ctc_sequence as r4b
    audio_root = r4b.r3.require_audio_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_started = time.perf_counter()

    train_words, train_population = load_population("train", train_speakers, contract, audio_root)
    train_rows, train_compute = score_population(train_words, device, REPO_ROOT / contract["source_identities"]["r4_4c2_checkpoint"]["path"], "train")
    calibration = select_train_threshold(train_rows, contract)
    write_json(LOCKED_OUTPUT / "threshold_calibration_metrics.json", calibration)
    if calibration["selected"] is None:
        write_json(LOCKED_OUTPUT / "final_status.json", {"status": calibration["status"],
                                                          "validation_scoring": False, "test_accessed": False})
        return 0
    theta = float(calibration["selected"]["threshold"])
    apply_threshold(train_rows, theta)
    train_binary, train_fd, train_three = row_metrics(train_rows)
    threshold_payload = {
        "status": "FROZEN_BEFORE_VALIDATION", "score_family": "RAW",
        "score_equation": contract["decision_score"]["equation"],
        "train_population": contract["populations"]["train"], "selected_threshold": theta,
        "selection_protocol": contract["train_threshold_calibration"],
        "selection_result": calibration["selected"],
        "train_metrics": {"binary": train_binary, "false_deletion": train_fd, "three": train_three},
        "source_shas": verification["actual"], "test_policy": "CLOSED",
    }
    threshold_identity = freeze_threshold_artifact(LOCKED_OUTPUT / "train_calibrated_threshold.json", threshold_payload)
    write_json(LOCKED_OUTPUT / "threshold_identity.json", threshold_identity)

    # No validation population is reconstructed or scored before threshold freeze verification above.
    preflight = json.loads((LOCKED_OUTPUT / "preflight.json").read_text(encoding="utf-8"))
    preflight.update({"train_threshold_calculated": True, "threshold_identity": threshold_identity,
                      "validation_scoring_started": True})
    write_json(LOCKED_OUTPUT / "preflight.json", preflight)
    validation_started = time.perf_counter()
    validation_words, validation_population = load_population("validation", validation_speakers, contract, audio_root)
    validation_rows, validation_compute = score_population(
        validation_words, device, REPO_ROOT / contract["source_identities"]["r4_4c2_checkpoint"]["path"], "validation"
    )
    apply_threshold(validation_rows, theta)
    matched_path = REPO_ROOT / contract["matched_control"]["path"]
    required_ids = read_matched_ids(matched_path, contract)
    matched_control_guard(required_ids, [row["source_csv_row"] for row in validation_rows],
                          contract["matched_control"]["total_rows"])
    diagnostics = validation_diagnostics(validation_rows, required_ids, contract)
    status = classify_result(True, True, diagnostics["binary"], diagnostics["false_deletion"],
                             diagnostics["matched"], diagnostics["three"], diagnostics["speakers"],
                             diagnostics["continuous"], contract)

    write_json(LOCKED_OUTPUT / "validation_binary_metrics.json", {**diagnostics["binary"], **diagnostics["false_deletion"]})
    write_json(LOCKED_OUTPUT / "validation_3class_metrics.json", diagnostics["three"])
    write_json(LOCKED_OUTPUT / "matched_control_metrics.json", diagnostics["matched"])
    write_json(LOCKED_OUTPUT / "speaker_metrics.json", diagnostics["speakers"])
    phone_output = {}
    for phone in sorted({row["expected_phone"] for row in validation_rows}):
        phone_output[phone] = grouped_deletion_metrics([row for row in validation_rows if row["expected_phone"] == phone])
    phone_output["groups"] = {
        "D_T_R_L": grouped_deletion_metrics([row for row in validation_rows if row["expected_phone"] in {"D", "T", "R", "L"}]),
        "OTHER": grouped_deletion_metrics([row for row in validation_rows if row["expected_phone"] not in {"D", "T", "R", "L"}]),
    }
    write_json(LOCKED_OUTPUT / "phone_metrics.json", phone_output)
    position_output = {name: grouped_deletion_metrics([row for row in validation_rows if row["position_group"] == name])
                       for name in ("initial", "medial", "final", "single")}
    write_json(LOCKED_OUTPUT / "position_metrics.json", position_output)
    multi = {
        "pure_deletion": grouped_deletion_metrics([row for row in validation_rows if row["word_deletions"] > 0 and row["word_substitutions"] == 0]),
        "substitution_plus_deletion": grouped_deletion_metrics([row for row in validation_rows if row["word_deletions"] > 0 and row["word_substitutions"] > 0]),
        "multiple_deletion": grouped_deletion_metrics([row for row in validation_rows if row["word_deletions"] > 1]),
    }
    write_json(LOCKED_OUTPUT / "multi_error_metrics.json", multi)
    length_buckets = {
        "1": lambda n: n == 1, "2": lambda n: n == 2, "3": lambda n: n == 3,
        "4": lambda n: n == 4, "5": lambda n: n == 5, "6+": lambda n: n >= 6,
    }
    write_json(LOCKED_OUTPUT / "length_metrics.json", {name: grouped_deletion_metrics(
        [row for row in validation_rows if predicate(row["word_expected_length"])]) for name, predicate in length_buckets.items()})
    duration_buckets = {
        "lt_150ms": lambda d: d < .150, "150_250ms": lambda d: .150 <= d < .250,
        "250_400ms": lambda d: .250 <= d < .400, "400_600ms": lambda d: .400 <= d < .600,
        "ge_600ms": lambda d: d >= .600,
    }
    write_json(LOCKED_OUTPUT / "duration_metrics.json", {name: grouped_deletion_metrics(
        [row for row in validation_rows if predicate(row["word_duration"])]) for name, predicate in duration_buckets.items()})
    greedy_rows = [row for row in validation_rows if row["greedy_relation"] == "deletion"]
    greedy_true = [row for row in greedy_rows if row["true_relation"] == "deletion"]
    greedy_false = [row for row in greedy_rows if row["true_relation"] != "deletion"]
    write_json(LOCKED_OUTPUT / "greedy_rescue_metrics.json", {
        "greedy_true_deletions": len(greedy_true),
        "true_deletion_retention_rate": safe_div(sum(row["predicted_relation"] == "deletion" for row in greedy_true), len(greedy_true)),
        "greedy_false_deletions": len(greedy_false),
        "false_deletion_rescue_rate": safe_div(sum(row["predicted_relation"] != "deletion" for row in greedy_false), len(greedy_false)),
    })
    score_distributions = {relation: distribution(row["score"] for row in validation_rows if row["true_relation"] == relation)
                           for relation in RELATIONS}
    write_json(LOCKED_OUTPUT / "score_distributions.json", score_distributions)
    train_medians = {relation: float(np.median([row["score"] for row in train_rows if row["true_relation"] == relation]))
                     for relation in RELATIONS}
    validation_medians = {relation: float(np.median([row["score"] for row in validation_rows if row["true_relation"] == relation]))
                          for relation in RELATIONS}
    write_json(LOCKED_OUTPUT / "threshold_transfer.json", {
        "threshold": theta, "train_medians": train_medians, "validation_medians": validation_medians,
        "absolute_median_shift": {name: abs(validation_medians[name] - train_medians[name]) for name in RELATIONS},
        "continuous": diagnostics["continuous"], "threshold_adjusted": False,
    })
    write_prediction_exports(validation_rows, validation_words)
    write_json(LOCKED_OUTPUT / "compute_report.json", {
        "device": str(device), "train": train_compute, "validation": validation_compute,
        "validation_seconds": time.perf_counter() - validation_started,
        "total_seconds": time.perf_counter() - total_started,
        "train_population": train_population, "validation_population": validation_population,
    })
    write_json(LOCKED_OUTPUT / "final_status.json", {
        "status": status, "threshold_sha256": threshold_identity["sha256"],
        "threshold_modified_after_validation": False, "training": False, "test_accessed": False,
    })
    (LOCKED_OUTPUT / "r4_4d1_report.md").write_text(
        f"# R4-4D1 Locked Hypothesis Validation\n\nFinal: **{status}**\n\nThreshold SHA: `{threshold_identity['sha256']}`\n",
        encoding="utf-8",
    )
    hashes = {path.name: sha256(path) for path in sorted(LOCKED_OUTPUT.iterdir())
              if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(LOCKED_OUTPUT / "artifact_hashes.json", {"algorithm": "SHA-256", "files": hashes,
                                                          "note": "manifest excludes itself"})
    print(json.dumps({"status": status, "threshold": theta, "output": str(LOCKED_OUTPUT)}, indent=2))
    return 0


def synthetic_contract() -> dict[str, Any]:
    _, contract = load_contracts(True)
    return contract


def run_synthetic_tests() -> dict[str, Any]:
    contract = synthetic_contract(); tests = {}
    tests["A_threshold_equality"] = relation_decision(1.0, 1.0, 2.0, 3.0, 7)[0] == "deletion"
    tests["B_keep_sub_equality"] = relation_decision(0.9, 1.0, 2.0, 2.0, 7)[0] == "correct"
    phone, score = best_substitution({8: -2.0, 3: -2.0, 9: -4.0})
    tests["C_best_sub_tie"] = phone == 3 and score == -2.0
    no_eligible_rows = [{"score": 0.0, "true_relation": "correct", "speaker": "S",
                         "keep_score": 1.0, "best_sub_score": 0.0}]
    calibration = select_train_threshold(no_eligible_rows, contract)
    callback_called = False
    if calibration["selected"] is not None:
        callback_called = True
    tests["D_no_eligible_threshold"] = (
        calibration["status"] == "R4_4D1_TRAIN_CALIBRATION_NO_ELIGIBLE_THRESHOLD" and not callback_called
    )
    speakers = {"S": {"deletion_support": 30, "deletion_recall": .50}}
    binary_confirmed = {"binary_macro_f1": .71, "deletion_recall": .46, "deletion_f1": .41}
    fd = {"substitution_false_deletion": .20}; matched_confirmed = {"binary_macro_f1": .61, "deletion_f1": .56}
    three_confirmed = {"macro_f1": .41}; high_auc = {"deletion_vs_nondeletion": {"roc_auc": .80},
                                                        "deletion_vs_substitution": {"roc_auc": .75}}
    tests["E_confirmed"] = classify_result(True, True, binary_confirmed, fd, matched_confirmed,
                                             three_confirmed, speakers, high_auc, contract) == "R4_4D1_HYPOTHESIS_DELETION_CONFIRMED"
    binary_partial = {"binary_macro_f1": .66, "deletion_recall": .41, "deletion_f1": .36}
    matched_partial = {"binary_macro_f1": .59, "deletion_f1": .51}; three_partial = {"macro_f1": .39}
    speakers_partial = {"S": {"deletion_support": 30, "deletion_recall": .21}}
    tests["F_strong_partial"] = classify_result(True, True, binary_partial, fd, matched_partial,
                                                  three_partial, speakers_partial, high_auc, contract) == "R4_4D1_HYPOTHESIS_DELETION_STRONG_PARTIAL"
    weak_binary = {"binary_macro_f1": .50, "deletion_recall": .30, "deletion_f1": .20}
    weak_matched = {"binary_macro_f1": .50, "deletion_f1": .20}; weak_three = {"macro_f1": .30}
    tests["G_transfer_fail"] = classify_result(True, True, weak_binary, fd, weak_matched,
                                                 weak_three, speakers_partial, high_auc, contract) == "R4_4D1_HYPOTHESIS_THRESHOLD_TRANSFER_FAIL"
    low_auc = {"deletion_vs_nondeletion": {"roc_auc": .74}, "deletion_vs_substitution": {"roc_auc": .75}}
    tests["H_signal_not_confirmed"] = classify_result(True, True, weak_binary, fd, weak_matched,
                                                        weak_three, speakers_partial, low_auc, contract) == "R4_4D1_HYPOTHESIS_SIGNAL_NOT_CONFIRMED"
    try:
        matched_control_guard([1, 2], [1], expected_count=2); matched_failed = False
    except MatchedControlError:
        matched_failed = True
    tests["I_matched_missing_identity"] = matched_failed
    resolver_called = False
    def resolver() -> None:
        nonlocal resolver_called; resolver_called = True
    try:
        guard_test_speakers_before_resolution(["ASI"], resolver); guard_failed = False
    except TestSpeakerGuardError:
        guard_failed = True
    tests["J_test_guard_before_resolution"] = guard_failed and not resolver_called
    logits = torch.randn(7, 41, generator=torch.Generator().manual_seed(42))
    empty_score = raw_ctc_score(torch.log_softmax(logits, dim=-1), [], blank=40)
    tests["K_empty_delete_finite"] = math.isfinite(empty_score)
    edge_candidates = threshold_candidates([1.0, 2.0, 2.0])
    tests["L_threshold_edges"] = (
        edge_candidates.tolist() == [np.nextafter(1.0, -np.inf), 1.0, 2.0, np.nextafter(2.0, np.inf)]
    )
    empty_binary = binary_metrics([], [])
    tests["M_zero_denominator"] = all(empty_binary[key] == 0.0 for key in (
        "accuracy", "balanced_accuracy", "binary_macro_f1", "deletion_precision",
        "deletion_recall", "deletion_f1", "nondeletion_f1"
    ))
    precedence = [
        classify_result(False, False, None, None, None, None, None, None, contract),
        classify_result(True, False, None, None, None, None, None, None, contract),
        classify_result(True, True, binary_confirmed, fd, matched_confirmed, three_confirmed, speakers, high_auc, contract),
        classify_result(True, True, binary_partial, fd, matched_partial, three_partial, speakers_partial, high_auc, contract),
        classify_result(True, True, weak_binary, fd, weak_matched, weak_three, speakers_partial, high_auc, contract),
        classify_result(True, True, weak_binary, fd, weak_matched, weak_three, speakers_partial, low_auc, contract),
    ]
    tests["N_classification_precedence"] = precedence == contract["result_classification"]["precedence"]
    passed = sum(bool(value) for value in tests.values())
    return {"status": "PASS" if passed == len(tests) else "FAIL", "passed": passed,
            "total": len(tests), "tests": tests, "empty_target_raw_score": empty_score,
            "real_train_threshold_calculated": False, "validation_hypothesis_scoring": False,
            "r4_test_accessed": False, "neural_training": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        report = run_synthetic_tests(); print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1
    return execute_locked()


if __name__ == "__main__":
    raise SystemExit(main())
