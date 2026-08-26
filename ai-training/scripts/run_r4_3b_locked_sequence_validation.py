from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import r4_3b_sequence_dp as sequence_dp  # noqa: E402
import run_l2_arctic_observed_phone_r3_1a as r3  # noqa: E402
import run_r4_3a_word_sequence_design_audit as r43a  # noqa: E402


REPO_ROOT = r3.REPO_ROOT
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r4_3b_locked_sequence_validation"
CONTRACT_JSON = REPO_ROOT / "ai-training/experiments/r4_3b_frozen_numerical_contract/r4_3b_complete_numerical_contract.json"
CONTRACT_MD = REPO_ROOT / "ai-training/experiments/r4_3b_frozen_numerical_contract/r4_3b_complete_numerical_contract.md"
CHECKPOINT = REPO_ROOT / "ai-training/experiments/r3_1d_observed_phone_seed42_48epochs/R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt"
V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
MATCHED_PATH = REPO_ROOT / "ai-training/experiments/r4_0_deletion_feasibility/matched_control_validation_rows.csv"
EXPECTED_HASHES = {
    "contract_json": "2E65880C3E4AFD0EFBBB8014AAD8E1B04B051267470C17A3E733244176D18FB5",
    "contract_md": "CDE8B1210610ACFA35A0AB1991F4C29A76441CEBBED09C192857810BF906BA43",
    "checkpoint": "5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "matched": "864591A259F0C7E6F16828C5F755049CFCF0F79A8B586ADA354D190F9C5C7823",
    "scorer": "3DA0A8A238C5FDF05F368F010558263CC98E73FC49E4C4600BBEB68E221A86D1",
}
SCORER_PATH = SCRIPTS_DIR / "r4_3b_sequence_dp.py"
RELATIONS = ("correct", "substitution", "deletion")
RELATION_TO_ID = {name: index for index, name in enumerate(RELATIONS)}
OP_TO_RELATION = {"MATCH": "correct", "SUBSTITUTION": "substitution", "DELETE_EXPECTED": "deletion"}
EXPECTED_TRAIN_COUNTS = Counter(correct=48_893, substitution=5_867, deletion=1_544)
EXPECTED_USABLE_WORDS = {"train": 16_259, "validation": 7_728}
EXPECTED_VALIDATION_PROBES = 77_515
FROZEN_PRIORS = {
    "MATCH": -0.14116417177608467,
    "SUBSTITUTION": -2.261305001061487,
    "DELETE_EXPECTED": -3.595794950992514,
}
BASELINES = {
    "duration_only": {"macro_f1": 0.668146, "balanced_accuracy": 0.706301, "deletion_f1": 0.364164},
    "r4_1_full": {"macro_f1": 0.657336, "balanced_accuracy": 0.683828, "deletion_f1": 0.341612},
    "r4_2a": {"macro_f1": 0.566997, "balanced_accuracy": 0.664977, "deletion_f1": 0.197525},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p25": None, "p75": None,
                "p90": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)), "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)), "min": float(array.min()), "max": float(array.max()),
    }


def binary_metrics(truth: list[int], prediction: list[int]) -> dict[str, Any]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(truth, prediction):
        matrix[actual, predicted] += 1
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    non_precision = tn / (tn + fn) if tn + fn else 0.0
    non_recall = tn / (tn + fp) if tn + fp else 0.0
    non_f1 = 2 * non_precision * non_recall / (non_precision + non_recall) if non_precision + non_recall else 0.0
    deletion_precision = tp / (tp + fp) if tp + fp else 0.0
    deletion_recall = tp / (tp + fn) if tp + fn else 0.0
    deletion_f1 = 2 * deletion_precision * deletion_recall / (deletion_precision + deletion_recall) if deletion_precision + deletion_recall else 0.0
    total = len(truth)
    return {
        "rows": total, "non_deletion_support": tn + fp, "deletion_support": fn + tp,
        "accuracy": (tn + tp) / total if total else 0.0,
        "balanced_accuracy": (non_recall + deletion_recall) / 2,
        "macro_f1": (non_f1 + deletion_f1) / 2,
        "deletion_precision": deletion_precision, "deletion_recall": deletion_recall,
        "deletion_f1": deletion_f1, "confusion_matrix": matrix.tolist(),
    }


def multiclass_metrics(truth: list[str], prediction: list[str]) -> dict[str, Any]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for actual, predicted in zip(truth, prediction):
        matrix[RELATION_TO_ID[actual], RELATION_TO_ID[predicted]] += 1
    per_class: dict[str, Any] = {}
    for index, relation in enumerate(RELATIONS):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[relation] = {"support": int(matrix[index, :].sum()), "precision": precision, "recall": recall, "f1": f1}
    return {
        "rows": len(truth), "labels": list(RELATIONS),
        "macro_f1": float(np.mean([per_class[name]["f1"] for name in RELATIONS])),
        "macro_precision": float(np.mean([per_class[name]["precision"] for name in RELATIONS])),
        "macro_recall": float(np.mean([per_class[name]["recall"] for name in RELATIONS])),
        "per_class": per_class, "confusion_matrix": matrix.tolist(),
    }


def subset_binary(rows: list[dict[str, Any]], indexes: list[int]) -> dict[str, Any]:
    return binary_metrics([rows[i]["binary_truth"] for i in indexes], [rows[i]["binary_prediction"] for i in indexes])


def verify_and_load_contract() -> tuple[dict[str, Any], dict[str, str]]:
    paths = {
        "contract_json": CONTRACT_JSON, "contract_md": CONTRACT_MD, "checkpoint": CHECKPOINT,
        "v4": V4_PATH, "matched": MATCHED_PATH, "scorer": SCORER_PATH,
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    mismatches = {name: {"expected": EXPECTED_HASHES[name], "actual": value}
                  for name, value in actual.items() if value != EXPECTED_HASHES[name]}
    if mismatches:
        raise RuntimeError(f"R4_3B_CONTRACT_VERIFICATION_FAIL: {mismatches}")
    contract = json.loads(CONTRACT_JSON.read_text(encoding="utf-8"))
    contract_priors = contract["train_only_operation_priors"]["values"]
    loaded = {
        "MATCH": float(contract_priors["MATCH"]["log_prior"]),
        "SUBSTITUTION": float(contract_priors["SUBSTITUTION"]["log_prior"]),
        "DELETE_EXPECTED": float(contract_priors["DELETE_EXPECTED"]["log_prior"]),
    }
    if loaded != FROZEN_PRIORS or sequence_dp.LOG_PRIORS != FROZEN_PRIORS:
        raise RuntimeError("Frozen scorer/contract operation priors differ")
    sequence_dp.LOG_PRIORS = loaded
    sequence_dp.TIE_EPSILON = float(contract["tie_policy"]["tie_epsilon"])
    if tuple(contract["model_and_data"]["phone_vocabulary_index_order"]) != sequence_dp.PHONE_VOCAB:
        raise RuntimeError("Frozen phone vocabulary mismatch")
    return contract, actual


def run_synthetic_tests() -> dict[str, Any]:
    import importlib.util
    test_path = REPO_ROOT / "ai-training/tests/test_r4_3b_sequence_dp.py"
    spec = importlib.util.spec_from_file_location("test_r4_3b_sequence_dp", test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load synthetic tests")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests = sorted(name for name in dir(module) if name.startswith("test_"))
    for name in tests:
        getattr(module, name)()
    if len(tests) != 7:
        raise RuntimeError(f"Expected 7 synthetic tests, got {len(tests)}")
    return {"status": "PASS", "passed": len(tests), "expected": 7, "tests": tests}


def load_v4_identity_rows() -> dict[int, dict[str, str]]:
    with V4_PATH.open(encoding="utf-8", newline="") as handle:
        return {
            index: row for index, row in enumerate(csv.DictReader(handle))
            if row["speaker_id"] in r3.VALIDATION_SPEAKERS
        }


def verify_matched(v4_rows: dict[int, dict[str, str]], scoreable_sources: set[int]) -> tuple[set[int], dict[str, Any]]:
    with MATCHED_PATH.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 1_564:
        raise RuntimeError(f"Frozen matched rows changed: {len(records)}")
    matched: set[int] = set()
    relation_counts: Counter[str] = Counter()
    missing_reason_rows: list[int] = []
    fields = ("speaker_id", "audio_path", "utterance_id", "start_time", "end_time", "expected_phone_canonical", "relation")
    for record in records:
        source = int(record["source_csv_row"]) - 2
        if source in matched or source not in v4_rows:
            raise RuntimeError("Frozen matched identity duplicate/missing")
        if any(str(v4_rows[source][field]) != str(record[field]) for field in fields):
            raise RuntimeError(f"Frozen matched identity mismatch at source row {source + 2}")
        matched.add(source)
        relation_counts[record["relation"]] += 1
        if source not in scoreable_sources:
            missing_reason_rows.append(source + 2)
    if relation_counts["deletion"] != 782 or len(records) - relation_counts["deletion"] != 782:
        raise RuntimeError(f"Frozen matched class counts differ: {dict(relation_counts)}")
    scoreable = matched & scoreable_sources
    return matched, {
        "identity_sha256_verified": True, "frozen_rows": len(matched), "frozen_deletion": 782,
        "frozen_non_deletion": 782, "scoreable_under_r4_3a_word_policy": len(scoreable),
        "excluded_by_r4_3a_word_policy": len(matched - scoreable_sources),
        "first_excluded_source_csv_rows": missing_reason_rows[:50],
        "exact_matched_metrics_evaluable": len(scoreable) == len(matched),
        "rebuilt_or_resampled": False,
    }


def infer_validation(words: list[dict[str, Any]], contract: dict[str, Any], audio_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    stride = float(contract["probe_grid"]["stride_seconds"])
    offsets = [0]
    centers: list[float] = []
    for word in words:
        word_centers = sequence_dp.probe_centers(float(word["mfa_start"]), float(word["mfa_end"]), stride)
        centers.extend(word_centers)
        offsets.append(len(centers))
    if len(centers) != EXPECTED_VALIDATION_PROBES:
        raise RuntimeError(f"Validation probe count mismatch: {len(centers)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = r3.SmallPronunciationCNNAttention().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    extractor = r3.FixedLogMel().to(device).eval()
    store = r3.SequentialWaveStore()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    waveform_buffer: list[np.ndarray] = []
    output_chunks: list[np.ndarray] = []

    def flush() -> None:
        if not waveform_buffer:
            return
        waveforms = torch.from_numpy(np.stack(waveform_buffer)).float().to(device)
        features = extractor(waveforms)
        for begin in range(0, len(features), r3.BATCH_SIZE):
            logits = model(features[begin:begin + r3.BATCH_SIZE])
            output_chunks.append(torch.log_softmax(logits, dim=1).cpu().numpy().astype(np.float32, copy=False))
        waveform_buffer.clear()

    with torch.inference_mode():
        for word_index, word in enumerate(words):
            audio = store.load(word["audio_path"])
            for center in centers[offsets[word_index]:offsets[word_index + 1]]:
                waveform_buffer.append(r3.centered_window(audio, center, center))
                if len(waveform_buffer) == r3.PREPROCESS_BATCH_SIZE:
                    flush()
            if (word_index + 1) % 1000 == 0:
                print(f"Acoustic evidence: {word_index + 1}/{len(words)} words", flush=True)
        flush()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    log_probabilities = np.concatenate(output_chunks, axis=0)
    if log_probabilities.shape != (len(centers), 40) or not np.isfinite(log_probabilities).all():
        raise RuntimeError(f"Invalid acoustic evidence: {log_probabilities.shape}")
    compute = {
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_epoch": int(checkpoint["epoch"]), "validation_probes": len(centers),
        "median_probes_per_word": float(np.median(np.diff(np.asarray(offsets)))),
        "acoustic_inference_seconds": inference_seconds,
        "probes_per_second": len(centers) / inference_seconds,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "audio_root_identity": str(audio_root),
    }
    return log_probabilities, np.asarray(offsets, dtype=np.int64), np.asarray(centers, dtype=np.float64), compute


def score_words(words: list[dict[str, Any]], log_probs: np.ndarray, offsets: np.ndarray, matched: set[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    phone_rows: list[dict[str, Any]] = []
    word_paths: list[dict[str, Any]] = []
    started = time.perf_counter()
    no_valid = 0
    for word_index, word in enumerate(words):
        evidence = np.asarray(log_probs[offsets[word_index]:offsets[word_index + 1]], dtype=np.float64)
        result = sequence_dp.align(word["expected"], evidence)
        if result["status"] != "OK":
            no_valid += 1
            word_paths.append({"word_id": word["word_id"], "status": "NO_VALID_PATH", "number_of_probes": len(evidence)})
            continue
        best = result["best"]
        second = result["second_best"]
        if len(best.steps) != len(word["clean_rows"]):
            raise RuntimeError(f"Step/ground-truth mismatch: {word['word_id']}")
        predicted_operations: list[dict[str, Any]] = []
        truth_path: list[str] = []
        for position, (step, source) in enumerate(zip(best.steps, word["clean_rows"])):
            relation = OP_TO_RELATION[step.operation]
            truth_path.append(source["relation"])
            expected_index = sequence_dp.PHONE_TO_ID[source["expected"]]
            predicted_phone = sequence_dp.PHONE_VOCAB[step.observed_phone_index] if step.operation == "SUBSTITUTION" else ""
            if step.span_length:
                span_mean = evidence[step.probe_start:step.probe_end].mean(axis=0)
                expected_evidence = float(span_mean[expected_index])
                alternative = span_mean.copy(); alternative[expected_index] = -np.inf
                alternative_index = int(np.argmax(alternative))
                alternative_evidence = float(span_mean[alternative_index])
                span_centers = "|".join(f"{value:.9f}" for value in (
                    # centers are reproducible from the frozen grid; indexes are authoritative.
                    sequence_dp.probe_centers(float(word["mfa_start"]), float(word["mfa_end"]), 0.04)[step.probe_start:step.probe_end]
                ))
            else:
                expected_evidence = alternative_evidence = None
                alternative_index = None
                span_centers = ""
            binary_truth = int(source["relation"] == "deletion")
            binary_prediction = int(relation == "deletion")
            row = {
                "source_csv_row": source["source_index"] + 2,
                "speaker": word["speaker_id"], "word_id": word["word_id"], "word": word["word"],
                "utterance": word["utterance_id"], "word_start": word["mfa_start"], "word_end": word["mfa_end"],
                "expected_phone": source["expected"], "expected_phone_index": position,
                "true_relation": source["relation"], "true_observed_phone": source["observed"],
                "predicted_relation": relation, "predicted_observed_phone": predicted_phone,
                "assigned_probe_start": step.probe_start, "assigned_probe_end": step.probe_end,
                "assigned_acoustic_span_centers": span_centers,
                "span_mean_expected_evidence": expected_evidence,
                "span_best_alternative_phone": sequence_dp.PHONE_VOCAB[alternative_index] if alternative_index is not None else "",
                "span_best_alternative_evidence": alternative_evidence,
                "operation_acoustic_contribution": step.acoustic_score,
                "operation_prior_contribution": step.prior_score,
                "operation_score_contribution": step.acoustic_score + step.prior_score,
                "complete_path_score": best.score,
                "ambiguity_flag": bool(result["numerically_ambiguous"]),
                "is_strict_matched_row": source["source_index"] in matched,
                "binary_truth": binary_truth, "binary_prediction": binary_prediction,
                "word_position": "single" if len(best.steps) == 1 else ("initial" if position == 0 else ("final" if position == len(best.steps) - 1 else "medial")),
            }
            phone_rows.append(row)
            predicted_operations.append({
                "expected": source["expected"], "operation": step.operation,
                "predicted_observed": predicted_phone or None,
                "probe_start": step.probe_start, "probe_end": step.probe_end,
                "score_contribution": step.acoustic_score + step.prior_score,
            })
        word_paths.append({
            "word_id": word["word_id"], "speaker": word["speaker_id"], "utterance": word["utterance_id"],
            "word": word["word"], "word_start": word["mfa_start"], "word_end": word["mfa_end"],
            "expected_sequence": word["expected"], "predicted_operation_path": predicted_operations,
            "ground_truth_relation_path": truth_path, "number_of_probes": len(evidence),
            "best_path_score": best.score, "second_best_score": second.score if second is not None else None,
            "score_gap": result["path_score_gap"], "ambiguity_flag": bool(result["numerically_ambiguous"]),
            "no_valid_path": False,
            "pure_deletion_word": word["deletion"] > 0 and word["substitution"] == 0,
            "substitution_plus_deletion_word": word["deletion"] > 0 and word["substitution"] > 0,
            "multiple_deletion_word": word["deletion"] > 1,
        })
    return phone_rows, word_paths, {"dp_seconds": time.perf_counter() - started, "no_valid_paths": no_valid,
                                     "no_valid_path_rate": no_valid / len(words)}


def grouped_metrics(rows: list[dict[str, Any]], key_fn, groups: Iterable[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in groups:
        indexes = [i for i, row in enumerate(rows) if key_fn(row) == group]
        output[group] = subset_binary(rows, indexes)
    return output


def evaluate(rows: list[dict[str, Any]], words: list[dict[str, Any]], word_paths: list[dict[str, Any]], matched: set[int], matched_info: dict[str, Any]) -> dict[str, Any]:
    binary = binary_metrics([row["binary_truth"] for row in rows], [row["binary_prediction"] for row in rows])
    three = multiclass_metrics([row["true_relation"] for row in rows], [row["predicted_relation"] for row in rows])
    correct_rows = [row for row in rows if row["true_relation"] == "correct"]
    substitution_rows = [row for row in rows if row["true_relation"] == "substitution"]
    correct_false = float(np.mean([row["binary_prediction"] for row in correct_rows]))
    substitution_false = float(np.mean([row["binary_prediction"] for row in substitution_rows]))

    if matched_info["exact_matched_metrics_evaluable"]:
        matched_indexes = [i for i, row in enumerate(rows) if int(row["source_csv_row"]) - 2 in matched]
        matched_metrics: dict[str, Any] = subset_binary(rows, matched_indexes)
        matched_metrics["status"] = "PASS_EXACT_IDENTITY_SET"
    else:
        matched_metrics = {
            "status": "NOT_EVALUABLE_EXACT_IDENTITY_SET",
            "reason": "135 frozen matched rows belong to addition-containing/unresolved words excluded by the frozen R4-3A eligibility policy",
            **matched_info,
            "metrics_computed_on_intersection": False,
        }

    speakers: dict[str, Any] = {}
    for speaker in r3.VALIDATION_SPEAKERS:
        indexes = [i for i, row in enumerate(rows) if row["speaker"] == speaker]
        metrics = subset_binary(rows, indexes)
        sub = [rows[i] for i in indexes if rows[i]["true_relation"] == "substitution"]
        metrics.update({
            "usable_words": sum(word["speaker_id"] == speaker for word in words),
            "substitution_support": len(sub),
            "substitution_false_deletion_rate": float(np.mean([row["binary_prediction"] for row in sub])) if sub else None,
        })
        speakers[speaker] = metrics

    required_phones = {"D", "T", "R", "L", "N", "Z", "V", "K", "AH", "HH", "S"}
    deletion_support = Counter(row["expected_phone"] for row in rows if row["true_relation"] == "deletion")
    focus_phones = required_phones | {phone for phone, count in deletion_support.items() if count >= 20}
    phones: dict[str, Any] = {}
    for phone in sorted(focus_phones, key=sequence_dp.PHONE_TO_ID.get):
        indexes = [i for i, row in enumerate(rows) if row["expected_phone"] == phone]
        phones[phone] = subset_binary(rows, indexes)
    dominant = {"D", "T", "R", "L"}
    concentration = {}
    for name, predicate in (
        ("D_T_R_L", lambda row: row["expected_phone"] in dominant),
        ("all_other_phones", lambda row: row["expected_phone"] not in dominant),
    ):
        concentration[name] = subset_binary(rows, [i for i, row in enumerate(rows) if predicate(row)])

    positions = grouped_metrics(rows, lambda row: row["word_position"], ("initial", "medial", "final", "single"))
    word_by_id = {word["word_id"]: word for word in word_paths}
    categories = {
        "pure_deletion_words": lambda path: path.get("pure_deletion_word", False),
        "substitution_plus_deletion_words": lambda path: path.get("substitution_plus_deletion_word", False),
        "multiple_deletion_words": lambda path: path.get("multiple_deletion_word", False),
    }
    multi_error = {}
    for name, predicate in categories.items():
        word_ids = {word_id for word_id, path in word_by_id.items() if predicate(path)}
        indexes = [i for i, row in enumerate(rows) if row["word_id"] in word_ids]
        multi_error[name] = {**subset_binary(rows, indexes), "words": len(word_ids)}

    ambiguous_word_ids = {path["word_id"] for path in word_paths if path.get("ambiguity_flag")}
    ambiguity = {
        "words": len(word_paths),
        "unique_optimal_words": len(word_paths) - len(ambiguous_word_ids),
        "numerically_ambiguous_words": len(ambiguous_word_ids),
        "numerically_ambiguous_rate": len(ambiguous_word_ids) / len(word_paths),
        "deletion_predictions_from_ambiguous_words": sum(row["predicted_relation"] == "deletion" and row["word_id"] in ambiguous_word_ids for row in rows),
        "path_score_gap": quantiles(path["score_gap"] for path in word_paths if path.get("score_gap") is not None),
    }
    for name, predicate in (("ambiguous", lambda row: row["word_id"] in ambiguous_word_ids),
                            ("unambiguous", lambda row: row["word_id"] not in ambiguous_word_ids)):
        indexes = [i for i, row in enumerate(rows) if predicate(row)]
        ambiguity[f"{name}_binary"] = subset_binary(rows, indexes)
        ambiguity[f"{name}_3class"] = multiclass_metrics(
            [rows[i]["true_relation"] for i in indexes], [rows[i]["predicted_relation"] for i in indexes]
        ) if indexes else None

    covered_sub = [row for row in substitution_rows if row["predicted_relation"] == "substitution"]
    substitution_phone = {
        "true_substitution_support": len(substitution_rows),
        "predicted_substitution_coverage": len(covered_sub),
        "coverage_rate": len(covered_sub) / len(substitution_rows),
        "observed_phone_top1_correct": sum(row["predicted_observed_phone"] == row["true_observed_phone"] for row in covered_sub),
    }
    substitution_phone["top1_accuracy_on_covered"] = (
        substitution_phone["observed_phone_top1_correct"] / len(covered_sub) if covered_sub else 0.0
    )

    gates = {
        "binary_macro_f1": binary["macro_f1"] >= 0.70,
        "deletion_f1": binary["deletion_f1"] >= 0.40,
        "deletion_recall": binary["deletion_recall"] >= 0.45,
        "macro_f1_gain_over_duration": binary["macro_f1"] - 0.668146 >= 0.03,
        "matched_macro_f1": matched_metrics.get("macro_f1", -math.inf) >= 0.60,
        "matched_deletion_f1": matched_metrics.get("deletion_f1", -math.inf) >= 0.55,
        "substitution_false_deletion": substitution_false <= 0.25,
        "speaker_recall": all(value["deletion_recall"] >= 0.25 for value in speakers.values() if value["deletion_support"] >= 30),
    }
    if all(gates.values()):
        status = "R4_3B_SEQUENCE_DELETION_CONFIRMED"
    elif binary["macro_f1"] > BASELINES["duration_only"]["macro_f1"] or binary["deletion_f1"] > BASELINES["duration_only"]["deletion_f1"]:
        status = "R4_3B_SEQUENCE_DELETION_PARTIAL"
    else:
        status = "R4_3B_SEQUENCE_DELETION_NOT_CONFIRMED"
    return {
        "binary": binary, "three_class": three, "matched": matched_metrics,
        "correct_false_deletion_rate": correct_false, "substitution_false_deletion_rate": substitution_false,
        "speakers": speakers, "phones": phones, "phone_concentration": concentration,
        "positions": positions, "multi_error": multi_error, "ambiguity": ambiguity,
        "substitution_phone": substitution_phone, "gates": gates, "final_status": status,
    }


def save_outputs(contract: dict[str, Any], hashes: dict[str, str], synthetic: dict[str, Any], words: list[dict[str, Any]],
                 log_probs: np.ndarray, offsets: np.ndarray, centers: np.ndarray, phone_rows: list[dict[str, Any]],
                 word_paths: list[dict[str, Any]], evaluation: dict[str, Any], compute: dict[str, Any],
                 dp_compute: dict[str, Any], matched_info: dict[str, Any], total_seconds: float) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        EXPERIMENT_DIR / "validation_acoustic_evidence.npz", log_probabilities=log_probs,
        word_probe_offsets=offsets, probe_centers_seconds=centers,
        word_ids=np.asarray([word["word_id"] for word in words]),
    )
    write_json(EXPERIMENT_DIR / "preflight.json", {
        "contract_and_sources_verified": True, "hashes": hashes, "synthetic_tests": synthetic,
        "train_usable_words": EXPECTED_USABLE_WORDS["train"], "validation_usable_words": len(words),
        "matched_control": matched_info, "r4_test_paths_resolved": False, "r4_test_audio_accessed": False,
        "r4_test_inference": False, "training_performed": False, "r3_modified": False, "mfa_runtime_modified": False,
    })
    write_json(EXPERIMENT_DIR / "scorer_identity.json", {
        "path": str(SCORER_PATH.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": hashes["scorer"],
        "synthetic_tests": synthetic, "modified_after_validation_started": False,
    })
    write_json(EXPERIMENT_DIR / "operation_priors.json", contract["train_only_operation_priors"])
    write_json(EXPERIMENT_DIR / "validation_binary_metrics.json", evaluation["binary"] | {
        "correct_false_deletion_rate": evaluation["correct_false_deletion_rate"],
        "substitution_false_deletion_rate": evaluation["substitution_false_deletion_rate"],
        "macro_f1_delta_vs_duration": evaluation["binary"]["macro_f1"] - 0.668146,
        "gates": evaluation["gates"],
    })
    write_json(EXPERIMENT_DIR / "validation_3class_metrics.json", evaluation["three_class"])
    write_json(EXPERIMENT_DIR / "matched_control_metrics.json", evaluation["matched"])
    write_json(EXPERIMENT_DIR / "speaker_metrics.json", evaluation["speakers"])
    write_json(EXPERIMENT_DIR / "phone_metrics.json", {"phones": evaluation["phones"], "groups": evaluation["phone_concentration"]})
    write_json(EXPERIMENT_DIR / "position_metrics.json", evaluation["positions"])
    write_json(EXPERIMENT_DIR / "multi_error_metrics.json", evaluation["multi_error"])
    write_json(EXPERIMENT_DIR / "ambiguity_metrics.json", evaluation["ambiguity"])
    write_json(EXPERIMENT_DIR / "substitution_phone_diagnostic.json", evaluation["substitution_phone"])
    compute_report = {
        "train_operation_count_computation_seconds": 0.0,
        "train_counts_reproduced_in_metadata_preflight": True,
        **compute, **dp_compute, "total_experiment_seconds": total_seconds,
    }
    write_json(EXPERIMENT_DIR / "compute_report.json", compute_report)
    write_json(EXPERIMENT_DIR / "baseline_comparison.json", {**BASELINES, "r4_3b": evaluation["binary"]})

    csv_fields = [
        "source_csv_row", "speaker", "word_id", "word", "utterance", "word_start", "word_end",
        "expected_phone", "expected_phone_index", "true_relation", "true_observed_phone",
        "predicted_relation", "predicted_observed_phone", "assigned_probe_start", "assigned_probe_end",
        "assigned_acoustic_span_centers", "span_mean_expected_evidence", "span_best_alternative_phone",
        "span_best_alternative_evidence", "operation_acoustic_contribution", "operation_prior_contribution",
        "operation_score_contribution", "complete_path_score", "ambiguity_flag", "is_strict_matched_row",
    ]
    with (EXPERIMENT_DIR / "validation_phone_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(phone_rows)
    with (EXPERIMENT_DIR / "validation_word_paths.jsonl").open("w", encoding="utf-8") as handle:
        for row in word_paths:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    status = {
        "final_status": evaluation["final_status"], "all_frozen_gates_pass": all(evaluation["gates"].values()),
        "gates": evaluation["gates"], "exact_matched_control_evaluable": matched_info["exact_matched_metrics_evaluable"],
        "post_hoc_changes": False, "r4_test_accessed": False, "test_paths_resolved": False,
        "test_audio_read": False, "test_inference": False, "new_neural_training": False,
        "frozen_r3_modified": False, "mfa_runtime_modified": False,
    }
    write_json(EXPERIMENT_DIR / "final_status.json", status)
    binary = evaluation["binary"]
    report = f"""# R4-3B Locked Word-Level Sequence Validation

Final: `{evaluation['final_status']}`

- Contract/source verification: PASS
- Validation words/probes: {len(words):,} / {compute['validation_probes']:,}
- Binary Macro-F1: {binary['macro_f1']:.6f}
- Balanced accuracy: {binary['balanced_accuracy']:.6f}
- Deletion precision/recall/F1: {binary['deletion_precision']:.6f} / {binary['deletion_recall']:.6f} / {binary['deletion_f1']:.6f}
- Substitution false-deletion rate: {evaluation['substitution_false_deletion_rate']:.6f}
- Correct false-deletion rate: {evaluation['correct_false_deletion_rate']:.6f}
- Exact frozen matched control: {'EVALUATED' if matched_info['exact_matched_metrics_evaluable'] else 'NOT EVALUABLE: 135 identities are outside the frozen usable-word population; no intersection metric or resampling was used'}
- R4 TEST accessed: NO
- New neural training: NO

The result is the first and only validation run under the frozen R4-3B0 contract. No parameters or scorer code were changed after inference began.
"""
    (EXPERIMENT_DIR / "r4_3b_locked_report.md").write_text(report, encoding="utf-8")
    key_names = [
        "preflight.json", "scorer_identity.json", "validation_binary_metrics.json", "validation_3class_metrics.json",
        "matched_control_metrics.json", "validation_phone_predictions.csv", "validation_word_paths.jsonl",
        "validation_acoustic_evidence.npz", "final_status.json", "r4_3b_locked_report.md",
    ]
    write_json(EXPERIMENT_DIR / "artifact_hashes.json", {
        "algorithm": "SHA-256",
        "files": {name: sha256(EXPERIMENT_DIR / name) for name in key_names},
        "note": "artifact_hashes.json intentionally does not hash itself",
    })


def main() -> int:
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite locked experiment: {EXPERIMENT_DIR}")
    overall_started = time.perf_counter()
    contract, hashes = verify_and_load_contract()
    hashes["runner"] = sha256(Path(__file__).resolve())
    synthetic = run_synthetic_tests()
    audio_root = r3.require_audio_root()
    r3.set_seed(42)
    r43a.AUDIT_SPEAKERS = r43a.TRAIN_SPEAKERS | r43a.VALIDATION_SPEAKERS
    metadata_started = time.perf_counter()
    words, _ = r43a.build_word_records(audio_root)
    usable = [word for word in words if word["usable"]]
    split_words = {name: [word for word in usable if word["split"] == name] for name in ("train", "validation")}
    if {name: len(value) for name, value in split_words.items()} != EXPECTED_USABLE_WORDS:
        raise RuntimeError(f"Usable word count mismatch: { {name: len(value) for name, value in split_words.items()} }")
    train_counts = Counter(row["relation"] for word in split_words["train"] for row in word["clean_rows"])
    if train_counts != EXPECTED_TRAIN_COUNTS:
        raise RuntimeError(f"Frozen TRAIN operation counts differ: {dict(train_counts)}")
    scoreable_sources = {row["source_index"] for word in split_words["validation"] for row in word["clean_rows"]}
    matched, matched_info = verify_matched(load_v4_identity_rows(), scoreable_sources)
    source_to_word = {
        row["source_index"]: word
        for word in words if word["split"] == "validation"
        for row in word["clean_rows"]
    }
    missing_reason_flags: Counter[str] = Counter()
    missing_relations: Counter[str] = Counter()
    for source in matched - scoreable_sources:
        word = source_to_word.get(source)
        if word is None:
            missing_reason_flags["not_in_clean_candidate_word"] += 1
            continue
        missing_relations[next(row["relation"] for row in word["clean_rows"] if row["source_index"] == source)] += 1
        if word["has_addition"]:
            missing_reason_flags["addition_word"] += 1
        if word["has_unresolved"]:
            missing_reason_flags["unresolved_word"] += 1
        if not word["boundary_available"]:
            missing_reason_flags["boundary_unavailable"] += 1
    matched_info["excluded_reason_flags_nonexclusive"] = dict(missing_reason_flags)
    matched_info["excluded_relations"] = dict(missing_relations)
    metadata_seconds = time.perf_counter() - metadata_started
    print(json.dumps({"preflight": "PASS", "matched_control": matched_info}, indent=2), flush=True)

    log_probs, offsets, centers, compute = infer_validation(split_words["validation"], contract, audio_root)
    compute["train_operation_count_computation_seconds"] = metadata_seconds
    phone_rows, word_paths, dp_compute = score_words(split_words["validation"], log_probs, offsets, matched)
    if len(phone_rows) != sum(len(word["clean_rows"]) for word in split_words["validation"]):
        raise RuntimeError("Phone prediction row count mismatch")
    evaluation = evaluate(phone_rows, split_words["validation"], word_paths, matched, matched_info)
    total_seconds = time.perf_counter() - overall_started
    save_outputs(contract, hashes, synthetic, split_words["validation"], log_probs, offsets, centers,
                 phone_rows, word_paths, evaluation, compute, dp_compute, matched_info, total_seconds)
    print(json.dumps({
        "final_status": evaluation["final_status"], "binary": evaluation["binary"],
        "matched": evaluation["matched"], "gates": evaluation["gates"],
        "compute": {**compute, **dp_compute, "total_experiment_seconds": total_seconds},
        "r4_test_accessed": False, "training": False,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
