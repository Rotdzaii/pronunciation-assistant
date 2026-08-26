from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import r4_3b_sequence_dp as dp  # noqa: E402
import run_l2_arctic_observed_phone_r3_1a as r3  # noqa: E402
import run_r4_3a_word_sequence_design_audit as r43a  # noqa: E402


REPO_ROOT = r3.REPO_ROOT
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r4_3c_train_only_prior_calibration"
CHECKPOINT = REPO_ROOT / "ai-training/experiments/r3_1d_observed_phone_seed42_48epochs/R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt"
V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
CONTRACT_PATH = REPO_ROOT / "ai-training/experiments/r4_3b_frozen_numerical_contract/r4_3b_complete_numerical_contract.json"
SCORER_PATH = SCRIPTS_DIR / "r4_3b_sequence_dp.py"
EXPECTED_HASHES = {
    "checkpoint": "5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "contract": "2E65880C3E4AFD0EFBBB8014AAD8E1B04B051267470C17A3E733244176D18FB5",
    "scorer": "3DA0A8A238C5FDF05F368F010558263CC98E73FC49E4C4600BBEB68E221A86D1",
}
TRAIN_SPEAKERS = tuple(r3.TRAIN_SPEAKERS)
LAMBDA_GRID = tuple(round(index / 10, 2) for index in range(1, 11))
EXPECTED_WORDS = 16_259
EXPECTED_PROBES = 158_772
EXPECTED_COUNTS = Counter(correct=48_893, substitution=5_867, deletion=1_544)
ALPHA = 1.0
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def binary_metrics(truth: list[int], prediction: list[int]) -> dict[str, Any]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(truth, prediction):
        matrix[actual, predicted] += 1
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    non_precision = tn / (tn + fn) if tn + fn else 0.0
    non_recall = tn / (tn + fp) if tn + fp else 0.0
    non_f1 = 2 * non_precision * non_recall / (non_precision + non_recall) if non_precision + non_recall else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": len(truth), "non_deletion_support": tn + fp, "deletion_support": fn + tp,
        "accuracy": (tn + tp) / len(truth) if truth else 0.0,
        "balanced_accuracy": (non_recall + recall) / 2,
        "macro_f1": (non_f1 + f1) / 2,
        "deletion_precision": precision, "deletion_recall": recall, "deletion_f1": f1,
        "confusion_matrix": matrix.tolist(),
    }


def verify_sources() -> tuple[dict[str, str], dict[str, Any]]:
    paths = {"checkpoint": CHECKPOINT, "v4": V4_PATH, "contract": CONTRACT_PATH, "scorer": SCORER_PATH}
    actual = {name: sha256(path) for name, path in paths.items()}
    mismatch = {name: {"expected": EXPECTED_HASHES[name], "actual": value}
                for name, value in actual.items() if value != EXPECTED_HASHES[name]}
    if mismatch:
        raise RuntimeError(f"Source verification failed: {mismatch}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if tuple(contract["model_and_data"]["phone_vocabulary_index_order"]) != dp.PHONE_VOCAB:
        raise RuntimeError("Phone vocabulary mismatch")
    return actual, contract


def synthetic_tests() -> dict[str, Any]:
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
        raise RuntimeError(f"Expected 7 tests, got {len(tests)}")
    return {"status": "PASS", "passed": 7, "tests": tests}


def relation_counts(words: list[dict[str, Any]], speakers: set[str] | None = None) -> Counter[str]:
    return Counter(
        row["relation"] for word in words
        if speakers is None or word["speaker_id"] in speakers
        for row in word["clean_rows"]
    )


def priors_from_counts(counts: Counter[str]) -> dict[str, Any]:
    total = counts["correct"] + counts["substitution"] + counts["deletion"]
    denominator = total + 3
    values = {}
    for relation, operation in (("correct", "MATCH"), ("substitution", "SUBSTITUTION"), ("deletion", "DELETE_EXPECTED")):
        probability = (counts[relation] + 1) / denominator
        values[operation] = {"count": counts[relation], "probability": probability, "log_prior": math.log(probability)}
    return {"alpha": ALPHA, "N": total, "denominator": denominator, "values": values}


def build_loso_priors(words: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    all_speakers = set(TRAIN_SPEAKERS)
    for held_out in TRAIN_SPEAKERS:
        calibration = all_speakers - {held_out}
        held_counts = relation_counts(words, {held_out})
        fold_counts = relation_counts(words, calibration)
        output[held_out] = {
            "held_out_speaker": held_out, "calibration_speakers": sorted(calibration),
            "held_out_counts": dict(held_counts), "calibration_priors": priors_from_counts(fold_counts),
        }
    return output


def duration_bin(duration: float) -> int:
    return int(math.floor(duration / 0.010 + 1e-9))


def freeze_train_matched(words: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[int], dict[str, Any]]:
    strata: dict[tuple[str, str, int], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: {0: [], 1: []})
    total_deletions = 0
    for word in words:
        for row in word["clean_rows"]:
            target = int(row["relation"] == "deletion")
            total_deletions += target
            key = (row["speaker_id"], row["expected"], duration_bin(row["end"] - row["start"]))
            strata[key][target].append(row)
    rng = np.random.default_rng(SEED)
    selected: list[dict[str, Any]] = []
    identities: set[int] = set()
    pair_id = 0
    for key in sorted(strata):
        negatives = sorted(strata[key][0], key=lambda row: row["source_index"])
        positives = sorted(strata[key][1], key=lambda row: row["source_index"])
        count = min(len(negatives), len(positives))
        if count == 0:
            continue
        chosen_negative = sorted(rng.choice(len(negatives), size=count, replace=False).tolist())
        chosen_positive = sorted(rng.choice(len(positives), size=count, replace=False).tolist())
        for negative_index, positive_index in zip(chosen_negative, chosen_positive):
            pair_id += 1
            for role, row in (("non_deletion", negatives[negative_index]), ("deletion", positives[positive_index])):
                if row["source_index"] in identities:
                    raise RuntimeError("Duplicate TRAIN matched identity")
                identities.add(row["source_index"])
                selected.append({
                    "pair_id": pair_id, "role": role, "source_csv_row": row["source_index"] + 2,
                    "speaker_id": row["speaker_id"], "utterance_id": row["utterance_id"],
                    "start_time": row["start"], "end_time": row["end"],
                    "duration_bin_10ms": key[2], "expected_phone": row["expected"], "relation": row["relation"],
                })
    manifest = {
        "seed": SEED, "sampling": "numpy default_rng(42), sorted strata and source indexes, without replacement",
        "matching": "same speaker + expected canonical phone + floor(duration/0.010 + 1e-9)",
        "pairs": pair_id, "rows": len(selected), "deletion_rows": pair_id, "non_deletion_rows": pair_id,
        "phones": len({row["expected_phone"] for row in selected}),
        "speakers": len({row["speaker_id"] for row in selected}),
        "deletion_coverage": pair_id / total_deletions,
        "duration_used_by_scorer": False, "resampled_per_lambda": False,
    }
    return selected, identities, manifest


def save_matched_csv(rows: list[dict[str, Any]]) -> str:
    path = EXPERIMENT_DIR / "train_matched_control.csv"
    fields = ["pair_id", "role", "source_csv_row", "speaker_id", "utterance_id", "start_time", "end_time",
              "duration_bin_10ms", "expected_phone", "relation"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return sha256(path)


def infer_train(words: list[dict[str, Any]], contract: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    stride = float(contract["probe_grid"]["stride_seconds"])
    offsets = [0]
    centers: list[float] = []
    for word in words:
        centers.extend(dp.probe_centers(float(word["mfa_start"]), float(word["mfa_end"]), stride))
        offsets.append(len(centers))
    if len(centers) != EXPECTED_PROBES:
        raise RuntimeError(f"TRAIN probe count mismatch: {len(centers)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = r3.SmallPronunciationCNNAttention().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    extractor = r3.FixedLogMel().to(device).eval()
    store = r3.SequentialWaveStore()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started = time.perf_counter()
    waveform_buffer: list[np.ndarray] = []
    chunks: list[np.ndarray] = []

    def flush() -> None:
        if not waveform_buffer:
            return
        waveforms = torch.from_numpy(np.stack(waveform_buffer)).float().to(device)
        features = extractor(waveforms)
        for begin in range(0, len(features), r3.BATCH_SIZE):
            logits = model(features[begin:begin + r3.BATCH_SIZE])
            chunks.append(torch.log_softmax(logits, dim=1).cpu().numpy().astype(np.float32, copy=False))
        waveform_buffer.clear()

    with torch.inference_mode():
        for word_index, word in enumerate(words):
            audio = store.load(word["audio_path"])
            for center in centers[offsets[word_index]:offsets[word_index + 1]]:
                waveform_buffer.append(r3.centered_window(audio, center, center))
                if len(waveform_buffer) == r3.PREPROCESS_BATCH_SIZE:
                    flush()
            if (word_index + 1) % 2000 == 0:
                print(f"TRAIN acoustic evidence: {word_index + 1}/{len(words)} words", flush=True)
        flush()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    evidence = np.concatenate(chunks, axis=0)
    if evidence.shape != (len(centers), 40) or not np.isfinite(evidence).all():
        raise RuntimeError(f"Invalid TRAIN evidence: {evidence.shape}")
    return evidence, np.asarray(offsets, dtype=np.int64), np.asarray(centers, dtype=np.float64), {
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_epoch": int(checkpoint["epoch"]), "train_probes": len(centers),
        "median_probes_per_word": float(np.median(np.diff(np.asarray(offsets)))),
        "acoustic_inference_seconds": elapsed, "probes_per_second": len(centers) / elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    }


def evaluate_grid(words: list[dict[str, Any]], evidence: np.ndarray, offsets: np.ndarray,
                  fold_priors: dict[str, Any], matched_ids: set[int]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    flat_rows = [row for word in words for row in word["clean_rows"]]
    truth = [int(row["relation"] == "deletion") for row in flat_rows]
    source_order = [row["source_index"] for row in flat_rows]
    if len(source_order) != len(set(source_order)) or len(flat_rows) != sum(EXPECTED_COUNTS.values()):
        raise RuntimeError("TRAIN OOF identity invariant failed")
    speaker_indexes = {speaker: [i for i, row in enumerate(flat_rows) if row["speaker_id"] == speaker] for speaker in TRAIN_SPEAKERS}
    matched_indexes = [i for i, source in enumerate(source_order) if source in matched_ids]
    metrics_by_lambda: dict[str, Any] = {}
    speaker_metrics: dict[str, Any] = {}
    matched_metrics: dict[str, Any] = {}
    no_valid_paths: dict[str, int] = {}
    started = time.perf_counter()

    for lambda_value in LAMBDA_GRID:
        key = f"{lambda_value:.2f}"
        predictions: list[int] = []
        invalid = 0
        for word_index, word in enumerate(words):
            base = fold_priors[word["speaker_id"]]["calibration_priors"]["values"]
            dp.LOG_PRIORS = {
                "MATCH": lambda_value * float(base["MATCH"]["log_prior"]),
                "SUBSTITUTION": lambda_value * float(base["SUBSTITUTION"]["log_prior"]),
                "DELETE_EXPECTED": lambda_value * float(base["DELETE_EXPECTED"]["log_prior"]),
            }
            result = dp.align(word["expected"], np.asarray(evidence[offsets[word_index]:offsets[word_index + 1]], dtype=np.float64))
            if result["status"] != "OK":
                invalid += 1
                predictions.extend([0] * len(word["clean_rows"]))
                continue
            predictions.extend(int(step.operation == "DELETE_EXPECTED") for step in result["best"].steps)
        if len(predictions) != len(truth):
            raise RuntimeError(f"OOF prediction length mismatch at lambda {key}")
        metrics = binary_metrics(truth, predictions)
        correct_indexes = [i for i, row in enumerate(flat_rows) if row["relation"] == "correct"]
        substitution_indexes = [i for i, row in enumerate(flat_rows) if row["relation"] == "substitution"]
        metrics["correct_false_deletion_rate"] = float(np.mean([predictions[i] for i in correct_indexes]))
        metrics["substitution_false_deletion_rate"] = float(np.mean([predictions[i] for i in substitution_indexes]))
        metrics_by_lambda[key] = metrics
        matched_metrics[key] = binary_metrics([truth[i] for i in matched_indexes], [predictions[i] for i in matched_indexes])
        speaker_metrics[key] = {}
        for speaker, indexes in speaker_indexes.items():
            values = binary_metrics([truth[i] for i in indexes], [predictions[i] for i in indexes])
            substitution = [i for i in indexes if flat_rows[i]["relation"] == "substitution"]
            values["substitution_false_deletion_rate"] = float(np.mean([predictions[i] for i in substitution])) if substitution else None
            speaker_metrics[key][speaker] = values
        no_valid_paths[key] = invalid
        print(f"lambda={key} MF1={metrics['macro_f1']:.6f} recall={metrics['deletion_recall']:.6f} "
              f"subFD={metrics['substitution_false_deletion_rate']:.6f}", flush=True)
    return metrics_by_lambda, speaker_metrics, matched_metrics, {
        "grid_dp_seconds": time.perf_counter() - started, "no_valid_paths_by_lambda": no_valid_paths,
        "oof_rows_per_lambda": len(truth), "each_train_row_once_per_lambda": True,
    }


def select_lambda(metrics: dict[str, Any], speakers: dict[str, Any], matched: dict[str, Any]) -> dict[str, Any]:
    traces: dict[str, Any] = {}
    eligible: list[str] = []
    for key in (f"{value:.2f}" for value in LAMBDA_GRID):
        metric = metrics[key]
        speaker_pass = all(
            values["deletion_recall"] >= 0.25
            for values in speakers[key].values() if values["deletion_support"] >= 30
        )
        checks = {
            "deletion_recall_ge_0_45": metric["deletion_recall"] >= 0.45,
            "substitution_false_deletion_le_0_25": metric["substitution_false_deletion_rate"] <= 0.25,
            "all_supported_speakers_recall_ge_0_25": speaker_pass,
        }
        traces[key] = checks
        if all(checks.values()):
            eligible.append(key)
    selected = None
    if eligible:
        selected = min(eligible, key=lambda key: (
            -metrics[key]["macro_f1"], -metrics[key]["deletion_f1"], -metrics[key]["deletion_precision"],
            -matched[key]["macro_f1"], -float(key),
        ))
    minimum = None
    if selected is not None:
        minimum = {
            "oof_macro_f1_ge_0_65": metrics[selected]["macro_f1"] >= 0.65,
            "oof_deletion_f1_ge_0_35": metrics[selected]["deletion_f1"] >= 0.35,
            "oof_balanced_accuracy_ge_0_65": metrics[selected]["balanced_accuracy"] >= 0.65,
            "matched_macro_f1_ge_0_58": matched[selected]["macro_f1"] >= 0.58,
            "matched_deletion_f1_ge_0_50": matched[selected]["deletion_f1"] >= 0.50,
        }
    if selected is not None and all(minimum.values()):
        status = "R4_3C_PRIOR_SCALE_CANDIDATE_READY"
    elif selected is not None:
        status = "R4_3C_PRIOR_SCALE_SIGNAL_WEAK"
    elif any(metrics[key]["deletion_recall"] >= 0.45 and metrics[key]["substitution_false_deletion_rate"] > 0.25 for key in metrics):
        status = "R4_3C_PRIOR_SCALE_TRADEOFF_NOT_RESOLVED"
    else:
        status = "R4_3C_NO_ELIGIBLE_PRIOR_SCALE"
    return {
        "eligibility_by_lambda": traces, "eligible_lambdas": eligible, "selected_lambda": selected,
        "selection_order": ["highest OOF Binary Macro-F1", "higher deletion F1", "higher deletion precision",
                            "higher matched Macro-F1", "larger lambda"],
        "minimum_development_gates": minimum, "final_status": status,
    }


def write_selected_contract(base_contract: dict[str, Any], selection: dict[str, Any], metrics: dict[str, Any],
                            speakers: dict[str, Any], matched: dict[str, Any], fold_priors: dict[str, Any],
                            source_hashes: dict[str, str]) -> tuple[str, str]:
    selected = selection["selected_lambda"]
    full_priors = priors_from_counts(EXPECTED_COUNTS)
    payload = {
        "contract_id": "R4-3C TRAIN-only globally rescaled operation priors",
        "status": "FROZEN_BEFORE_EXTERNAL_VALIDATION",
        "selected_lambda": float(selected),
        "score_equations": {
            "MATCH": "lambda * log_prior_MATCH + mean_span_log_probability(expected)",
            "SUBSTITUTION": "lambda * log_prior_SUB + max_other_phone_mean_span_log_probability",
            "DELETE_EXPECTED": "lambda * log_prior_DEL",
        },
        "inherited_contract_sha256": EXPECTED_HASHES["contract"],
        "inherited_rules": "all R4-3B0 rules unchanged except one global lambda multiplying every operation log prior",
        "source_hashes": source_hashes, "selection_protocol": selection,
        "lambda_grid": list(LAMBDA_GRID), "oof_metrics_by_lambda": metrics,
        "oof_speaker_metrics": speakers, "matched_metrics": matched, "loso_fold_priors": fold_priors,
        "full_train_priors": full_priors,
        "future_validation_gates": {
            "binary_macro_f1_min": 0.70, "deletion_f1_min": 0.40, "deletion_recall_min": 0.45,
            "macro_f1_gain_over_duration_min": 0.03, "duration_baseline_macro_f1": 0.668146,
            "substitution_false_deletion_rate_max": 0.25,
            "speaker_rule": "speaker with >=30 deletions must have deletion recall >=0.25",
            "matched_control": "must be newly frozen from word-eligible external rows before validation; old R4-0 set is incompatible",
        },
        "validation_used_for_selection": False, "r4_test_accessed": False, "training_performed": False,
    }
    json_path = EXPERIMENT_DIR / "r4_3c_selected_sequence_contract.json"
    write_json(json_path, payload)
    md_path = EXPERIMENT_DIR / "r4_3c_selected_sequence_contract.md"
    md_path.write_text(
        f"# R4-3C Selected Sequence Contract\n\nSelected global prior weight: `{selected}`.\n\n"
        "Every R4-3B0 rule remains unchanged. The only change is multiplication of MATCH, SUBSTITUTION, and "
        "DELETE_EXPECTED natural-log priors by the same global lambda. Selection used 12-fold TRAIN-speaker LOSO only. "
        "No validation inference or metrics and no R4 TEST access occurred. The old R4-0 matched set must not be reused.\n",
        encoding="utf-8",
    )
    return sha256(json_path), sha256(md_path)


def main() -> int:
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite experiment: {EXPERIMENT_DIR}")
    overall_started = time.perf_counter()
    source_hashes, contract = verify_sources()
    tests = synthetic_tests()
    runner_hash = sha256(Path(__file__).resolve())
    r3.set_seed(SEED)
    audio_root = r3.require_audio_root()
    r43a.AUDIT_SPEAKERS = r43a.TRAIN_SPEAKERS
    metadata_started = time.perf_counter()
    words, _ = r43a.build_word_records(audio_root)
    usable = [word for word in words if word["usable"]]
    if len(usable) != EXPECTED_WORDS:
        raise RuntimeError(f"TRAIN usable word mismatch: {len(usable)}")
    counts = relation_counts(usable)
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"TRAIN relation mismatch: {dict(counts)}")
    fold_priors = build_loso_priors(usable)
    matched_rows, matched_ids, matched_manifest = freeze_train_matched(usable)
    metadata_seconds = time.perf_counter() - metadata_started

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
    matched_sha = save_matched_csv(matched_rows)
    matched_manifest["sha256"] = matched_sha
    write_json(EXPERIMENT_DIR / "preflight.json", {
        "source_verification": "PASS", "source_hashes": source_hashes, "runner_sha256": runner_hash,
        "synthetic_tests": tests, "train_only": True, "train_usable_words": len(usable),
        "train_relation_counts": dict(counts), "validation_paths_resolved": False,
        "validation_acoustic_inference": False, "validation_candidate_metrics": False,
        "previous_validation_used_for_selection": False, "r4_test_paths_resolved": False,
        "r4_test_audio_accessed": False, "r4_test_inference": False, "training_performed": False,
    })
    write_json(EXPERIMENT_DIR / "loso_fold_priors.json", fold_priors)
    write_json(EXPERIMENT_DIR / "lambda_grid.json", {
        "values": list(LAMBDA_GRID), "degree_of_freedom": "one global lambda applied to all three log priors",
        "selection": "pre-registered TRAIN OOF protocol", "grid_modified_after_results": False,
    })
    write_json(EXPERIMENT_DIR / "train_matched_control_manifest.json", matched_manifest)
    print(json.dumps({"preflight": "PASS", "train_words": len(usable), "counts": dict(counts),
                      "matched": matched_manifest}, indent=2), flush=True)

    evidence, offsets, centers, acoustic_compute = infer_train(usable, contract)
    np.savez_compressed(EXPERIMENT_DIR / "train_acoustic_evidence.npz", log_probabilities=evidence,
                        word_probe_offsets=offsets, probe_centers_seconds=centers,
                        word_ids=np.asarray([word["word_id"] for word in usable]))
    metrics, speaker_metrics, matched_metrics, grid_compute = evaluate_grid(
        usable, evidence, offsets, fold_priors, matched_ids
    )
    selection = select_lambda(metrics, speaker_metrics, matched_metrics)
    write_json(EXPERIMENT_DIR / "oof_metrics_by_lambda.json", metrics)
    write_json(EXPERIMENT_DIR / "oof_speaker_metrics.json", speaker_metrics)
    write_json(EXPERIMENT_DIR / "train_matched_metrics.json", matched_metrics)
    write_json(EXPERIMENT_DIR / "selection_result.json", selection)
    contract_hashes = None
    if selection["final_status"] == "R4_3C_PRIOR_SCALE_CANDIDATE_READY":
        contract_hashes = write_selected_contract(contract, selection, metrics, speaker_metrics, matched_metrics,
                                                   fold_priors, source_hashes)
    total_seconds = time.perf_counter() - overall_started
    compute = {
        "train_metadata_and_prior_seconds": metadata_seconds, **acoustic_compute, **grid_compute,
        "total_experiment_seconds": total_seconds,
    }
    write_json(EXPERIMENT_DIR / "compute_report.json", compute)
    final = {
        "final_status": selection["final_status"], "selected_lambda": selection["selected_lambda"],
        "eligible_lambdas": selection["eligible_lambdas"], "candidate_contract_hashes": contract_hashes,
        "validation_acoustic_inference": False, "validation_candidate_metrics": False,
        "previous_validation_used_for_selection": False, "r4_test_accessed": False,
        "training_performed": False, "post_hoc_grid_change": False,
    }
    write_json(EXPERIMENT_DIR / "final_status.json", final)
    report_lines = [
        "# R4-3C TRAIN-Only Prior-Weight Calibration", "", f"Final: `{selection['final_status']}`", "",
        f"TRAIN usable words/positions: {len(usable):,} / {sum(counts.values()):,}",
        f"TRAIN matched pairs/rows: {matched_manifest['pairs']:,} / {matched_manifest['rows']:,}",
        f"Eligible lambdas: {selection['eligible_lambdas']}", f"Selected lambda: {selection['selected_lambda']}", "",
        "No validation acoustic inference or candidate metrics were run. R4 TEST remained closed. No neural training occurred.",
    ]
    (EXPERIMENT_DIR / "r4_3c_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    key_names = [
        "preflight.json", "loso_fold_priors.json", "lambda_grid.json", "oof_metrics_by_lambda.json",
        "oof_speaker_metrics.json", "train_matched_control.csv", "train_matched_control_manifest.json",
        "train_matched_metrics.json", "selection_result.json", "compute_report.json", "final_status.json", "r4_3c_report.md",
    ]
    if contract_hashes is not None:
        key_names += ["r4_3c_selected_sequence_contract.json", "r4_3c_selected_sequence_contract.md"]
    write_json(EXPERIMENT_DIR / "artifact_hashes.json", {
        "algorithm": "SHA-256", "files": {name: sha256(EXPERIMENT_DIR / name) for name in key_names},
        "train_acoustic_evidence_sha256": sha256(EXPERIMENT_DIR / "train_acoustic_evidence.npz"),
        "note": "artifact_hashes.json intentionally does not hash itself",
    })
    print(json.dumps({"final": final, "selection": selection, "metrics": metrics,
                      "matched_metrics": matched_metrics, "compute": compute}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
