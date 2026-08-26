from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
SCRIPTS_DIR = REPO_ROOT / "ai-training" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_l2_arctic_observed_phone_r3_1a as r3  # noqa: E402
import run_r4_2c_mfa_missing_phone_anchor_audit as r42c  # noqa: E402
import run_r4_3a_word_sequence_design_audit as r43a  # noqa: E402
import run_r4_4b_ctc_sequence as r4b  # noqa: E402
import run_r4_4c2_bigru_ctc_sequence as r4c2  # noqa: E402


TRAIN = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
VALIDATION = ("ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI")
TEST = ("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK")
TRAIN_SET, VALIDATION_SET, TEST_SET = map(frozenset, (TRAIN, VALIDATION, TEST))
PHONE_VOCAB = tuple(r3.PHONE_VOCAB)
PHONE_TO_ID = dict(r3.PHONE_TO_ID)
BLANK = 40
HYPOTHESIS_BATCH_SIZE = 4096

V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
CHECKPOINT_PATH = REPO_ROOT / (
    "ai-training/experiments/r4_4c2_bigru_ctc_seed42/"
    "R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt"
)
CONTRACT_PATH = EXPERIMENT_DIR / "R5_1_DEVELOPMENT_CONTRACT.json"
PREREG_PATH = EXPERIMENT_DIR / "R5_1_PREREGISTRATION.md"
CONTRACT_MANIFEST_PATH = EXPERIMENT_DIR / "artifact_hashes.json"
R5_0_DIR = REPO_ROOT / "ai-training/experiments/r5_0_addition_feasibility_audit"

EXPECTED_SHA = {
    "contract": "18697A0C7DED130E122F49A9440AFA2D79B02E8630F2F7583C775F67797DCE77",
    "preregistration": "3F69286BC25DA1AFF0A8C91028DC8C7C54D222F65825773F334EE737ABE2A633",
    "contract_manifest": "EE6026D7A863A0354B51D79E1BBD9CE82099EA26B2A38A6A4047F646FC864F21",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "checkpoint": "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085",
    "r5_0_preregistration": "14CBFADAC0BE35D53C01DC966A030A71F24EC4D86EA88384BC449544CE12AEF7",
    "r5_0_report": "DF484A09333A2F53A05E98E1C66B3C330CBA35F029B04B0BBB1623A3A9BAA3BA",
    "r5_0_final_status": "29F06679FD24F8A2BBBED25B9B57A693D15046444A97BF5D951D575E5850D0B6",
    "r5_0_manifest": "3A6F62C5BD354E5FD21149135ECC9FB4CD673947A0E5E213A513B4F2005339E8",
}
IDENTITY_PATHS = {
    "contract": CONTRACT_PATH,
    "preregistration": PREREG_PATH,
    "contract_manifest": CONTRACT_MANIFEST_PATH,
    "v4": V4_PATH,
    "checkpoint": CHECKPOINT_PATH,
    "r5_0_preregistration": R5_0_DIR / "R5_0_PREREGISTRATION.json",
    "r5_0_report": R5_0_DIR / "R5_0_ADDITION_FEASIBILITY_REPORT.md",
    "r5_0_final_status": R5_0_DIR / "r5_0_final_status.json",
    "r5_0_manifest": R5_0_DIR / "artifact_hashes.json",
}

EXPECTED_WORDS = 16582
EXPECTED_POSITIVE_WORDS = 323
EXPECTED_NEGATIVE_WORDS = 16259
EXPECTED_SOURCE_EVENTS = 423
EXPECTED_RUNTIME_EVENTS = 342
GATE_LIMITS = {
    "G1": ("addition_vs_all_nonaddition_roc_auc", ">=", 0.70),
    "G2": ("addition_vs_correct_only_roc_auc", ">=", 0.70),
    "G3": ("oof_binary_macro_f1", ">", 0.548179),
    "G4": ("oof_addition_f1", ">", 0.129246),
    "G5": ("correct_only_false_addition_rate", "<=", 0.054352),
    "G6": ("exact_event_f1", ">", 0.026688),
}

OUTPUT_NAMES = (
    "r5_1_row_accounting.json",
    "r5_1_train_scores.jsonl",
    "r5_1_loso_fold_thresholds.json",
    "r5_1_oof_predictions.jsonl",
    "r5_1_continuous_metrics.json",
    "r5_1_binary_metrics.json",
    "r5_1_event_metrics.json",
    "r5_1_score_distribution_audit.json",
    "r5_1_gate_results.json",
    "r5_1_execution_protocol_audit.json",
    "r5_1_implementation_identity.json",
    "r5_1_compute_report.json",
    "r5_1_final_status.json",
    "R5_1_DEVELOPMENT_RESULT.md",
    "R5_1_EXECUTION_ARTIFACT_MANIFEST.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2.0 * precision * recall, precision + recall)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": precision, "recall": recall, "f1": f1}


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = truth.astype(bool)
    predicted = predicted.astype(bool)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum((~truth) & predicted))
    fn = int(np.sum(truth & (~predicted)))
    tn = int(np.sum((~truth) & (~predicted)))
    positive = prf(tp, fp, fn)
    negative = prf(tn, fn, fp)
    return {
        "words": int(truth.size), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": ratio(tp + tn, truth.size),
        "balanced_accuracy": (float(positive["recall"]) + float(negative["recall"])) / 2.0,
        "binary_macro_f1": (float(positive["f1"]) + float(negative["f1"])) / 2.0,
        "addition_precision": positive["precision"], "addition_recall": positive["recall"],
        "addition_f1": positive["f1"], "nonaddition_f1": negative["f1"],
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size), "mean": float(array.mean()), "std": float(array.std(ddof=0)),
        "min": float(array.min()), "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)), "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def verify_identities() -> dict[str, Any]:
    actual = {name: sha256(path) for name, path in IDENTITY_PATHS.items()}
    mismatches = {
        name: {"expected": EXPECTED_SHA[name], "actual": actual[name], "path": str(IDENTITY_PATHS[name])}
        for name in EXPECTED_SHA if actual[name] != EXPECTED_SHA[name]
    }
    if mismatches:
        contract_keys = {"contract", "preregistration", "contract_manifest"}
        status = (
            "R5_1_EXECUTION_BLOCKED_CONTRACT_IDENTITY"
            if set(mismatches) & contract_keys else "R5_1_EXECUTION_BLOCKED_SOURCE_IDENTITY"
        )
        raise RuntimeError(f"{status}: {mismatches}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if tuple(contract["speaker_splits"]["TRAIN"]) != TRAIN:
        raise RuntimeError("R5_1_EXECUTION_BLOCKED_CONTRACT_IDENTITY: TRAIN speakers differ")
    if tuple(contract["speaker_splits"]["VALIDATION"]) != VALIDATION or tuple(contract["speaker_splits"]["TEST"]) != TEST:
        raise RuntimeError("R5_1_EXECUTION_BLOCKED_CONTRACT_IDENTITY: held-out speakers differ")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if int(checkpoint["blank_index"]) != BLANK or tuple(checkpoint["vocabulary"]) != PHONE_VOCAB:
        raise RuntimeError("R5_1_EXECUTION_BLOCKED_MODEL_CONTRACT: checkpoint vocabulary/blank mismatch")
    return {
        "status": "PASS", "expected_sha256": EXPECTED_SHA, "actual_sha256": actual,
        "paths": {name: str(path) for name, path in IDENTITY_PATHS.items()},
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_per": float(checkpoint["validation_per"]),
    }


def scan_train_source() -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    detail_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    train_relations: Counter[str] = Counter()
    with V4_PATH.open(encoding="utf-8", newline="") as handle:
        for source_index, source in enumerate(csv.DictReader(handle)):
            speaker = source["speaker_id"]
            if speaker not in TRAIN_SET:
                # Deliberately inspect no held-out field beyond the speaker guard.
                continue
            relation = source["relation"]
            train_relations[relation] += 1
            detail_rows[(speaker, source["utterance_id"])].append({
                "source_index": source_index, "speaker_id": speaker, "utterance_id": source["utterance_id"],
                "start": float(source["start_time"]), "end": float(source["end_time"]),
                "relation": relation, "expected": source["expected_phone_canonical"],
                "observed": source["observed_phone_canonical"], "label_quality": source["label_quality"],
            })
    return detail_rows, {
        "train_relation_counts": dict(train_relations),
        "clean_train_addition_events": int(train_relations["addition"]),
        "heldout_fields_inspected": ["speaker_id"],
        "validation_paths_resolved": False, "test_paths_resolved": False,
    }


def map_train_additions(
    detail_rows: dict[tuple[str, str], list[dict[str, Any]]], audio_root: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    events_by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mapped_multiplicity: Counter[int] = Counter()
    mapping: Counter[str] = Counter()
    for (speaker, utterance), rows in sorted(detail_rows.items()):
        if not any(row["relation"] == "addition" for row in rows):
            continue
        manual = r42c.parse_textgrid(audio_root / speaker / "annotation" / f"{utterance}.TextGrid")
        rows_by_word: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            word_index = r42c.containing_word(manual["words"], row["start"], row["end"])
            if word_index is not None:
                rows_by_word[word_index].append(row)
            elif row["relation"] == "addition":
                mapping["clean_addition_without_manual_word"] += 1
        for word_index, word_rows in sorted(rows_by_word.items()):
            ordered = sorted(word_rows, key=lambda item: item["source_index"])
            expected_rows = [
                row for row in ordered
                if row["relation"] in {"correct", "substitution", "deletion"}
                and row["expected"] in PHONE_TO_ID and row["label_quality"] == "clean"
            ]
            additions = [
                row for row in ordered
                if row["relation"] == "addition" and row["observed"] in PHONE_TO_ID
                and row["label_quality"] == "clean"
            ]
            if not additions:
                continue
            mapped_multiplicity[len(additions)] += 1
            for addition in additions:
                if not expected_rows:
                    mapping["clean_addition_without_expected_sequence"] += 1
                    continue
                boundary = sum(row["source_index"] < addition["source_index"] for row in expected_rows)
                n = len(expected_rows)
                position = "BEFORE_FIRST" if boundary == 0 else ("AFTER_FINAL" if boundary == n else "BETWEEN")
                word_id = f"{speaker}/{utterance}/{word_index}"
                events_by_word[word_id].append({
                    "word_id": word_id, "source_index": int(addition["source_index"]),
                    "phone": addition["observed"], "phone_index": PHONE_TO_ID[addition["observed"]],
                    "boundary": int(boundary), "position": position,
                })
                mapping["position_mapped"] += 1
    return events_by_word, {
        "mapping": dict(mapping),
        "mapped_word_multiplicity": {str(k): int(v) for k, v in sorted(mapped_multiplicity.items())},
        "mapped_words": len(events_by_word), "mapped_events": sum(map(len, events_by_word.values())),
    }


def build_train_words(audio_root: Path, events_by_word: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r43a.AUDIT_SPEAKERS = TRAIN_SET
    records, reconstruction = r43a.build_word_records(audio_root)
    words = [
        word for word in records
        if word["split"] == "train" and bool(word["expected"])
        and not word["has_unresolved"] and word["boundary_available"]
    ]
    for word in words:
        word["expected_ids"] = [PHONE_TO_ID[phone] for phone in word["expected"]]
        word["true_addition_events"] = list(events_by_word.get(word["word_id"], []))
        word["is_addition"] = bool(word["true_addition_events"])
        word["correct_only"] = (
            not word["is_addition"] and int(word["substitution"]) == 0 and int(word["deletion"]) == 0
        )
        word["substitution_negative"] = not word["is_addition"] and int(word["substitution"]) > 0
        word["deletion_negative"] = not word["is_addition"] and int(word["deletion"]) > 0
    positives = [word for word in words if word["is_addition"]]
    source_addition_words = int(reconstruction["counts"]["addition_words_excluded"])
    runtime_events = sum(len(word["true_addition_events"]) for word in positives)
    accounting = {
        "status": "PASS",
        "clean_train_addition_events_source": EXPECTED_SOURCE_EVENTS,
        "clean_addition_containing_words_frozen_source_reconstruction": source_addition_words,
        "runtime_evaluable_words": len(words),
        "runtime_evaluable_addition_words": len(positives),
        "runtime_evaluable_negative_words": len(words) - len(positives),
        "excluded_non_runtime_evaluable_positive_words": source_addition_words - len(positives),
        "runtime_addition_events": runtime_events,
        "source_events_not_in_runtime_population": EXPECTED_SOURCE_EVENTS - runtime_events,
        "multiple_addition_positive_words": sum(len(word["true_addition_events"]) > 1 for word in positives),
        "events_in_multiple_addition_words": sum(
            len(word["true_addition_events"]) for word in positives if len(word["true_addition_events"]) > 1
        ),
        "mixed_substitution_addition_words": sum(int(word["substitution"]) > 0 for word in positives),
        "mixed_deletion_addition_words": sum(int(word["deletion"]) > 0 for word in positives),
        "mixed_substitution_and_deletion_addition_words": sum(
            int(word["substitution"]) > 0 and int(word["deletion"]) > 0 for word in positives
        ),
        "other_mixed_error_positive_words": sum(
            int(word["substitution"]) == 0 and int(word["deletion"]) == 0 and int(word["correct"]) == 0
            for word in positives
        ),
        "runtime_positive_event_multiplicity": dict(Counter(len(word["true_addition_events"]) for word in positives)),
        "reconstruction": reconstruction,
    }
    expected = (EXPECTED_WORDS, EXPECTED_POSITIVE_WORDS, EXPECTED_NEGATIVE_WORDS, EXPECTED_RUNTIME_EVENTS)
    actual = (len(words), len(positives), len(words) - len(positives), runtime_events)
    if actual != expected:
        accounting["status"] = "R5_1_EXECUTION_BLOCKED_ROW_ACCOUNTING"
        accounting["expected_tuple"] = expected
        accounting["actual_tuple"] = actual
    return words, accounting


def ctc_minimum(target: list[int]) -> int:
    return len(target) + sum(left == right for left, right in zip(target, target[1:]))


def audit_candidate_alignability(words: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    unalignable: list[dict[str, Any]] = []
    for word in words:
        samples = int(round(float(word["mfa_end"]) * r4b.SAMPLE_RATE)) - int(round(float(word["mfa_start"]) * r4b.SAMPLE_RATE))
        steps = r4b.encoder_steps(r4b.feature_frames(samples))
        expected = word["expected_ids"]
        if ctc_minimum(expected) > steps:
            unalignable.append({"word_id": word["word_id"], "kind": "KEEP"})
        total += 1
        for boundary in range(len(expected) + 1):
            for phone in range(40):
                target = expected[:boundary] + [phone] + expected[boundary:]
                total += 1
                if ctc_minimum(target) > steps:
                    unalignable.append({
                        "word_id": word["word_id"], "kind": "INSERT", "boundary": boundary,
                        "phone_index": phone, "minimum_steps": ctc_minimum(target), "encoder_steps": steps,
                    })
    return {
        "hypotheses": total, "insert_hypotheses": total - len(words),
        "unalignable": len(unalignable), "examples": unalignable[:20],
        "status": "PASS" if not unalignable else "FAIL",
    }


def infer_train(
    words: list[dict[str, Any]], device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    overall_started = time.perf_counter()
    features, feature_report = r4b.materialize_features(words, device, "r5_1_train_only")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = r4c2.WordBiGRUCTCModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log_probs: list[torch.Tensor | None] = [None] * len(words)
    started = time.perf_counter()
    lengths = [feature.shape[-1] for feature in features]
    ordered = sorted(range(len(words)), key=lambda index: (lengths[index], index))
    with torch.no_grad():
        for start in range(0, len(ordered), 8):
            indexes = ordered[start:start + 8]
            maximum = max(lengths[index] for index in indexes)
            batch = torch.zeros((len(indexes), 1, r4b.N_MELS, maximum), dtype=torch.float32)
            frame_lengths: list[int] = []
            for position, index in enumerate(indexes):
                feature = features[index]
                batch[position, 0, :, :feature.shape[-1]] = feature
                frame_lengths.append(feature.shape[-1])
            frame_tensor = torch.tensor(frame_lengths, dtype=torch.long, device=device)
            logits, output_lengths = model(batch.to(device, non_blocking=True), frame_tensor)
            values = torch.log_softmax(logits, dim=-1).detach().cpu()
            for position, index in enumerate(indexes):
                steps = int(output_lengths[position].item())
                log_probs[index] = values[position, :steps].contiguous()
            if start == 0 or start + len(indexes) == len(ordered) or (start // 8) % 250 == 0:
                print(f"r5_1_train_inference={min(start + len(indexes), len(ordered))}/{len(ordered)}", flush=True)
    if any(value is None for value in log_probs):
        raise RuntimeError("R5_1_EXECUTION_BLOCKED_MODEL_CONTRACT: incomplete TRAIN inference")
    inference_seconds = time.perf_counter() - started
    return [value for value in log_probs if value is not None], features, {
        "feature_report": feature_report,
        "feature_seconds": float(feature_report["seconds"]),
        "inference_seconds": inference_seconds,
        "total_feature_and_inference_seconds": time.perf_counter() - overall_started,
    }


def score_train(
    words: list[dict[str, Any]], log_probs: list[torch.Tensor], device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    keep = np.full(len(words), -np.inf, dtype=np.float64)
    best = np.full(len(words), -np.inf, dtype=np.float64)
    best_boundary = np.full(len(words), -1, dtype=np.int16)
    best_phone = np.full(len(words), -1, dtype=np.int16)
    equivalence_count = np.zeros(len(words), dtype=np.int32)
    criterion = torch.nn.CTCLoss(blank=BLANK, reduction="none", zero_infinity=True)
    order = sorted(range(len(words)), key=lambda index: (log_probs[index].shape[0], index))
    processed = 0
    nonfinite = 0
    unalignable = 0
    started = time.perf_counter()

    def process(batch_items: list[tuple[int, str, int, int, list[int]]]) -> None:
        nonlocal processed, nonfinite, unalignable
        maximum = max(log_probs[item[0]].shape[0] for item in batch_items)
        acoustic = torch.zeros((maximum, len(batch_items), 41), dtype=torch.float32)
        input_lengths: list[int] = []
        target_lengths: list[int] = []
        flat_targets: list[int] = []
        for column, (word_index, _kind, _boundary, _phone, target) in enumerate(batch_items):
            evidence = log_probs[word_index]
            acoustic[:evidence.shape[0], column] = evidence
            input_lengths.append(evidence.shape[0])
            target_lengths.append(len(target))
            flat_targets.extend(target)
            unalignable += int(ctc_minimum(target) > evidence.shape[0])
        with torch.no_grad():
            nll = criterion(
                acoustic.to(device),
                torch.tensor(flat_targets, dtype=torch.long, device=device),
                torch.tensor(input_lengths, dtype=torch.long, device=device),
                torch.tensor(target_lengths, dtype=torch.long, device=device),
            ).detach().cpu().numpy()
        raw = -nll.astype(np.float64)
        target_scores = raw / np.maximum(np.asarray(target_lengths, dtype=np.float64), 1.0)
        nonfinite += int(np.sum(~np.isfinite(target_scores)))
        for item, score in zip(batch_items, target_scores):
            word_index, kind, boundary, phone, _target = item
            if kind == "KEEP":
                keep[word_index] = score
            elif score > best[word_index]:
                best[word_index] = score
                best_boundary[word_index] = boundary
                best_phone[word_index] = phone
                equivalence_count[word_index] = 1
            elif score == best[word_index]:
                equivalence_count[word_index] += 1
                if (boundary, phone) < (int(best_boundary[word_index]), int(best_phone[word_index])):
                    best_boundary[word_index] = boundary
                    best_phone[word_index] = phone
        processed += len(batch_items)
        if processed % (HYPOTHESIS_BATCH_SIZE * 25) < len(batch_items):
            print(f"r5_1_hypotheses_scored={processed}", flush=True)

    batch: list[tuple[int, str, int, int, list[int]]] = []
    for word_index in order:
        expected = words[word_index]["expected_ids"]
        group: list[tuple[int, str, int, int, list[int]]] = [(word_index, "KEEP", -1, -1, expected)]
        for boundary in range(len(expected) + 1):
            for phone in range(40):
                group.append((word_index, "INSERT", boundary, phone, expected[:boundary] + [phone] + expected[boundary:]))
        if batch and len(batch) + len(group) > HYPOTHESIS_BATCH_SIZE:
            process(batch)
            batch = []
        batch.extend(group)
    if batch:
        process(batch)
    expected_count = len(words) + sum((len(word["expected_ids"]) + 1) * 40 for word in words)
    if processed != expected_count:
        raise RuntimeError(f"Hypothesis count mismatch: {processed} != {expected_count}")
    if unalignable or nonfinite or not (
        np.isfinite(keep).all() and np.isfinite(best).all() and (best_boundary >= 0).all() and (best_phone >= 0).all()
    ):
        raise RuntimeError(
            f"R5_1 technical scoring invariant failed: unalignable={unalignable} nonfinite={nonfinite}"
        )
    addition_score = best - keep
    return keep, best, addition_score, best_boundary, best_phone, {
        "total_hypotheses": processed,
        "keep_hypotheses": len(words), "insert_hypotheses": processed - len(words),
        "average_hypotheses_per_word": processed / len(words),
        "batch_size": HYPOTHESIS_BATCH_SIZE, "seconds": time.perf_counter() - started,
        "unalignable_hypotheses": unalignable, "nonfinite_scores": nonfinite,
        "words_with_tied_best_sequence_score": int(np.sum(equivalence_count > 1)),
        "best_score_equivalence_count": distribution(equivalence_count.astype(np.float64)),
    }


def threshold_metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float, float]:
    positive = prf(tp, fp, fn)
    negative = prf(tn, fn, fp)
    return (
        (float(positive["f1"]) + float(negative["f1"])) / 2.0,
        float(positive["f1"]),
        float(positive["precision"]),
    )


def select_threshold(
    scores: np.ndarray, truth: np.ndarray, correct_only: np.ndarray
) -> tuple[float, dict[str, Any]]:
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    correct_only = np.asarray(correct_only, dtype=bool)
    unique = np.unique(scores)
    if not np.isfinite(unique).all() or unique.size == 0:
        raise RuntimeError("Non-finite/empty threshold calibration scores")
    groups: dict[float, np.ndarray] = {float(value): np.flatnonzero(scores == value) for value in unique}
    tp, fp, fn, tn = int(truth.sum()), int((~truth).sum()), 0, 0
    correct_positive = int(correct_only.sum())
    correct_support = int(correct_only.sum())
    best_key: tuple[float, float, float, float] | None = None
    best_threshold = 0.0
    best_trace: dict[str, Any] = {}

    def consider(threshold: float) -> None:
        nonlocal best_key, best_threshold, best_trace
        macro, addition_f1, addition_precision = threshold_metrics_from_counts(tp, fp, fn, tn)
        correct_far = ratio(correct_positive, correct_support)
        key = (macro, addition_f1, -correct_far, float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_trace = {
                "calibration_binary_macro_f1": macro, "calibration_addition_f1": addition_f1,
                "calibration_addition_precision": addition_precision,
                "calibration_correct_only_false_addition_rate": correct_far,
                "calibration_confusion": [[tn, fp], [fn, tp]],
            }

    lower = float(np.nextafter(unique[0], -np.inf))
    upper = float(np.nextafter(unique[-1], np.inf))
    consider(lower)
    for value in unique:
        scalar = float(value)
        consider(scalar)
        indexes = groups[scalar]
        positive_removed = int(truth[indexes].sum())
        negative_removed = int(indexes.size - positive_removed)
        tp -= positive_removed
        fn += positive_removed
        fp -= negative_removed
        tn += negative_removed
        correct_positive -= int(correct_only[indexes].sum())
    consider(upper)
    return best_threshold, {
        "candidate_count": int(unique.size + 2), "unique_score_count": int(unique.size),
        "lower_edge": lower, "upper_edge": upper,
        "selection_precedence": [
            "higher_binary_macro_f1", "higher_addition_f1",
            "lower_correct_only_false_addition_rate", "higher_threshold",
        ],
        "selected_threshold": best_threshold, "selected_metrics": best_trace,
    }


def run_loso(
    words: list[dict[str, Any]], scores: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    truth = np.asarray([word["is_addition"] for word in words], dtype=bool)
    correct_only = np.asarray([word["correct_only"] for word in words], dtype=bool)
    speakers = np.asarray([word["speaker_id"] for word in words])
    predictions = np.zeros(len(words), dtype=bool)
    fold_records: list[dict[str, Any]] = []
    for heldout in TRAIN:
        calibration = speakers != heldout
        evaluation = speakers == heldout
        threshold, trace = select_threshold(scores[calibration], truth[calibration], correct_only[calibration])
        predictions[evaluation] = scores[evaluation] >= np.float64(threshold)
        fold_metric = binary_metrics(truth[evaluation], predictions[evaluation])
        fold_records.append({
            "heldout_speaker": heldout,
            "calibration_speakers": [speaker for speaker in TRAIN if speaker != heldout],
            "calibration_words": int(calibration.sum()), "heldout_words": int(evaluation.sum()),
            **trace, "heldout_metrics": fold_metric,
            "heldout_correct_only_false_addition_rate": ratio(
                int(np.sum(predictions & evaluation & correct_only)), int(np.sum(evaluation & correct_only))
            ),
        })
    if any(np.sum(speakers == speaker) == 0 for speaker in TRAIN):
        raise RuntimeError("Missing TRAIN speaker in LOSO")
    return predictions, fold_records


def false_addition_rate(words: list[dict[str, Any]], predicted: np.ndarray, key: str) -> dict[str, Any]:
    mask = np.asarray([bool(word[key]) for word in words], dtype=bool)
    return {
        "cohort": key, "support": int(mask.sum()), "predicted_addition": int(np.sum(mask & predicted)),
        "false_addition_rate": ratio(int(np.sum(mask & predicted)), int(mask.sum())),
    }


def event_evaluation(
    words: list[dict[str, Any]], predicted: np.ndarray, best_boundary: np.ndarray, best_phone: np.ndarray
) -> dict[str, Any]:
    true_counter: Counter[tuple[str, int, int]] = Counter()
    pred_counter: Counter[tuple[str, int, int]] = Counter()
    true_by_position: dict[str, Counter[tuple[str, int, int]]] = {
        name: Counter() for name in ("BEFORE_FIRST", "BETWEEN", "AFTER_FINAL")
    }
    pred_by_position: dict[str, Counter[tuple[str, int, int]]] = {
        name: Counter() for name in ("BEFORE_FIRST", "BETWEEN", "AFTER_FINAL")
    }
    for index, word in enumerate(words):
        for event in word["true_addition_events"]:
            identity = (word["word_id"], int(event["boundary"]), int(event["phone_index"]))
            true_counter[identity] += 1
            true_by_position[event["position"]][identity] += 1
        if predicted[index]:
            boundary = int(best_boundary[index])
            phone = int(best_phone[index])
            identity = (word["word_id"], boundary, phone)
            pred_counter[identity] += 1
            n = len(word["expected_ids"])
            position = "BEFORE_FIRST" if boundary == 0 else ("AFTER_FINAL" if boundary == n else "BETWEEN")
            pred_by_position[position][identity] += 1

    def counter_metric(true_values: Counter[Any], pred_values: Counter[Any]) -> dict[str, Any]:
        tp = int(sum((true_values & pred_values).values()))
        result = prf(tp, int(sum(pred_values.values())) - tp, int(sum(true_values.values())) - tp)
        return {
            **result, "true_events": int(sum(true_values.values())),
            "predicted_events": int(sum(pred_values.values())),
        }

    return {
        "exact_event": counter_metric(true_counter, pred_counter),
        "by_position": {
            name: counter_metric(true_by_position[name], pred_by_position[name]) for name in true_by_position
        },
        "matching": "deterministic Counter intersection on (word_id,boundary,phone_index)",
        "one_prediction_maximum_per_positive_word": True,
        "multiple_addition_limitation": (
            "All multiple-addition words are retained, but one BEST_INSERT candidate permits at most one "
            "recovered event per word; additional true events are necessarily false negatives."
        ),
    }


def score_distribution_audit(words: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    by_speaker = {
        speaker: distribution(scores[[word["speaker_id"] == speaker for word in words]]) for speaker in TRAIN
    }
    masks = {
        "true_addition": np.asarray([word["is_addition"] for word in words], dtype=bool),
        "correct_only": np.asarray([word["correct_only"] for word in words], dtype=bool),
        "substitution_containing_negative": np.asarray([word["substitution_negative"] for word in words], dtype=bool),
        "deletion_containing_negative": np.asarray([word["deletion_negative"] for word in words], dtype=bool),
    }
    expected_lengths = np.asarray([len(word["expected_ids"]) for word in words], dtype=np.float64)
    durations = np.asarray([float(word["mfa_end"]) - float(word["mfa_start"]) for word in words], dtype=np.float64)
    length_values = sorted({int(value) for value in expected_lengths})
    duration_edges = [0.0, 0.15, 0.25, 0.4, 0.6, float("inf")]
    duration_names = ["<150ms", "150-250ms", "250-400ms", "400-600ms", ">=600ms"]
    return {
        "by_speaker": by_speaker,
        "by_cohort": {name: distribution(scores[mask]) for name, mask in masks.items()},
        "expected_length": {
            str(length): distribution(scores[expected_lengths == length]) for length in length_values
        },
        "duration": {
            name: distribution(scores[(durations >= low) & (durations < high)])
            for name, low, high in zip(duration_names, duration_edges[:-1], duration_edges[1:])
        },
        "relationships": {
            "score_expected_length_pearson_r": float(np.corrcoef(scores, expected_lengths)[0, 1]),
            "score_audio_duration_pearson_r": float(np.corrcoef(scores, durations)[0, 1]),
        },
        "descriptive_only_no_formula_change": True,
    }


def build_report(
    identity: dict[str, Any], accounting: dict[str, Any], continuous: dict[str, Any], binary: dict[str, Any],
    folds: list[dict[str, Any]], events: dict[str, Any], gates: dict[str, Any], robust: dict[str, Any],
    protocol: dict[str, Any], status: str
) -> str:
    lines = [
        "# R5-1 Frozen TRAIN Development Result", "",
        f"Final status: `{status}`", "",
        "## Identity", "",
        f"- Contract SHA: `{identity['actual_sha256']['contract']}`",
        f"- Preregistration SHA: `{identity['actual_sha256']['preregistration']}`",
        f"- V4 SHA: `{identity['actual_sha256']['v4']}`",
        f"- Checkpoint SHA: `{identity['actual_sha256']['checkpoint']}`", "",
        "## Row accounting", "",
        f"- Runtime-evaluable words: {accounting['runtime_evaluable_words']:,}",
        f"- Positive / negative words: {accounting['runtime_evaluable_addition_words']:,} / {accounting['runtime_evaluable_negative_words']:,}",
        f"- Source / runtime clean addition events: {accounting['clean_train_addition_events_source']:,} / {accounting['runtime_addition_events']:,}",
        f"- Multiple-addition positive words: {accounting['multiple_addition_positive_words']:,}", "",
        "## Continuous discrimination", "",
        f"- Addition vs all non-addition ROC-AUC: {continuous['addition_vs_all_nonaddition']['roc_auc']:.6f}",
        f"- Addition vs correct-only ROC-AUC: {continuous['addition_vs_correct_only']['roc_auc']:.6f}", "",
        "## Speaker-LOSO binary decision", "",
        f"- Thresholds: {', '.join(format(item['selected_threshold'], '.17g') for item in folds)}",
        f"- Confusion (TP/FP/FN/TN): {binary['TP']}/{binary['FP']}/{binary['FN']}/{binary['TN']}",
        f"- Binary Macro-F1: {binary['binary_macro_f1']:.6f}",
        f"- Addition P/R/F1: {binary['addition_precision']:.6f} / {binary['addition_recall']:.6f} / {binary['addition_f1']:.6f}",
        f"- Correct-only false-addition rate: {binary['false_addition_rates']['correct_only']['false_addition_rate']:.6f}", "",
        "## Exact event localization", "",
        f"- Event P/R/F1: {events['exact_event']['precision']:.6f} / {events['exact_event']['recall']:.6f} / {events['exact_event']['f1']:.6f}",
        "- Multiple-addition words are retained, but the single BEST_INSERT output can recover at most one event per word.", "",
        "## Frozen gates", "",
    ]
    for name in ("G1", "G2", "G3", "G4", "G5", "G6"):
        gate = gates["gates"][name]
        lines.append(f"- {name}: **{gate['result']}** — {gate['metric']}={gate['value']:.9f} {gate['operator']} {gate['threshold']}")
    lines.extend([
        "", f"Gates passed: {gates['passed_count']} / 6.", "",
        "## Robust threshold", "",
        f"- {'ROBUST_THETA=' + format(robust['value'], '.17g') if robust['authorized'] else 'ROBUST_THETA_NOT_AUTHORIZED'}", "",
        "## Interpretation", "",
        "Continuous ranking, speaker-held-out binary decisions, and exact phone-plus-boundary localization are reported separately. "
        "A strong ROC-AUC alone is not treated as successful addition detection.", "",
        "## Protocol closure", "",
        f"- Neural training: {str(protocol['neural_training']).upper()}",
        f"- TRAIN inference: {str(protocol['train_inference']).upper()}",
        f"- VALIDATION inference: {str(protocol['validation_inference_run']).upper()}",
        f"- TEST inference: {str(protocol['test_inference_run']).upper()}",
    ])
    return "\n".join(lines) + "\n"


def execute() -> None:
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started = time.perf_counter()
    for name in OUTPUT_NAMES:
        if (EXPERIMENT_DIR / name).exists():
            raise RuntimeError(f"Refusing rerun/overwrite; execution artifact already exists: {name}")
    identity = verify_identities()
    implementation_path = Path(__file__).resolve()
    implementation_sha = sha256(implementation_path)
    write_json(EXPERIMENT_DIR / "r5_1_implementation_identity.json", {
        "path": str(implementation_path), "bytes": implementation_path.stat().st_size,
        "sha256": implementation_sha, "frozen_before_train_inference": True,
    })

    audio_root_value = os.environ.get("L2_ARCTIC_ROOT")
    if not audio_root_value:
        raise RuntimeError("R5_1_EXECUTION_BLOCKED_MODEL_CONTRACT: L2_ARCTIC_ROOT not set")
    audio_root = Path(audio_root_value).resolve()
    detail_rows, source_scan = scan_train_source()
    events_by_word, addition_mapping = map_train_additions(detail_rows, audio_root)
    words, accounting = build_train_words(audio_root, events_by_word)
    accounting["source_scan"] = source_scan
    accounting["addition_mapping"] = addition_mapping
    accounting["candidate_alignability"] = audit_candidate_alignability(words)
    write_json(EXPERIMENT_DIR / "r5_1_row_accounting.json", accounting)
    if accounting["status"] != "PASS":
        raise RuntimeError("R5_1_EXECUTION_BLOCKED_ROW_ACCOUNTING")
    if accounting["clean_train_addition_events_source"] != EXPECTED_SOURCE_EVENTS:
        raise RuntimeError("R5_1_EXECUTION_BLOCKED_ROW_ACCOUNTING: source addition count mismatch")
    if accounting["candidate_alignability"]["status"] != "PASS":
        raise RuntimeError("R5_1 technical scoring alignability failure before inference")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log_probs, features, inference_report = infer_train(words, device)
    keep, best, scores, best_boundary, best_phone, scoring_report = score_train(words, log_probs, device)
    del log_probs
    del features
    if device.type == "cuda":
        torch.cuda.empty_cache()

    truth = np.asarray([word["is_addition"] for word in words], dtype=bool)
    correct_only = np.asarray([word["correct_only"] for word in words], dtype=bool)
    continuous = {
        "addition_vs_all_nonaddition": {
            "roc_auc": float(roc_auc_score(truth, scores)),
            "positive_words": int(truth.sum()), "negative_words": int((~truth).sum()),
        },
        "addition_vs_correct_only": {
            "roc_auc": float(roc_auc_score(truth[truth | correct_only], scores[truth | correct_only])),
            "positive_words": int(truth.sum()), "correct_only_negative_words": int(correct_only.sum()),
        },
        "score": "A=BEST_INSERT_TARGET_SCORE-KEEP_TARGET_SCORE",
        "selection_use": "diagnostic gates only; not threshold selection",
    }
    write_json(EXPERIMENT_DIR / "r5_1_continuous_metrics.json", continuous)

    predictions, folds = run_loso(words, scores)
    thresholds_payload = {
        "protocol": "12-fold TRAIN-speaker LOSO",
        "folds": folds,
        "thresholds_in_train_speaker_order": [float(item["selected_threshold"]) for item in folds],
    }
    write_json(EXPERIMENT_DIR / "r5_1_loso_fold_thresholds.json", thresholds_payload)

    binary = binary_metrics(truth, predictions)
    false_rates = {
        "correct_only": false_addition_rate(words, predictions, "correct_only"),
        "substitution_containing_negative": false_addition_rate(words, predictions, "substitution_negative"),
        "deletion_containing_negative": false_addition_rate(words, predictions, "deletion_negative"),
    }
    binary["false_addition_rates"] = false_rates
    binary["oof_definition"] = "concatenated once-held-out predictions from 12 TRAIN-speaker folds"
    write_json(EXPERIMENT_DIR / "r5_1_binary_metrics.json", binary)

    events = event_evaluation(words, predictions, best_boundary, best_phone)
    write_json(EXPERIMENT_DIR / "r5_1_event_metrics.json", events)
    score_audit = score_distribution_audit(words, scores)
    write_json(EXPERIMENT_DIR / "r5_1_score_distribution_audit.json", score_audit)

    score_rows = []
    prediction_rows = []
    speaker_threshold = {item["heldout_speaker"]: float(item["selected_threshold"]) for item in folds}
    for index, word in enumerate(words):
        base = {
            "source_identity": word["word_id"], "speaker": word["speaker_id"],
            "utterance_id": word["utterance_id"], "manual_word_index": int(word["manual_word_index"]),
            "word": word["word"], "expected_canonical_sequence": list(word["expected"]),
            "clean_addition_word": bool(word["is_addition"]),
            "ground_truth_addition_count": len(word["true_addition_events"]),
            "ground_truth_additions": word["true_addition_events"],
            "cohorts": {
                "correct_only": bool(word["correct_only"]),
                "substitution_containing_negative": bool(word["substitution_negative"]),
                "deletion_containing_negative": bool(word["deletion_negative"]),
                "mixed_substitution_addition": bool(word["is_addition"] and int(word["substitution"]) > 0),
                "mixed_deletion_addition": bool(word["is_addition"] and int(word["deletion"]) > 0),
                "multiple_addition": len(word["true_addition_events"]) > 1,
            },
            "keep_target_score": float(keep[index]), "best_insert_target_score": float(best[index]),
            "addition_score_A": float(scores[index]), "best_insert_phone_index": int(best_phone[index]),
            "best_insert_phone": PHONE_VOCAB[int(best_phone[index])],
            "best_insert_boundary": int(best_boundary[index]),
            "expected_length": len(word["expected_ids"]),
            "audio_duration_seconds": float(word["mfa_end"]) - float(word["mfa_start"]),
        }
        score_rows.append(base)
        prediction_rows.append({
            **base, "heldout_fold": word["speaker_id"],
            "fold_threshold": speaker_threshold[word["speaker_id"]],
            "predicted_addition": bool(predictions[index]),
        })
    write_jsonl(EXPERIMENT_DIR / "r5_1_train_scores.jsonl", score_rows)
    write_jsonl(EXPERIMENT_DIR / "r5_1_oof_predictions.jsonl", prediction_rows)

    values = {
        "addition_vs_all_nonaddition_roc_auc": continuous["addition_vs_all_nonaddition"]["roc_auc"],
        "addition_vs_correct_only_roc_auc": continuous["addition_vs_correct_only"]["roc_auc"],
        "oof_binary_macro_f1": binary["binary_macro_f1"],
        "oof_addition_f1": binary["addition_f1"],
        "correct_only_false_addition_rate": false_rates["correct_only"]["false_addition_rate"],
        "exact_event_f1": events["exact_event"]["f1"],
    }
    gate_records: dict[str, Any] = {}
    for name, (metric, operator, threshold) in GATE_LIMITS.items():
        value = float(values[metric])
        passed = (
            value >= threshold if operator == ">=" else value > threshold if operator == ">" else value <= threshold
        )
        gate_records[name] = {
            "metric": metric, "value": value, "operator": operator, "threshold": threshold,
            "result": "PASS" if passed else "FAIL", "full_precision_comparison": True,
        }
    passed_count = sum(item["result"] == "PASS" for item in gate_records.values())
    all_pass = passed_count == 6
    robust = {
        "authorized": all_pass,
        "fold_thresholds": [float(item["selected_threshold"]) for item in folds],
        "sorted_fold_thresholds": sorted(float(item["selected_threshold"]) for item in folds),
        "value": float(np.median(np.asarray([item["selected_threshold"] for item in folds], dtype=np.float64))) if all_pass else None,
        "status": "AUTHORIZED" if all_pass else "ROBUST_THETA_NOT_AUTHORIZED",
    }
    gates = {"gates": gate_records, "passed_count": passed_count, "total": 6, "all_pass": all_pass, "robust_theta": robust}
    write_json(EXPERIMENT_DIR / "r5_1_gate_results.json", gates)

    status = (
        "R5_1_INSERTION_HYPOTHESIS_SCORING_DEVELOPMENT_PASS"
        if all_pass else "R5_1_INSERTION_HYPOTHESIS_SCORING_NOT_CONFIRMED"
    )
    protocol = {
        "execution_count": 1,
        "neural_training": False, "optimizer_instantiated": False, "backpropagation": False,
        "train_inference": True, "train_inference_split": list(TRAIN),
        "validation_audio_paths_resolved": False, "validation_audio_accessed": False,
        "validation_inference_run": False, "validation_performance_consumed": False,
        "test_audio_paths_resolved": False, "test_audio_accessed": False,
        "test_inference_run": False, "test_performance_consumed": False,
        "r4_modified": False, "r5_0_modified": False, "frozen_r5_1_contract_modified": False,
        "checkpoint_selection_performed": False, "threshold_tuning_on_heldout_speaker": False,
        "alternative_score_family_evaluated": False,
    }
    write_json(EXPERIMENT_DIR / "r5_1_execution_protocol_audit.json", protocol)

    compute = {
        "started_at": started_wall, "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "inference": inference_report, "hypothesis_scoring": scoring_report,
        "total_seconds_before_hashing": time.perf_counter() - started,
    }
    write_json(EXPERIMENT_DIR / "r5_1_compute_report.json", compute)
    final_status = {
        "status": status, "gates_passed": passed_count, "gates_total": 6,
        "robust_theta": robust, "protocol_audit": protocol,
        "scientific_interpretation": {
            "continuous_discrimination": "reported separately from thresholded decisions",
            "binary_decision": "12-speaker TRAIN LOSO OOF only",
            "event_localization": "exact phone plus expected-sequence boundary",
        },
        "multiple_addition_limitation": events["multiple_addition_limitation"],
    }
    write_json(EXPERIMENT_DIR / "r5_1_final_status.json", final_status)
    report = build_report(identity, accounting, continuous, binary, folds, events, gates, robust, protocol, status)
    (EXPERIMENT_DIR / "R5_1_DEVELOPMENT_RESULT.md").write_text(report, encoding="utf-8")

    manifest_names = [name for name in OUTPUT_NAMES if name != "R5_1_EXECUTION_ARTIFACT_MANIFEST.json"]
    manifest_entries = []
    for name in manifest_names:
        path = EXPERIMENT_DIR / name
        manifest_entries.append({"relative_path": name, "byte_size": path.stat().st_size, "sha256": sha256(path)})
    failures = [entry["relative_path"] for entry in manifest_entries if sha256(EXPERIMENT_DIR / entry["relative_path"]) != entry["sha256"]]
    manifest = {
        "manifest_type": "additive R5-1 execution artifact manifest",
        "self_excluded": True, "artifact_count": len(manifest_entries),
        "hash_algorithm": "SHA-256", "hash_audit": "HASH_AUDIT_PASS" if not failures else "HASH_AUDIT_FAIL",
        "failures": failures, "artifacts": manifest_entries,
        "preserved_contract_manifest_sha256": EXPECTED_SHA["contract_manifest"],
    }
    write_json(EXPERIMENT_DIR / "R5_1_EXECUTION_ARTIFACT_MANIFEST.json", manifest)
    print(json.dumps({
        "status": status, "gates_passed": passed_count,
        "manifest_sha256": sha256(EXPERIMENT_DIR / "R5_1_EXECUTION_ARTIFACT_MANIFEST.json"),
        "artifact_count": len(manifest_entries), "hash_audit": manifest["hash_audit"],
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Frozen driver requires explicit --execute")
    execute()


if __name__ == "__main__":
    main()
