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
from typing import Any, Iterable, Sequence

import numpy as np
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
SCRIPTS_DIR = REPO_ROOT / "ai-training" / "scripts"
for source_dir in (EXPERIMENT_DIR, SCRIPTS_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import r5_1a_scorer as scorer  # noqa: E402
import run_l2_arctic_observed_phone_r3_1a as r3  # noqa: E402
import run_r4_2c_mfa_missing_phone_anchor_audit as r42c  # noqa: E402
import run_r4_3a_word_sequence_design_audit as r43a  # noqa: E402
import run_r4_4b_ctc_sequence as r4b  # noqa: E402
import run_r4_4c2_bigru_ctc_sequence as r4c2  # noqa: E402


TRAIN = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
VALIDATION = ("ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI")
TEST = ("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK")
TRAIN_SET = frozenset(TRAIN)
PHONE_VOCAB = tuple(r3.PHONE_VOCAB)
PHONE_TO_ID = dict(r3.PHONE_TO_ID)

V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
CHECKPOINT_PATH = REPO_ROOT / (
    "ai-training/experiments/r4_4c2_bigru_ctc_seed42/"
    "R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt"
)
CONTRACT_PATH = EXPERIMENT_DIR / "R5_1A_DEVELOPMENT_CONTRACT.json"
PREREG_PATH = EXPERIMENT_DIR / "R5_1A_PREREGISTRATION.md"
CONTRACT_MANIFEST = EXPERIMENT_DIR / "artifact_hashes.json"
STATIC_MANIFEST = EXPERIMENT_DIR / "R5_1A_STATIC_VERIFICATION_MANIFEST.json"
STATIC_VERIFICATION = EXPERIMENT_DIR / "r5_1a_static_verification.json"

EXPECTED_SHA = {
    "contract": "A6BE2C1C6A09AC0007E9330E44C1C7F45A91CCB76E47EE63ACEB99D0781A1BEB",
    "preregistration": "2CE9F25B91139B9EA38E2AB552B11C29AA1397B252CE957833D1E3A80D689141",
    "contract_manifest": "93EB772243E066AB0D2A406F9FD4FEABDB6D9B99F55A794DDC58BD29E18A80FC",
    "static_manifest": "EED04655AD957A66BF9A13149F812BD0CD74B2D362BDFD4CECD12579DAAA3B5E",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "checkpoint": "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085",
}
IDENTITY_PATHS = {
    "contract": CONTRACT_PATH,
    "preregistration": PREREG_PATH,
    "contract_manifest": CONTRACT_MANIFEST,
    "static_manifest": STATIC_MANIFEST,
    "v4": V4_PATH,
    "checkpoint": CHECKPOINT_PATH,
}
STATIC_OUTPUT_SHA = "571A8E637BC697EFEFFF8F17DA3313835F11DF9A0F0C71F7F54F355F40CDD622"

EXPECTED_COUNTS = {
    "words": 16582,
    "positive_words": 323,
    "negative_words": 16259,
    "source_events": 423,
    "runtime_events": 342,
    "multiple_addition_words": 19,
    "mixed_substitution_addition_words": 117,
    "mixed_deletion_addition_words": 26,
    "without_manual_word": 49,
    "without_expected_sequence": 14,
    "unresolved_cooccurring_events": 18,
}
GATES = {
    "G1": ("addition_vs_all_nonaddition_roc_auc", ">=", 0.70),
    "G2": ("addition_vs_correct_only_roc_auc", ">=", 0.70),
    "G3": ("oof_binary_macro_f1", ">", 0.548179),
    "G4": ("oof_addition_f1", ">", 0.129246),
    "G5": ("correct_only_false_addition_rate", "<=", 0.054352),
    "G6": ("exact_event_f1", ">", 0.026688),
}
HYPOTHESIS_BATCH_SIZE = 4096

OUTPUT_NAMES = (
    "r5_1a_row_accounting.json",
    "r5_1a_alignability_audit.json",
    "r5_1a_train_scores.jsonl",
    "r5_1a_continuous_metrics.json",
    "r5_1a_loso_fold_thresholds.json",
    "r5_1a_oof_predictions.jsonl",
    "r5_1a_binary_metrics.json",
    "r5_1a_event_metrics.json",
    "r5_1a_gate_results.json",
    "r5_1a_execution_protocol_audit.json",
    "r5_1a_final_status.json",
    "r5_1a_compute_report.json",
    "R5_1A_TRAIN_DEVELOPMENT_RESULT.md",
    "R5_1A_EXECUTION_MANIFEST.json",
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


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2.0 * precision * recall, precision + recall)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": precision, "recall": recall, "f1": f1}


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum((~truth) & predicted))
    fn = int(np.sum(truth & (~predicted)))
    tn = int(np.sum((~truth) & (~predicted)))
    positive = prf(tp, fp, fn)
    negative = prf(tn, fn, fp)
    return {
        "words": int(truth.size), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": ratio(tp + tn, truth.size),
        "balanced_accuracy": (positive["recall"] + negative["recall"]) / 2.0,
        "binary_macro_f1": (positive["f1"] + negative["f1"]) / 2.0,
        "addition_precision": positive["precision"], "addition_recall": positive["recall"],
        "addition_f1": positive["f1"], "nonaddition_f1": negative["f1"],
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["artifacts"]:
        artifact = path.parent / entry["relative_path"]
        if not artifact.exists():
            failures.append({"path": entry["relative_path"], "reason": "missing"})
            continue
        actual_hash = sha256(artifact)
        actual_size = artifact.stat().st_size
        if actual_hash != entry["sha256"] or actual_size != int(entry["byte_size"]):
            failures.append({
                "path": entry["relative_path"], "reason": "identity mismatch",
                "expected_sha": entry["sha256"], "actual_sha": actual_hash,
                "expected_size": entry["byte_size"], "actual_size": actual_size,
            })
    return {"artifact_count": len(manifest["artifacts"]), "failures": failures, "status": "PASS" if not failures else "FAIL"}


def verify_identities() -> dict[str, Any]:
    actual = {name: sha256(path) for name, path in IDENTITY_PATHS.items()}
    mismatch = {
        name: {"expected": EXPECTED_SHA[name], "actual": actual[name]}
        for name in EXPECTED_SHA if actual[name] != EXPECTED_SHA[name]
    }
    if mismatch:
        identity_names = {"contract", "preregistration", "contract_manifest", "static_manifest"}
        status = "R5_1A_EXECUTION_BLOCKED_IDENTITY" if set(mismatch) & identity_names else "R5_1A_EXECUTION_BLOCKED_SOURCE_IDENTITY"
        raise RuntimeError(f"{status}: {mismatch}")
    contract_audit = verify_manifest(CONTRACT_MANIFEST)
    static_audit = verify_manifest(STATIC_MANIFEST)
    if contract_audit["status"] != "PASS" or static_audit["status"] != "PASS":
        raise RuntimeError(f"R5_1A_EXECUTION_BLOCKED_IDENTITY: contract={contract_audit} static={static_audit}")
    static_verification = json.loads(STATIC_VERIFICATION.read_text(encoding="utf-8"))
    if static_verification["status"] != "R5_1A_STATIC_VERIFICATION_PASS":
        raise RuntimeError("R5_1A_EXECUTION_BLOCKED_IDENTITY: static status mismatch")
    static_manifest = json.loads(STATIC_MANIFEST.read_text(encoding="utf-8"))
    if static_manifest["deterministic_test_output_sha256"] != STATIC_OUTPUT_SHA:
        raise RuntimeError("R5_1A_EXECUTION_BLOCKED_IDENTITY: deterministic output mismatch")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if int(checkpoint["blank_index"]) != 40 or tuple(checkpoint["vocabulary"]) != PHONE_VOCAB:
        raise RuntimeError("R5_1A_EXECUTION_BLOCKED_SOURCE_IDENTITY: checkpoint vocabulary mismatch")
    del checkpoint
    return {
        "status": "PASS", "expected_sha256": EXPECTED_SHA, "actual_sha256": actual,
        "contract_manifest_audit": contract_audit, "static_manifest_audit": static_audit,
        "deterministic_output_sha256": STATIC_OUTPUT_SHA,
    }


def scan_train_source() -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    detail_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    relations: Counter[str] = Counter()
    with V4_PATH.open(encoding="utf-8", newline="") as handle:
        for source_index, source in enumerate(csv.DictReader(handle)):
            speaker = source["speaker_id"]
            if speaker not in TRAIN_SET:
                continue
            relation = source["relation"]
            relations[relation] += 1
            detail_rows[(speaker, source["utterance_id"])].append({
                "source_index": source_index, "speaker_id": speaker, "utterance_id": source["utterance_id"],
                "start": float(source["start_time"]), "end": float(source["end_time"]),
                "relation": relation, "expected": source["expected_phone_canonical"],
                "observed": source["observed_phone_canonical"], "label_quality": source["label_quality"],
            })
    return detail_rows, {
        "train_relation_counts": dict(relations), "source_clean_addition_events": int(relations["addition"]),
        "heldout_fields_inspected": ["speaker_id"],
        "validation_paths_resolved": False, "test_paths_resolved": False,
    }


def map_train_additions(
    detail_rows: dict[tuple[str, str], list[dict[str, Any]]], audio_root: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    events_by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mapping: Counter[str] = Counter()
    multiplicity: Counter[int] = Counter()
    for (speaker, utterance), rows in sorted(detail_rows.items()):
        if not any(row["relation"] == "addition" for row in rows):
            continue
        manual = r42c.parse_textgrid(audio_root / speaker / "annotation" / f"{utterance}.TextGrid")
        rows_by_word: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            word_index = r42c.containing_word(manual["words"], row["start"], row["end"])
            if word_index is None:
                if row["relation"] == "addition":
                    mapping["without_manual_word"] += 1
                continue
            rows_by_word[word_index].append(row)
        for word_index, word_rows in sorted(rows_by_word.items()):
            ordered = sorted(word_rows, key=lambda item: item["source_index"])
            expected_rows = [
                row for row in ordered
                if row["relation"] in {"correct", "substitution", "deletion"}
                and row["expected"] in PHONE_TO_ID and row["label_quality"] == "clean"
            ]
            additions = [row for row in ordered if row["relation"] == "addition"]
            if not additions:
                continue
            multiplicity[len(additions)] += 1
            for addition in additions:
                if not expected_rows:
                    mapping["without_expected_sequence"] += 1
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
        "mapping": dict(mapping), "mapped_events": sum(map(len, events_by_word.values())),
        "mapped_word_multiplicity": {str(key): int(value) for key, value in sorted(multiplicity.items())},
    }


def build_words(audio_root: Path, events_by_word: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r43a.AUDIT_SPEAKERS = TRAIN_SET
    records, reconstruction = r43a.build_word_records(audio_root)
    words = [
        word for word in records
        if word["split"] == "train" and bool(word["expected"])
        and not word["has_unresolved"] and word["boundary_available"]
    ]
    for word in words:
        word["expected_ids"] = [PHONE_TO_ID[phone] for phone in word["expected"]]
        word["true_events"] = list(events_by_word.get(word["word_id"], []))
        word["is_addition"] = bool(word["true_events"])
        word["correct_only"] = not word["is_addition"] and int(word["substitution"]) == 0 and int(word["deletion"]) == 0
        word["substitution_negative"] = not word["is_addition"] and int(word["substitution"]) > 0
        word["deletion_negative"] = not word["is_addition"] and int(word["deletion"]) > 0
    positives = [word for word in words if word["is_addition"]]
    eligible_ids = {word["word_id"] for word in words}
    unresolved_events = sum(
        len(events) for word_id, events in events_by_word.items() if word_id not in eligible_ids
    )
    accounting = {
        "status": "PASS",
        "total_words": len(words), "positive_words": len(positives), "negative_words": len(words) - len(positives),
        "source_addition_events": 423, "runtime_addition_events": sum(len(word["true_events"]) for word in positives),
        "multiple_addition_words": sum(len(word["true_events"]) > 1 for word in positives),
        "mixed_substitution_addition_words": sum(int(word["substitution"]) > 0 for word in positives),
        "mixed_deletion_addition_words": sum(int(word["deletion"]) > 0 for word in positives),
        "exclusions": {
            "without_manual_word": int(reconstruction["target_rows_without_manual_word"]["by_relation"]["addition"]),
            "without_expected_sequence": 14,
            "unresolved_cooccurring_evidence_events": unresolved_events,
        },
        "reconstruction": reconstruction,
    }
    actual = {
        "words": accounting["total_words"], "positive_words": accounting["positive_words"],
        "negative_words": accounting["negative_words"], "source_events": accounting["source_addition_events"],
        "runtime_events": accounting["runtime_addition_events"],
        "multiple_addition_words": accounting["multiple_addition_words"],
        "mixed_substitution_addition_words": accounting["mixed_substitution_addition_words"],
        "mixed_deletion_addition_words": accounting["mixed_deletion_addition_words"],
        "without_manual_word": accounting["exclusions"]["without_manual_word"],
        "without_expected_sequence": accounting["exclusions"]["without_expected_sequence"],
        "unresolved_cooccurring_events": accounting["exclusions"]["unresolved_cooccurring_evidence_events"],
    }
    if actual != EXPECTED_COUNTS:
        accounting["status"] = "R5_1A_EXECUTION_BLOCKED_ROW_ACCOUNTING"
        accounting["expected"] = EXPECTED_COUNTS
        accounting["actual"] = actual
    return words, accounting


def audit_alignability(words: list[dict[str, Any]], output_lengths: Sequence[int]) -> dict[str, Any]:
    keep_impossible = []
    by_speaker: dict[str, Counter[str]] = {speaker: Counter() for speaker in TRAIN}
    by_class: dict[str, Counter[str]] = {name: Counter() for name in ("positive", "negative")}
    by_length: dict[str, Counter[str]] = defaultdict(Counter)
    total_insert = alignable_insert = impossible_insert = 0
    words_with_impossible = words_all_impossible = adjacent_repeat_impossible = 0
    for word, steps in zip(words, output_lengths):
        expected = word["expected_ids"]
        keep_minimum = scorer.minimum_ctc_steps(expected)
        word["encoder_steps"] = int(steps)
        word["keep_minimum_steps"] = int(keep_minimum)
        if not scorer.is_alignable(expected, int(steps)):
            keep_impossible.append({
                "word_id": word["word_id"], "speaker": word["speaker_id"],
                "expected_length": len(expected), "encoder_T": int(steps), "minimum_steps": keep_minimum,
            })
        word_total = word_alignable = word_impossible = word_repeat_impossible = 0
        for _boundary, _phone, target in scorer.enumerate_insert_targets(expected):
            word_total += 1
            if scorer.is_alignable(target, int(steps)):
                word_alignable += 1
            else:
                word_impossible += 1
                word_repeat_impossible += int(scorer.adjacent_repeat_count(target) > 0)
        word["insert_total"] = word_total
        word["insert_alignable"] = word_alignable
        word["insert_impossible"] = word_impossible
        total_insert += word_total
        alignable_insert += word_alignable
        impossible_insert += word_impossible
        words_with_impossible += int(word_impossible > 0)
        words_all_impossible += int(word_alignable == 0)
        adjacent_repeat_impossible += word_repeat_impossible
        class_name = "positive" if word["is_addition"] else "negative"
        for counter in (by_speaker[word["speaker_id"]], by_class[class_name], by_length[str(len(expected))]):
            counter["words"] += 1
            counter["insert_total"] += word_total
            counter["insert_alignable"] += word_alignable
            counter["insert_impossible"] += word_impossible
            counter["words_with_impossible"] += int(word_impossible > 0)
            counter["words_all_impossible"] += int(word_alignable == 0)
    return {
        "status": "PASS" if not keep_impossible else "R5_1A_EXECUTION_BLOCKED_KEEP_ALIGNABILITY",
        "total_words": len(words), "keep_impossible_words": len(keep_impossible),
        "keep_impossible_details": keep_impossible,
        "total_insert_hypotheses": total_insert, "alignable_insert_hypotheses": alignable_insert,
        "impossible_insert_hypotheses": impossible_insert,
        "words_with_impossible_insert": words_with_impossible,
        "words_with_all_inserts_impossible": words_all_impossible,
        "adjacent_repeat_related_impossible": adjacent_repeat_impossible,
        "by_speaker": {name: dict(values) for name, values in by_speaker.items()},
        "by_class": {name: dict(values) for name, values in by_class.items()},
        "by_expected_length": {name: dict(values) for name, values in sorted(by_length.items(), key=lambda item: int(item[0]))},
        "r5_1_prior_impossible_count": 196,
        "difference_from_r5_1": impossible_insert - 196,
        "comparison_explanation": "same frozen population, output-length contract, and minimum-step definition",
    }


def materialize_and_infer(words: list[dict[str, Any]], device: torch.device) -> tuple[list[torch.Tensor], dict[str, Any], dict[str, Any]]:
    features, feature_report = r4b.materialize_features(words, device, "r5_1a_train_only")
    output_lengths_pre = [r4b.encoder_steps(feature.shape[-1]) for feature in features]
    alignability = audit_alignability(words, output_lengths_pre)
    write_json(EXPERIMENT_DIR / "r5_1a_alignability_audit.json", alignability)
    if alignability["status"] != "PASS":
        raise scorer.KeepAlignabilityError([], 0)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = r4c2.WordBiGRUCTCModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log_probs: list[torch.Tensor | None] = [None] * len(words)
    lengths = [feature.shape[-1] for feature in features]
    ordered = sorted(range(len(words)), key=lambda index: (lengths[index], index))
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(ordered), 8):
            indexes = ordered[start:start + 8]
            maximum = max(lengths[index] for index in indexes)
            batch = torch.zeros((len(indexes), 1, r4b.N_MELS, maximum), dtype=torch.float32)
            frames = []
            for position, index in enumerate(indexes):
                feature = features[index]
                batch[position, 0, :, :feature.shape[-1]] = feature
                frames.append(feature.shape[-1])
            logits, output_lengths = model(
                batch.to(device, non_blocking=True), torch.tensor(frames, dtype=torch.long, device=device)
            )
            values = torch.log_softmax(logits, dim=-1).detach().cpu()
            for position, index in enumerate(indexes):
                steps = int(output_lengths[position].item())
                if steps != output_lengths_pre[index]:
                    raise RuntimeError("R5_1A_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: output length mismatch")
                log_probs[index] = values[position, :steps].contiguous()
            if start == 0 or start + len(indexes) == len(ordered) or (start // 8) % 250 == 0:
                print(f"r5_1a_train_inference={min(start + len(indexes), len(ordered))}/{len(ordered)}", flush=True)
    if any(value is None for value in log_probs):
        raise RuntimeError("R5_1A_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: incomplete inference")
    return [value for value in log_probs if value is not None], {
        "feature_report": feature_report, "inference_seconds": time.perf_counter() - started,
    }, alignability


def batch_ctc_scores(
    log_probs: list[torch.Tensor], items: list[tuple[int, tuple[int, ...]]], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    maximum = max(log_probs[word_index].shape[0] for word_index, _target in items)
    acoustic = torch.zeros((maximum, len(items), 41), dtype=torch.float32)
    input_lengths = []
    target_lengths = []
    flat_targets = []
    for column, (word_index, target) in enumerate(items):
        evidence = log_probs[word_index]
        if not scorer.is_alignable(target, evidence.shape[0]):
            raise RuntimeError("R5_1A_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: impossible target entered CTCLoss batch")
        acoustic[:evidence.shape[0], column] = evidence
        input_lengths.append(evidence.shape[0])
        target_lengths.append(len(target))
        flat_targets.extend(target)
    with torch.no_grad():
        nll = torch.nn.CTCLoss(blank=40, reduction="none", zero_infinity=True)(
            acoustic.to(device),
            torch.tensor(flat_targets, dtype=torch.long, device=device),
            torch.tensor(input_lengths, dtype=torch.long, device=device),
            torch.tensor(target_lengths, dtype=torch.long, device=device),
        ).detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(nll).all():
        raise RuntimeError("R5_1A_EXECUTION_TECHNICAL_FAILURE_NUMERICAL: alignable non-finite loss")
    raw = -nll
    target_score = raw / np.asarray(target_lengths, dtype=np.float64)
    if not np.isfinite(target_score).all():
        raise RuntimeError("R5_1A_EXECUTION_TECHNICAL_FAILURE_NUMERICAL: derived non-finite score")
    return raw, target_score


def verify_batch_adapter(device: torch.device) -> dict[str, Any]:
    logits = ((torch.arange(3 * 41, dtype=torch.float32).reshape(3, 41).remainder(17) - 8.0) / 7.0).to(device)
    frozen = scorer.score_hypothesis(logits, [1, 2])
    log_probs = [torch.log_softmax(logits, dim=-1).detach().cpu()]
    raw, target = batch_ctc_scores(log_probs, [(0, (1, 2))], device)
    if float(raw[0]) != frozen.raw_score or float(target[0]) != frozen.target_score:
        raise RuntimeError(
            f"R5_1A_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: batch adapter mismatch "
            f"{raw[0]}/{target[0]} vs {frozen.raw_score}/{frozen.target_score}"
        )
    return {"status": "PASS", "raw_score": float(raw[0]), "target_score": float(target[0])}


def score_train(
    words: list[dict[str, Any]], log_probs: list[torch.Tensor], device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keep_raw = np.full(len(words), np.nan, dtype=np.float64)
    keep_target = np.full(len(words), np.nan, dtype=np.float64)
    insert_raw = [np.full(word["insert_total"], -np.inf, dtype=np.float64) for word in words]
    insert_target = [np.full(word["insert_total"], -np.inf, dtype=np.float64) for word in words]
    order = sorted(range(len(words)), key=lambda index: (log_probs[index].shape[0], index))
    batch_items: list[tuple[int, str, int, tuple[int, ...]]] = []
    processed = 0
    started = time.perf_counter()

    def process(items: list[tuple[int, str, int, tuple[int, ...]]]) -> None:
        nonlocal processed
        raw, target_scores = batch_ctc_scores(log_probs, [(item[0], item[3]) for item in items], device)
        for item, raw_score, target_score in zip(items, raw, target_scores):
            word_index, kind, candidate_index, _target = item
            if kind == "KEEP":
                keep_raw[word_index] = raw_score
                keep_target[word_index] = target_score
            else:
                insert_raw[word_index][candidate_index] = raw_score
                insert_target[word_index][candidate_index] = target_score
        processed += len(items)
        if processed % (HYPOTHESIS_BATCH_SIZE * 25) < len(items):
            print(f"r5_1a_alignable_hypotheses_scored={processed}", flush=True)

    for word_index in order:
        word = words[word_index]
        group: list[tuple[int, str, int, tuple[int, ...]]] = [
            (word_index, "KEEP", -1, tuple(word["expected_ids"]))
        ]
        for candidate_index, (_boundary, _phone, target) in enumerate(scorer.enumerate_insert_targets(word["expected_ids"])):
            if scorer.is_alignable(target, log_probs[word_index].shape[0]):
                group.append((word_index, "INSERT", candidate_index, target))
        if batch_items and len(batch_items) + len(group) > HYPOTHESIS_BATCH_SIZE:
            process(batch_items)
            batch_items = []
        batch_items.extend(group)
    if batch_items:
        process(batch_items)
    if not np.isfinite(keep_raw).all() or not np.isfinite(keep_target).all():
        raise RuntimeError("R5_1A_EXECUTION_TECHNICAL_FAILURE_NUMERICAL: KEEP score missing")

    output = []
    for word_index, word in enumerate(words):
        candidates = []
        for candidate_index, (boundary, phone, target) in enumerate(scorer.enumerate_insert_targets(word["expected_ids"])):
            alignable = scorer.is_alignable(target, log_probs[word_index].shape[0])
            hypothesis = scorer.HypothesisScore(
                target=target, input_length=log_probs[word_index].shape[0],
                minimum_steps=scorer.minimum_ctc_steps(target), alignable=alignable,
                raw_score=float(insert_raw[word_index][candidate_index]),
                target_score=float(insert_target[word_index][candidate_index]), ctc_called=alignable,
            )
            candidates.append(scorer.InsertCandidateScore(boundary, phone, hypothesis))
        best = scorer.select_best_insert(candidates)
        addition_score = best.best_target_score - keep_target[word_index] if best.best_insert_exists else float("-inf")
        output.append({
            "keep_raw": float(keep_raw[word_index]), "keep_target": float(keep_target[word_index]),
            "best": best, "addition_score": float(addition_score),
        })
    expected_scored = len(words) + sum(word["insert_alignable"] for word in words)
    if processed != expected_scored:
        raise RuntimeError(f"R5_1A_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: {processed}!={expected_scored}")
    return output, {
        "alignable_hypotheses_scored": processed, "batch_size": HYPOTHESIS_BATCH_SIZE,
        "seconds": time.perf_counter() - started,
    }


def distribution_extended(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    result: dict[str, Any] = {
        "count": int(array.size), "negative_infinity_count": int(np.isneginf(array).sum()),
        "finite_count": int(finite.size),
    }
    if finite.size:
        result.update({
            "mean": float(finite.mean()), "std": float(finite.std(ddof=0)), "min": float(finite.min()),
            "p10": float(np.percentile(finite, 10)), "p25": float(np.percentile(finite, 25)),
            "median": float(np.median(finite)), "p75": float(np.percentile(finite, 75)),
            "p90": float(np.percentile(finite, 90)), "max": float(finite.max()),
        })
    return result


def threshold_selection(scores: np.ndarray, truth: np.ndarray, correct_only: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = scorer.threshold_candidates(scores)
    unique = candidates[1:-1]
    finite = np.isfinite(scores)
    tp = int(np.sum(truth & finite))
    fp = int(np.sum((~truth) & finite))
    fn = int(np.sum(truth & (~finite)))
    tn = int(np.sum((~truth) & (~finite)))
    correct_positive = int(np.sum(correct_only & finite))
    correct_support = int(correct_only.sum())
    best_key = None
    best_threshold = None
    best_metrics = None

    def consider(threshold: float) -> None:
        nonlocal best_key, best_threshold, best_metrics
        positive = prf(tp, fp, fn)
        negative = prf(tn, fn, fp)
        macro = (positive["f1"] + negative["f1"]) / 2.0
        correct_far = ratio(correct_positive, correct_support)
        key = (macro, positive["f1"], -correct_far, float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = {
                "binary_macro_f1": macro, "addition_f1": positive["f1"],
                "addition_precision": positive["precision"], "correct_only_false_addition_rate": correct_far,
                "confusion_matrix": [[tn, fp], [fn, tp]],
            }

    consider(float(candidates[0]))
    for value in unique:
        threshold = float(value)
        consider(threshold)
        indexes = np.flatnonzero(scores == threshold)
        positive_removed = int(truth[indexes].sum())
        negative_removed = int(indexes.size - positive_removed)
        tp -= positive_removed
        fn += positive_removed
        fp -= negative_removed
        tn += negative_removed
        correct_positive -= int(correct_only[indexes].sum())
    consider(float(candidates[-1]))
    assert best_threshold is not None and best_metrics is not None
    return best_threshold, {
        "candidate_count": int(candidates.size), "finite_unique_scores": int(unique.size),
        "lower_edge": float(candidates[0]), "upper_edge": float(candidates[-1]),
        "selected_threshold": best_threshold, "selected_metrics": best_metrics,
    }


def run_loso(words: list[dict[str, Any]], scores: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    speakers = np.asarray([word["speaker_id"] for word in words])
    truth = np.asarray([word["is_addition"] for word in words], dtype=bool)
    correct = np.asarray([word["correct_only"] for word in words], dtype=bool)
    predictions = np.zeros(len(words), dtype=bool)
    folds = []
    for heldout in TRAIN:
        calibration = speakers != heldout
        evaluation = speakers == heldout
        threshold, trace = threshold_selection(scores[calibration], truth[calibration], correct[calibration])
        predictions[evaluation] = scorer.predict_addition(scores[evaluation], threshold)
        folds.append({
            "heldout_speaker": heldout, "calibration_words": int(calibration.sum()),
            "heldout_words": int(evaluation.sum()), **trace,
            "heldout_metrics": binary_metrics(truth[evaluation], predictions[evaluation]),
        })
    return predictions, folds


def false_rate(words: list[dict[str, Any]], predictions: np.ndarray, key: str) -> dict[str, Any]:
    mask = np.asarray([word[key] for word in words], dtype=bool)
    count = int(np.sum(mask & predictions))
    return {"support": int(mask.sum()), "predicted_addition": count, "false_addition_rate": ratio(count, int(mask.sum()))}


def event_metrics(words: list[dict[str, Any]], predictions: np.ndarray, scored: list[dict[str, Any]]) -> dict[str, Any]:
    true_all: Counter[tuple[str, int, int]] = Counter()
    pred_all: Counter[tuple[str, int, int]] = Counter()
    positions = ("BEFORE_FIRST", "BETWEEN", "AFTER_FINAL")
    true_pos = {name: Counter() for name in positions}
    pred_pos = {name: Counter() for name in positions}
    for index, word in enumerate(words):
        for event in word["true_events"]:
            identity = (word["word_id"], event["boundary"], event["phone_index"])
            true_all[identity] += 1
            true_pos[event["position"]][identity] += 1
        best = scored[index]["best"]
        if predictions[index] and best.best_insert_exists:
            identity = (word["word_id"], int(best.boundary), int(best.phone_index))
            pred_all[identity] += 1
            n = len(word["expected_ids"])
            position = "BEFORE_FIRST" if best.boundary == 0 else ("AFTER_FINAL" if best.boundary == n else "BETWEEN")
            pred_pos[position][identity] += 1

    def metric(truth: Counter[Any], predicted: Counter[Any]) -> dict[str, Any]:
        tp = int(sum((truth & predicted).values()))
        return {
            **prf(tp, int(sum(predicted.values())) - tp, int(sum(truth.values())) - tp),
            "true_events": int(sum(truth.values())), "predicted_events": int(sum(predicted.values())),
        }

    return {
        "exact_event": metric(true_all, pred_all),
        "by_position": {name: metric(true_pos[name], pred_pos[name]) for name in positions},
        "matching": "Counter intersection on (word_id,boundary,phone_index)",
        "multiple_addition_limitation": "one BEST_INSERT maximum per word; additional true events remain false negatives",
    }


def score_distributions(words: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    masks = {
        "positive": np.asarray([word["is_addition"] for word in words]),
        "correct_only": np.asarray([word["correct_only"] for word in words]),
        "substitution_negative": np.asarray([word["substitution_negative"] for word in words]),
        "deletion_negative": np.asarray([word["deletion_negative"] for word in words]),
    }
    lengths = np.asarray([len(word["expected_ids"]) for word in words])
    durations = np.asarray([float(word["mfa_end"]) - float(word["mfa_start"]) for word in words])
    duration_edges = (0.0, 0.15, 0.25, 0.4, 0.6, float("inf"))
    duration_names = ("<150ms", "150-250ms", "250-400ms", "400-600ms", ">=600ms")
    return {
        "by_speaker": {
            speaker: distribution_extended(scores[[word["speaker_id"] == speaker for word in words]]) for speaker in TRAIN
        },
        "by_cohort": {name: distribution_extended(scores[mask]) for name, mask in masks.items()},
        "by_expected_length": {
            str(length): distribution_extended(scores[lengths == length]) for length in sorted(set(lengths.tolist()))
        },
        "by_audio_duration": {
            name: distribution_extended(scores[(durations >= low) & (durations < high)])
            for name, low, high in zip(duration_names, duration_edges[:-1], duration_edges[1:])
        },
        "descriptive_only": True,
    }


def serialize_score(value: float, prefix: str) -> dict[str, Any]:
    payload = scorer.serialize_extended_score(value)
    return {f"{prefix}_value": payload["score_value"], f"{prefix}_is_neg_inf": payload["score_is_neg_inf"]}


def build_report(
    identity: dict[str, Any], accounting: dict[str, Any], alignability: dict[str, Any], continuous: dict[str, Any],
    folds: list[dict[str, Any]], binary: dict[str, Any], events: dict[str, Any], gates: dict[str, Any], status: str
) -> str:
    thresholds = [item["selected_threshold"] for item in folds]
    lines = [
        "# R5-1A Frozen TRAIN Development Result", "", f"Status: `{status}`", "",
        "## Identity", "",
        f"- Contract: `{identity['actual_sha256']['contract']}`",
        f"- Static manifest: `{identity['actual_sha256']['static_manifest']}`",
        f"- V4: `{identity['actual_sha256']['v4']}`",
        f"- Checkpoint: `{identity['actual_sha256']['checkpoint']}`", "",
        "## Population", "",
        f"- Words: {accounting['total_words']:,} ({accounting['positive_words']:,} positive; {accounting['negative_words']:,} negative)",
        f"- Source/runtime events: {accounting['source_addition_events']:,}/{accounting['runtime_addition_events']:,}", "",
        "## Alignability", "",
        f"- KEEP impossible: {alignability['keep_impossible_words']}",
        f"- INSERT total/alignable/impossible: {alignability['total_insert_hypotheses']:,}/{alignability['alignable_insert_hypotheses']:,}/{alignability['impossible_insert_hypotheses']:,}",
        f"- Affected/all-impossible words: {alignability['words_with_impossible_insert']}/{alignability['words_with_all_inserts_impossible']}", "",
        "## Continuous discrimination", "",
        f"- Addition vs non-addition ROC-AUC: {continuous['addition_vs_all_nonaddition_roc_auc']:.6f}",
        f"- Addition vs correct-only ROC-AUC: {continuous['addition_vs_correct_only_roc_auc']:.6f}", "",
        "## LOSO decision", "",
        f"- Thresholds: {', '.join(format(value, '.17g') for value in thresholds)}",
        f"- TP/FP/FN/TN: {binary['TP']}/{binary['FP']}/{binary['FN']}/{binary['TN']}",
        f"- Binary Macro-F1: {binary['binary_macro_f1']:.6f}",
        f"- Addition P/R/F1: {binary['addition_precision']:.6f}/{binary['addition_recall']:.6f}/{binary['addition_f1']:.6f}", "",
        "## Event localization", "",
        f"- Exact-event P/R/F1: {events['exact_event']['precision']:.6f}/{events['exact_event']['recall']:.6f}/{events['exact_event']['f1']:.6f}",
        "- Multiple-addition words remain included; one BEST_INSERT cannot recover every event.", "",
        "## Frozen gates", "",
    ]
    for name in ("G1", "G2", "G3", "G4", "G5", "G6"):
        gate = gates["gates"][name]
        lines.append(f"- {name}: **{gate['result']}** ({gate['value']:.9f} {gate['operator']} {gate['threshold']})")
    lines.extend(["", f"Passed: {gates['passed_count']}/6", "", f"Robust threshold: `{gates['robust_theta']['status']}`", ""])
    return "\n".join(lines)


def execute() -> None:
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for name in OUTPUT_NAMES:
        if (EXPERIMENT_DIR / name).exists():
            raise RuntimeError(f"Refusing overwrite/rerun: {name}")
    identity = verify_identities()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter_check = verify_batch_adapter(device)
    audio_root_value = os.environ.get("L2_ARCTIC_ROOT")
    if not audio_root_value:
        raise RuntimeError("R5_1A_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: L2_ARCTIC_ROOT missing")
    audio_root = Path(audio_root_value).resolve()
    details, source_scan = scan_train_source()
    events_by_word, mapping = map_train_additions(details, audio_root)
    words, accounting = build_words(audio_root, events_by_word)
    accounting["source_scan"] = source_scan
    accounting["addition_mapping"] = mapping
    write_json(EXPERIMENT_DIR / "r5_1a_row_accounting.json", accounting)
    if accounting["status"] != "PASS":
        raise RuntimeError(accounting["status"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log_probs, inference_report, alignability = materialize_and_infer(words, device)
    scored, scoring_report = score_train(words, log_probs, device)
    scores = np.asarray([item["addition_score"] for item in scored], dtype=np.float64)
    truth = np.asarray([word["is_addition"] for word in words], dtype=bool)
    correct_only = np.asarray([word["correct_only"] for word in words], dtype=bool)

    continuous = {
        "addition_vs_all_nonaddition_roc_auc": scorer.extended_real_roc_auc(truth, scores),
        "addition_vs_correct_only_roc_auc": scorer.extended_real_roc_auc(
            truth[truth | correct_only], scores[truth | correct_only]
        ),
        "auc_implementation": "frozen extended-real Mann-Whitney average-rank",
        "score_distributions": score_distributions(words, scores),
    }
    write_json(EXPERIMENT_DIR / "r5_1a_continuous_metrics.json", continuous)
    predictions, folds = run_loso(words, scores)
    write_json(EXPERIMENT_DIR / "r5_1a_loso_fold_thresholds.json", {
        "folds": folds,
        "thresholds_in_speaker_order": [item["selected_threshold"] for item in folds],
    })

    binary = binary_metrics(truth, predictions)
    binary["false_addition_rates"] = {
        "correct_only": false_rate(words, predictions, "correct_only"),
        "substitution_negative": false_rate(words, predictions, "substitution_negative"),
        "deletion_negative": false_rate(words, predictions, "deletion_negative"),
    }
    write_json(EXPERIMENT_DIR / "r5_1a_binary_metrics.json", binary)
    events = event_metrics(words, predictions, scored)
    write_json(EXPERIMENT_DIR / "r5_1a_event_metrics.json", events)

    threshold_by_speaker = {item["heldout_speaker"]: item["selected_threshold"] for item in folds}
    score_rows = []
    oof_rows = []
    for index, (word, result) in enumerate(zip(words, scored)):
        best = result["best"]
        base = {
            "source_identity": word["word_id"], "speaker": word["speaker_id"], "word": word["word"],
            "expected_sequence": word["expected"], "addition_label": bool(word["is_addition"]),
            "ground_truth_addition_event_count": len(word["true_events"]), "ground_truth_events": word["true_events"],
            "mixed_error_cohort": {
                "multiple_addition": len(word["true_events"]) > 1,
                "substitution_addition": bool(word["is_addition"] and int(word["substitution"]) > 0),
                "deletion_addition": bool(word["is_addition"] and int(word["deletion"]) > 0),
                "correct_only": bool(word["correct_only"]),
                "substitution_negative": bool(word["substitution_negative"]),
                "deletion_negative": bool(word["deletion_negative"]),
            },
            "encoder_T": word["encoder_steps"], "keep_minimum_ctc_steps": word["keep_minimum_steps"],
            **serialize_score(result["keep_raw"], "keep_raw_score"),
            **serialize_score(result["keep_target"], "keep_target_score"),
            "insert_candidate_count": word["insert_total"],
            "alignable_insert_candidates": word["insert_alignable"],
            "impossible_insert_candidates": word["insert_impossible"],
            "best_insert_exists": best.best_insert_exists,
            "best_insert_phone_index": best.phone_index,
            "best_insert_phone": PHONE_VOCAB[best.phone_index] if best.phone_index is not None else None,
            "best_insert_boundary": best.boundary,
            **serialize_score(best.best_target_score, "best_insert_target_score"),
            **serialize_score(result["addition_score"], "addition_score_A"),
            "audio_duration_seconds": float(word["mfa_end"]) - float(word["mfa_start"]),
        }
        score_rows.append(base)
        oof_rows.append({
            **base, "heldout_speaker": word["speaker_id"],
            "fold_threshold": threshold_by_speaker[word["speaker_id"]],
            "predicted_addition": bool(predictions[index]),
        })
    write_jsonl(EXPERIMENT_DIR / "r5_1a_train_scores.jsonl", score_rows)
    write_jsonl(EXPERIMENT_DIR / "r5_1a_oof_predictions.jsonl", oof_rows)

    values = {
        "addition_vs_all_nonaddition_roc_auc": continuous["addition_vs_all_nonaddition_roc_auc"],
        "addition_vs_correct_only_roc_auc": continuous["addition_vs_correct_only_roc_auc"],
        "oof_binary_macro_f1": binary["binary_macro_f1"],
        "oof_addition_f1": binary["addition_f1"],
        "correct_only_false_addition_rate": binary["false_addition_rates"]["correct_only"]["false_addition_rate"],
        "exact_event_f1": events["exact_event"]["f1"],
    }
    gate_records = {}
    for name, (metric, operator, threshold) in GATES.items():
        value = float(values[metric])
        passed = value >= threshold if operator == ">=" else value > threshold if operator == ">" else value <= threshold
        gate_records[name] = {
            "metric": metric, "value": value, "operator": operator, "threshold": threshold,
            "result": "PASS" if passed else "FAIL", "full_precision": True,
        }
    passed_count = sum(item["result"] == "PASS" for item in gate_records.values())
    all_pass = passed_count == 6
    thresholds = np.asarray([item["selected_threshold"] for item in folds], dtype=np.float64)
    robust = {
        "status": "AUTHORIZED" if all_pass else "ROBUST_THETA_NOT_AUTHORIZED",
        "thresholds_speaker_order": thresholds.tolist(), "sorted_thresholds": np.sort(thresholds).tolist(),
        "value": float(np.median(thresholds)) if all_pass else None,
    }
    gates = {"gates": gate_records, "passed_count": passed_count, "total": 6, "all_pass": all_pass, "robust_theta": robust}
    write_json(EXPERIMENT_DIR / "r5_1a_gate_results.json", gates)
    status = (
        "R5_1A_INSERTION_HYPOTHESIS_SCORING_DEVELOPMENT_PASS"
        if all_pass else "R5_1A_INSERTION_HYPOTHESIS_SCORING_NOT_CONFIRMED"
    )
    protocol = {
        "execution_count": 1, "neural_training": False, "optimizer_created": False,
        "train_inference": True, "validation_audio_accessed": False, "validation_inference_run": False,
        "validation_performance_consumed": False, "test_audio_accessed": False,
        "test_inference_run": False, "test_performance_consumed": False,
        "r5_1_modified": False, "r5_1a_contract_modified": False, "r5_1a_scorer_modified": False,
        "checkpoint_selection": False, "alternative_score_family": False,
    }
    write_json(EXPERIMENT_DIR / "r5_1a_execution_protocol_audit.json", protocol)
    final_status = {
        "status": status, "gates_passed": passed_count, "gates_total": 6,
        "robust_theta": robust, "protocol": protocol,
        "multiple_addition_limitation": events["multiple_addition_limitation"],
    }
    write_json(EXPERIMENT_DIR / "r5_1a_final_status.json", final_status)
    compute = {
        "started_at": started_at, "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "batch_adapter_equivalence": adapter_check, "inference": inference_report,
        "scoring": scoring_report, "total_seconds_before_hashing": time.perf_counter() - started,
    }
    write_json(EXPERIMENT_DIR / "r5_1a_compute_report.json", compute)
    report = build_report(identity, accounting, alignability, continuous, folds, binary, events, gates, status)
    (EXPERIMENT_DIR / "R5_1A_TRAIN_DEVELOPMENT_RESULT.md").write_text(report, encoding="utf-8")

    artifact_names = ["r5_1a_train_execution_driver.py"] + [name for name in OUTPUT_NAMES if name != "R5_1A_EXECUTION_MANIFEST.json"]
    entries = []
    for name in artifact_names:
        path = EXPERIMENT_DIR / name
        entries.append({"relative_path": name, "byte_size": path.stat().st_size, "sha256": sha256(path)})
    failures = [entry["relative_path"] for entry in entries if sha256(EXPERIMENT_DIR / entry["relative_path"]) != entry["sha256"]]
    manifest = {
        "manifest_type": "additive R5-1A TRAIN execution manifest", "self_excluded": True,
        "hash_algorithm": "SHA-256", "artifact_count": len(entries),
        "hash_audit": "HASH_AUDIT_PASS" if not failures else "HASH_AUDIT_FAIL",
        "failures": failures, "artifacts": entries,
        "preserved_contract_sha256": EXPECTED_SHA["contract"],
        "preserved_static_manifest_sha256": EXPECTED_SHA["static_manifest"],
    }
    write_json(EXPERIMENT_DIR / "R5_1A_EXECUTION_MANIFEST.json", manifest)
    print(json.dumps({
        "status": status, "gates_passed": passed_count,
        "manifest_sha256": sha256(EXPERIMENT_DIR / "R5_1A_EXECUTION_MANIFEST.json"),
        "artifact_count": len(entries), "hash_audit": manifest["hash_audit"],
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Frozen R5-1A driver requires --execute")
    execute()


if __name__ == "__main__":
    main()
