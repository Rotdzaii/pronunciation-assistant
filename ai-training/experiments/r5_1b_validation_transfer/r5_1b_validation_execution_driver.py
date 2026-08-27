from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
R5_1A_DIR = REPO_ROOT / "ai-training/experiments/r5_1a_alignability_safe_addition_scoring"
SCRIPTS_DIR = REPO_ROOT / "ai-training/scripts"
for source_dir in (R5_1A_DIR, SCRIPTS_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import r5_1a_scorer as scorer  # noqa: E402
import r5_1a_train_execution_driver as r5a  # noqa: E402
import run_l2_arctic_observed_phone_r3_1a as r3  # noqa: E402
import run_r4_2c_mfa_missing_phone_anchor_audit as r42c  # noqa: E402
import run_r4_3a_word_sequence_design_audit as r43a  # noqa: E402
import run_r4_4b_ctc_sequence as r4b  # noqa: E402
import run_r4_4c2_bigru_ctc_sequence as r4c2  # noqa: E402


TRAIN = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
VALIDATION = ("ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI")
TEST = ("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK")
VALIDATION_SET = frozenset(VALIDATION)
TEST_SET = frozenset(TEST)
PHONE_VOCAB = tuple(r3.PHONE_VOCAB)
PHONE_TO_ID = dict(r3.PHONE_TO_ID)
ROBUST_THETA = np.float64(0.7485884030659993)

V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
CHECKPOINT_PATH = REPO_ROOT / (
    "ai-training/experiments/r4_4c2_bigru_ctc_seed42/"
    "R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt"
)
R5_1B_CONTRACT = EXPERIMENT_DIR / "R5_1B_VALIDATION_CONTRACT.json"
R5_1B_PREREG = EXPERIMENT_DIR / "R5_1B_PREREGISTRATION.md"
R5_1B_CONTRACT_MANIFEST = EXPERIMENT_DIR / "artifact_hashes.json"
R5_1A_CONTRACT = R5_1A_DIR / "R5_1A_DEVELOPMENT_CONTRACT.json"
R5_1A_EXECUTION_MANIFEST = R5_1A_DIR / "R5_1A_EXECUTION_MANIFEST.json"
R5_1A_SCORER = R5_1A_DIR / "r5_1a_scorer.py"

EXPECTED_SHA = {
    "r5_1b_contract": "BD169175C0777B1C37506E95350CB6AD90A992045527396DDE5DA4419779AC4A",
    "r5_1b_preregistration": "E6AC9C3526065EC87E01E003A208DB2639711A4C6959A8CD7FAB961F7B1B890C",
    "r5_1b_contract_manifest": "1C91E2FD10D9EBFC97C9F1C10068209E50071EE7E642382E962436263E313939",
    "r5_1a_contract": "A6BE2C1C6A09AC0007E9330E44C1C7F45A91CCB76E47EE63ACEB99D0781A1BEB",
    "r5_1a_execution_manifest": "C9343A75CE26C2BEECA388EBA855E91AE0992D22C0C5D785308D0A98B89A3CD6",
    "r5_1a_scorer": "4DE49C9070C973EE44EFBD09DFC063C436779E723D12EC7A7A2BC4A06AF35F90",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "checkpoint": "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085",
}
IDENTITY_PATHS = {
    "r5_1b_contract": R5_1B_CONTRACT,
    "r5_1b_preregistration": R5_1B_PREREG,
    "r5_1b_contract_manifest": R5_1B_CONTRACT_MANIFEST,
    "r5_1a_contract": R5_1A_CONTRACT,
    "r5_1a_execution_manifest": R5_1A_EXECUTION_MANIFEST,
    "r5_1a_scorer": R5_1A_SCORER,
    "v4": V4_PATH,
    "checkpoint": CHECKPOINT_PATH,
}
FROZEN_TRAIN = {
    "addition_vs_all_nonaddition_roc_auc": 0.7734833025081417,
    "addition_vs_correct_only_roc_auc": 0.802352821409537,
    "binary_macro_f1": 0.5551978767901391,
    "addition_precision": 0.1005665722379603,
    "addition_recall": 0.2198142414860681,
    "addition_f1": 0.1379980563654033,
    "correct_only_false_addition_rate": 0.03491295938104449,
    "exact_event_f1": 0.04389312977099236,
}
GATES = {
    "G1": ("addition_vs_all_nonaddition_roc_auc", ">=", 0.70),
    "G2": ("addition_vs_correct_only_roc_auc", ">=", 0.70),
    "G3": ("fixed_threshold_binary_macro_f1", ">", 0.548179),
    "G4": ("fixed_threshold_addition_f1", ">", 0.129246),
    "G5": ("correct_only_false_addition_rate", "<=", 0.054352),
    "G6": ("exact_event_f1", ">", 0.026688),
}
OUTPUT_NAMES = (
    "r5_1b_population_accounting.json",
    "r5_1b_alignability_audit.json",
    "r5_1b_validation_scores.jsonl",
    "r5_1b_continuous_metrics.json",
    "r5_1b_binary_metrics.json",
    "r5_1b_per_speaker_metrics.json",
    "r5_1b_event_metrics.json",
    "r5_1b_train_validation_deltas.json",
    "r5_1b_gate_results.json",
    "r5_1b_execution_protocol_audit.json",
    "r5_1b_final_status.json",
    "r5_1b_compute_report.json",
    "R5_1B_VALIDATION_TRANSFER_RESULT.md",
    "R5_1B_EXECUTION_MANIFEST.json",
)


class ExecutionStop(RuntimeError):
    def __init__(self, status: str, detail: Any):
        super().__init__(status)
        self.status = status
        self.detail = detail


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
    mismatch = {name: {"expected": EXPECTED_SHA[name], "actual": actual[name]} for name in EXPECTED_SHA if actual[name] != EXPECTED_SHA[name]}
    if mismatch:
        raise ExecutionStop("R5_1B_VALIDATION_BLOCKED_IDENTITY", mismatch)
    contract_audit = verify_manifest(R5_1B_CONTRACT_MANIFEST)
    upstream_audit = verify_manifest(R5_1A_EXECUTION_MANIFEST)
    if contract_audit["status"] != "PASS" or upstream_audit["status"] != "PASS":
        raise ExecutionStop("R5_1B_VALIDATION_BLOCKED_IDENTITY", {"contract": contract_audit, "upstream": upstream_audit})
    contract = json.loads(R5_1B_CONTRACT.read_text(encoding="utf-8"))
    if float(contract["fixed_threshold_decision"]["ROBUST_THETA"]) != float(ROBUST_THETA):
        raise ExecutionStop("R5_1B_VALIDATION_BLOCKED_IDENTITY", "ROBUST_THETA mismatch")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if int(checkpoint["blank_index"]) != 40 or tuple(checkpoint["vocabulary"]) != PHONE_VOCAB:
        raise ExecutionStop("R5_1B_VALIDATION_BLOCKED_IDENTITY", "checkpoint vocabulary or blank mismatch")
    del checkpoint
    return {
        "status": "PASS", "expected_sha256": EXPECTED_SHA, "actual_sha256": actual,
        "r5_1b_contract_manifest_audit": contract_audit,
        "r5_1a_execution_manifest_audit": upstream_audit,
        "robust_theta": float(ROBUST_THETA),
    }


def guard_validation_population() -> None:
    requested = frozenset(VALIDATION)
    if requested & TEST_SET:
        raise ExecutionStop("R5_1B_VALIDATION_TECHNICAL_FAILURE_IMPLEMENTATION", "TEST speaker supplied before path resolution")
    if requested != VALIDATION_SET:
        raise ExecutionStop("R5_1B_VALIDATION_TECHNICAL_FAILURE_IMPLEMENTATION", "VALIDATION speaker set mismatch")


def scan_validation_source() -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    detail_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    relations: Counter[str] = Counter()
    additions_by_speaker: Counter[str] = Counter()
    utterances: set[tuple[str, str]] = set()
    with V4_PATH.open(encoding="utf-8", newline="") as handle:
        for source_index, source in enumerate(csv.DictReader(handle)):
            speaker = source["speaker_id"]
            if speaker not in VALIDATION_SET:
                continue
            relation = source["relation"]
            relations[relation] += 1
            additions_by_speaker[speaker] += int(relation == "addition")
            utterances.add((speaker, source["utterance_id"]))
            detail_rows[(speaker, source["utterance_id"])].append({
                "source_index": source_index, "speaker_id": speaker, "utterance_id": source["utterance_id"],
                "start": float(source["start_time"]), "end": float(source["end_time"]),
                "relation": relation, "expected": source["expected_phone_canonical"],
                "observed": source["observed_phone_canonical"], "label_quality": source["label_quality"],
            })
    return detail_rows, {
        "validation_relation_counts": dict(relations),
        "source_clean_addition_events": int(relations["addition"]),
        "clean_additions_by_speaker": {speaker: int(additions_by_speaker[speaker]) for speaker in VALIDATION},
        "validation_utterances": len(utterances),
        "nonvalidation_fields_inspected": ["speaker_id"],
        "test_paths_resolved": False,
    }


def map_validation_additions(
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


def build_validation_words(
    audio_root: Path, events_by_word: dict[str, list[dict[str, Any]]], source_scan: dict[str, Any], mapping: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r43a.AUDIT_SPEAKERS = VALIDATION_SET
    records, reconstruction = r43a.build_word_records(audio_root)
    source_records = [word for word in records if word["split"] == "validation"]
    words = [
        word for word in source_records
        if bool(word["expected"]) and not word["has_unresolved"] and word["boundary_available"]
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
    unresolved_events = sum(len(events) for word_id, events in events_by_word.items() if word_id not in eligible_ids)
    source_events = int(source_scan["source_clean_addition_events"])
    runtime_events = sum(len(word["true_events"]) for word in positives)
    without_manual = int(mapping["mapping"].get("without_manual_word", 0))
    without_expected = int(mapping["mapping"].get("without_expected_sequence", 0))
    exclusions = {
        "without_manual_word": without_manual,
        "without_expected_sequence": without_expected,
        "unresolved_or_nonruntime_cooccurring_evidence_events": int(unresolved_events),
    }
    accounting = {
        "status": "PASS",
        "source_validation_words": len(source_records),
        "runtime_evaluable_words": len(words),
        "positive_words": len(positives), "negative_words": len(words) - len(positives),
        "source_addition_events": source_events, "runtime_addition_events": runtime_events,
        "multiple_addition_words": sum(len(word["true_events"]) > 1 for word in positives),
        "mixed_substitution_addition_words": sum(int(word["substitution"]) > 0 for word in positives),
        "mixed_deletion_addition_words": sum(int(word["deletion"]) > 0 for word in positives),
        "exclusions": exclusions,
        "source_scan": source_scan, "addition_mapping": mapping, "reconstruction": reconstruction,
        "source_word_definition": "manual word records reconstructed for the frozen VALIDATION speakers before runtime eligibility filtering",
        "runtime_eligibility": "same frozen R5-1A rules; no validation-specific exclusions",
    }
    reconciliation = runtime_events + sum(exclusions.values())
    support_pass = source_events == 296 and all(source_scan["clean_additions_by_speaker"][speaker] >= 5 for speaker in VALIDATION)
    if not support_pass or reconciliation != source_events or not words or not positives:
        accounting["status"] = "R5_1B_VALIDATION_BLOCKED_ROW_ACCOUNTING"
        accounting["checks"] = {
            "source_events_expected_296": source_events == 296,
            "all_speakers_at_least_5": all(source_scan["clean_additions_by_speaker"][speaker] >= 5 for speaker in VALIDATION),
            "event_reconciliation": reconciliation == source_events,
            "nonempty_runtime_population": bool(words), "nonempty_positive_population": bool(positives),
        }
    return words, accounting


def audit_alignability(words: list[dict[str, Any]], output_lengths: Sequence[int]) -> dict[str, Any]:
    keep_impossible = []
    by_speaker: dict[str, Counter[str]] = {speaker: Counter() for speaker in VALIDATION}
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
                "expected_length": len(expected), "encoder_T": int(steps), "minimum_steps": int(keep_minimum),
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
        "status": "PASS" if not keep_impossible else "R5_1B_VALIDATION_BLOCKED_KEEP_ALIGNABILITY",
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
    }


def materialize_and_infer(words: list[dict[str, Any]], device: torch.device) -> tuple[list[torch.Tensor], dict[str, Any], dict[str, Any]]:
    feature_started = time.perf_counter()
    features, feature_report = r4b.materialize_features(words, device, "r5_1b_validation_only")
    feature_seconds = time.perf_counter() - feature_started
    output_lengths_pre = [r4b.encoder_steps(feature.shape[-1]) for feature in features]
    alignability = audit_alignability(words, output_lengths_pre)
    write_json(EXPERIMENT_DIR / "r5_1b_alignability_audit.json", alignability)
    if alignability["status"] != "PASS":
        raise ExecutionStop(alignability["status"], alignability["keep_impossible_details"])
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
                    raise ExecutionStop("R5_1B_VALIDATION_TECHNICAL_FAILURE_IMPLEMENTATION", "encoder output length mismatch")
                log_probs[index] = values[position, :steps].contiguous()
            if start == 0 or start + len(indexes) == len(ordered) or (start // 8) % 250 == 0:
                print(f"r5_1b_validation_inference={min(start + len(indexes), len(ordered))}/{len(ordered)}", flush=True)
    if any(value is None for value in log_probs):
        raise ExecutionStop("R5_1B_VALIDATION_TECHNICAL_FAILURE_IMPLEMENTATION", "incomplete inference")
    return [value for value in log_probs if value is not None], {
        "feature_report": feature_report, "feature_seconds": feature_seconds,
        "inference_seconds": time.perf_counter() - started,
    }, alignability


def distribution_audit(words: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    masks = {
        "true_addition": np.asarray([word["is_addition"] for word in words], dtype=bool),
        "correct_only": np.asarray([word["correct_only"] for word in words], dtype=bool),
        "substitution_negative": np.asarray([word["substitution_negative"] for word in words], dtype=bool),
        "deletion_negative": np.asarray([word["deletion_negative"] for word in words], dtype=bool),
    }
    return {
        "by_speaker": {
            speaker: r5a.distribution_extended(scores[np.asarray([word["speaker_id"] == speaker for word in words])])
            for speaker in VALIDATION
        },
        "by_cohort": {name: r5a.distribution_extended(scores[mask]) for name, mask in masks.items()},
        "descriptive_only": True,
    }


def per_speaker_metrics(words: list[dict[str, Any]], truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    output = {}
    for speaker in VALIDATION:
        mask = np.asarray([word["speaker_id"] == speaker for word in words], dtype=bool)
        subset_words = [word for word in words if word["speaker_id"] == speaker]
        metrics = r5a.binary_metrics(truth[mask], predictions[mask])
        metrics.update({
            "positive_support": int(truth[mask].sum()),
            "predicted_positive_rate": r5a.ratio(int(predictions[mask].sum()), int(mask.sum())),
            "correct_only_false_addition": r5a.false_rate(subset_words, predictions[mask], "correct_only"),
        })
        output[speaker] = metrics
    return {"speakers": output, "diagnostic_only": True, "threshold_tuning": False}


def serialize_score(value: float, prefix: str) -> dict[str, Any]:
    return r5a.serialize_score(value, prefix)


def build_report(
    identity: dict[str, Any], accounting: dict[str, Any], alignability: dict[str, Any],
    continuous: dict[str, Any], binary: dict[str, Any], events: dict[str, Any],
    gates: dict[str, Any], deltas: dict[str, Any], status: str,
) -> str:
    lines = [
        "# R5-1B Frozen VALIDATION Transfer Result", "", f"Status: `{status}`", "",
        "## Identity", "",
        f"- R5-1B contract: `{identity['actual_sha256']['r5_1b_contract']}`",
        f"- R5-1A execution manifest: `{identity['actual_sha256']['r5_1a_execution_manifest']}`",
        f"- V4: `{identity['actual_sha256']['v4']}`",
        f"- Checkpoint: `{identity['actual_sha256']['checkpoint']}`", "",
        "## Population", "",
        f"- Source/runtime words: {accounting['source_validation_words']:,}/{accounting['runtime_evaluable_words']:,}",
        f"- Positive/negative runtime words: {accounting['positive_words']:,}/{accounting['negative_words']:,}",
        f"- Source/runtime addition events: {accounting['source_addition_events']:,}/{accounting['runtime_addition_events']:,}", "",
        "## Alignability", "",
        f"- KEEP impossible: {alignability['keep_impossible_words']}",
        f"- INSERT total/alignable/impossible: {alignability['total_insert_hypotheses']:,}/{alignability['alignable_insert_hypotheses']:,}/{alignability['impossible_insert_hypotheses']:,}", "",
        "## Continuous transfer", "",
        f"- Addition vs non-addition ROC-AUC: {continuous['addition_vs_all_nonaddition_roc_auc']:.9f}",
        f"- Addition vs correct-only ROC-AUC: {continuous['addition_vs_correct_only_roc_auc']:.9f}",
        f"- Deltas vs TRAIN: {deltas['addition_vs_all_nonaddition_roc_auc']:+.9f}, {deltas['addition_vs_correct_only_roc_auc']:+.9f}", "",
        "## Fixed threshold", "",
        f"- ROBUST_THETA: `{float(ROBUST_THETA):.17g}`",
        f"- TP/FP/FN/TN: {binary['TP']}/{binary['FP']}/{binary['FN']}/{binary['TN']}",
        f"- Binary Macro-F1: {binary['binary_macro_f1']:.9f}",
        f"- Addition P/R/F1: {binary['addition_precision']:.9f}/{binary['addition_recall']:.9f}/{binary['addition_f1']:.9f}", "",
        "## Event localization", "",
        f"- Exact-event P/R/F1: {events['exact_event']['precision']:.9f}/{events['exact_event']['recall']:.9f}/{events['exact_event']['f1']:.9f}",
        "- Multiple-addition words remain included; one BEST_INSERT cannot recover every event.", "",
        "## Frozen gates", "",
    ]
    for name in ("G1", "G2", "G3", "G4", "G5", "G6"):
        gate = gates["gates"][name]
        lines.append(f"- {name}: **{gate['result']}** ({gate['value']:.12g} {gate['operator']} {gate['threshold']})")
    lines.extend(["", f"Passed: {gates['passed_count']}/6", "", "This is iterative development-validation transfer evidence, not independent final confirmation.", ""])
    return "\n".join(lines)


def write_execution_manifest() -> dict[str, Any]:
    artifact_names = ["r5_1b_validation_execution_driver.py"] + [name for name in OUTPUT_NAMES if name != "R5_1B_EXECUTION_MANIFEST.json"]
    entries = []
    for name in artifact_names:
        path = EXPERIMENT_DIR / name
        if path.exists():
            entries.append({"relative_path": name, "byte_size": path.stat().st_size, "sha256": sha256(path)})
    failures = []
    for entry in entries:
        path = EXPERIMENT_DIR / entry["relative_path"]
        if sha256(path) != entry["sha256"] or path.stat().st_size != entry["byte_size"]:
            failures.append(entry["relative_path"])
    manifest = {
        "manifest_type": "additive R5-1B fixed-threshold VALIDATION execution manifest",
        "self_excluded": True, "hash_algorithm": "SHA-256", "artifact_count": len(entries),
        "hash_audit": "HASH_AUDIT_PASS" if not failures else "HASH_AUDIT_FAIL",
        "failures": failures, "artifacts": entries,
        "preserved_r5_1b_contract_sha256": EXPECTED_SHA["r5_1b_contract"],
        "preserved_r5_1b_contract_manifest_sha256": EXPECTED_SHA["r5_1b_contract_manifest"],
        "preserved_r5_1a_scorer_sha256": EXPECTED_SHA["r5_1a_scorer"],
        "robust_theta": float(ROBUST_THETA), "validation_threshold_search": False,
        "test_accessed": False,
    }
    write_json(EXPERIMENT_DIR / "R5_1B_EXECUTION_MANIFEST.json", manifest)
    return {
        "artifact_count": len(entries), "hash_audit": manifest["hash_audit"],
        "manifest_sha256": sha256(EXPERIMENT_DIR / "R5_1B_EXECUTION_MANIFEST.json"),
    }


def execute() -> None:
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for name in OUTPUT_NAMES:
        if (EXPERIMENT_DIR / name).exists():
            raise ExecutionStop("R5_1B_VALIDATION_TECHNICAL_FAILURE_IMPLEMENTATION", f"refusing overwrite/rerun: {name}")
    guard_validation_population()
    identity = verify_identities()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter_check = r5a.verify_batch_adapter(device)
    audio_root_value = os.environ.get("L2_ARCTIC_ROOT")
    if not audio_root_value:
        raise ExecutionStop("R5_1B_VALIDATION_TECHNICAL_FAILURE_IMPLEMENTATION", "L2_ARCTIC_ROOT missing")
    audio_root = Path(audio_root_value).resolve()
    details, source_scan = scan_validation_source()
    events_by_word, mapping = map_validation_additions(details, audio_root)
    words, accounting = build_validation_words(audio_root, events_by_word, source_scan, mapping)
    write_json(EXPERIMENT_DIR / "r5_1b_population_accounting.json", accounting)
    if accounting["status"] != "PASS":
        raise ExecutionStop(accounting["status"], accounting)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log_probs, inference_report, alignability = materialize_and_infer(words, device)
    try:
        scored, scoring_report = r5a.score_train(words, log_probs, device)
    except RuntimeError as exc:
        status = "R5_1B_VALIDATION_TECHNICAL_FAILURE_NUMERICAL" if "NUMERICAL" in str(exc) else "R5_1B_VALIDATION_TECHNICAL_FAILURE_IMPLEMENTATION"
        raise ExecutionStop(status, str(exc)) from exc
    scores = np.asarray([item["addition_score"] for item in scored], dtype=np.float64)
    truth = np.asarray([word["is_addition"] for word in words], dtype=bool)
    correct_only = np.asarray([word["correct_only"] for word in words], dtype=bool)
    predictions = scorer.predict_addition(scores, float(ROBUST_THETA))

    continuous = {
        "addition_vs_all_nonaddition_roc_auc": scorer.extended_real_roc_auc(truth, scores),
        "addition_vs_correct_only_roc_auc": scorer.extended_real_roc_auc(truth[truth | correct_only], scores[truth | correct_only]),
        "auc_implementation": "frozen extended-real Mann-Whitney average-rank",
        "score_distributions": distribution_audit(words, scores),
        "threshold_influence": False,
    }
    write_json(EXPERIMENT_DIR / "r5_1b_continuous_metrics.json", continuous)
    binary = r5a.binary_metrics(truth, predictions)
    binary["fixed_threshold"] = float(ROBUST_THETA)
    binary["threshold_search"] = False
    binary["false_addition_rates"] = {
        "correct_only": r5a.false_rate(words, predictions, "correct_only"),
        "substitution_negative": r5a.false_rate(words, predictions, "substitution_negative"),
        "deletion_negative": r5a.false_rate(words, predictions, "deletion_negative"),
    }
    write_json(EXPERIMENT_DIR / "r5_1b_binary_metrics.json", binary)
    speaker_metrics = per_speaker_metrics(words, truth, predictions)
    write_json(EXPERIMENT_DIR / "r5_1b_per_speaker_metrics.json", speaker_metrics)
    events = r5a.event_metrics(words, predictions, scored)
    write_json(EXPERIMENT_DIR / "r5_1b_event_metrics.json", events)

    score_rows = []
    for word, result, prediction in zip(words, scored, predictions):
        best = result["best"]
        score_rows.append({
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
            "fixed_threshold": float(ROBUST_THETA), "predicted_addition": bool(prediction),
        })
    write_jsonl(EXPERIMENT_DIR / "r5_1b_validation_scores.jsonl", score_rows)

    validation_values = {
        "addition_vs_all_nonaddition_roc_auc": continuous["addition_vs_all_nonaddition_roc_auc"],
        "addition_vs_correct_only_roc_auc": continuous["addition_vs_correct_only_roc_auc"],
        "binary_macro_f1": binary["binary_macro_f1"],
        "addition_precision": binary["addition_precision"], "addition_recall": binary["addition_recall"],
        "addition_f1": binary["addition_f1"],
        "correct_only_false_addition_rate": binary["false_addition_rates"]["correct_only"]["false_addition_rate"],
        "exact_event_f1": events["exact_event"]["f1"],
    }
    deltas = {name: float(validation_values[name] - train_value) for name, train_value in FROZEN_TRAIN.items()}
    delta_payload = {"definition": "VALIDATION minus frozen TRAIN OOF", "train": FROZEN_TRAIN, "validation": validation_values, "deltas": deltas}
    write_json(EXPERIMENT_DIR / "r5_1b_train_validation_deltas.json", delta_payload)

    gate_values = {
        "addition_vs_all_nonaddition_roc_auc": continuous["addition_vs_all_nonaddition_roc_auc"],
        "addition_vs_correct_only_roc_auc": continuous["addition_vs_correct_only_roc_auc"],
        "fixed_threshold_binary_macro_f1": binary["binary_macro_f1"],
        "fixed_threshold_addition_f1": binary["addition_f1"],
        "correct_only_false_addition_rate": binary["false_addition_rates"]["correct_only"]["false_addition_rate"],
        "exact_event_f1": events["exact_event"]["f1"],
    }
    gate_records = {}
    for name, (metric, operator, threshold) in GATES.items():
        value = float(gate_values[metric])
        passed = value >= threshold if operator == ">=" else value > threshold if operator == ">" else value <= threshold
        gate_records[name] = {
            "metric": metric, "value": value, "operator": operator, "threshold": threshold,
            "result": "PASS" if passed else "FAIL", "full_precision": True,
        }
    passed_count = sum(item["result"] == "PASS" for item in gate_records.values())
    all_pass = passed_count == 6
    gates = {"gates": gate_records, "passed_count": passed_count, "total": 6, "all_pass": all_pass, "threshold_modified": False}
    write_json(EXPERIMENT_DIR / "r5_1b_gate_results.json", gates)
    status = "R5_1B_VALIDATION_TRANSFER_PASS" if all_pass else "R5_1B_VALIDATION_TRANSFER_NOT_CONFIRMED"
    protocol = {
        "execution_count": 1, "neural_training": False, "optimizer_created": False,
        "train_inference": False, "validation_inference": True, "validation_performance_consumed": True,
        "threshold_search": False, "threshold_modified": False, "validation_optimal_threshold_calculated": False,
        "test_paths_resolved": False, "test_audio_accessed": False, "test_inference_run": False,
        "test_performance_consumed": False, "r5_1a_modified": False,
        "r5_1b_contract_modified": False, "r5_1a_scorer_modified": False,
        "checkpoint_selection": False, "alternative_score_family": False,
    }
    write_json(EXPERIMENT_DIR / "r5_1b_execution_protocol_audit.json", protocol)
    final_status = {
        "status": status, "gates_passed": passed_count, "gates_total": 6,
        "robust_theta": float(ROBUST_THETA), "protocol": protocol,
        "evidence_role": "iterative development-validation transfer; not independent final confirmation",
        "multiple_addition_limitation": events["multiple_addition_limitation"],
    }
    write_json(EXPERIMENT_DIR / "r5_1b_final_status.json", final_status)
    compute = {
        "started_at": started_at, "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "batch_adapter_equivalence": adapter_check, "inference": inference_report,
        "scoring": scoring_report, "total_seconds_before_hashing": time.perf_counter() - started,
    }
    write_json(EXPERIMENT_DIR / "r5_1b_compute_report.json", compute)
    report = build_report(identity, accounting, alignability, continuous, binary, events, gates, deltas, status)
    (EXPERIMENT_DIR / "R5_1B_VALIDATION_TRANSFER_RESULT.md").write_text(report, encoding="utf-8")
    manifest = write_execution_manifest()
    print(json.dumps({"status": status, "gates_passed": passed_count, **manifest}, indent=2), flush=True)


def write_technical_stop(status: str, detail: Any, started: float, started_at: str) -> dict[str, Any]:
    protocol = {
        "execution_count": 1, "neural_training": False, "train_inference": False,
        "validation_inference": (EXPERIMENT_DIR / "r5_1b_alignability_audit.json").exists(),
        "validation_performance_consumed": (EXPERIMENT_DIR / "r5_1b_continuous_metrics.json").exists(),
        "threshold_search": False, "threshold_modified": False,
        "test_paths_resolved": False, "test_audio_accessed": False, "test_inference_run": False,
        "test_performance_consumed": False, "r5_1a_modified": False,
        "r5_1b_contract_modified": False, "r5_1a_scorer_modified": False,
    }
    write_json(EXPERIMENT_DIR / "r5_1b_execution_protocol_audit.json", protocol)
    write_json(EXPERIMENT_DIR / "r5_1b_final_status.json", {"status": status, "technical_detail": detail, "protocol": protocol})
    write_json(EXPERIMENT_DIR / "r5_1b_compute_report.json", {
        "started_at": started_at, "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_seconds_before_hashing": time.perf_counter() - started,
    })
    (EXPERIMENT_DIR / "R5_1B_VALIDATION_TRANSFER_RESULT.md").write_text(
        f"# R5-1B Frozen VALIDATION Transfer Result\n\nStatus: `{status}`\n\nNo scientific status was assigned.\n",
        encoding="utf-8",
    )
    return write_execution_manifest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Frozen R5-1B driver requires --execute")
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        execute()
    except ExecutionStop as exc:
        manifest = write_technical_stop(exc.status, exc.detail, started, started_at)
        print(json.dumps({"status": exc.status, **manifest}, indent=2), flush=True)
        raise SystemExit(2) from exc
    except Exception as exc:
        status = "R5_1B_VALIDATION_TECHNICAL_FAILURE_IMPLEMENTATION"
        manifest = write_technical_stop(status, {"type": type(exc).__name__, "message": str(exc)}, started, started_at)
        print(json.dumps({"status": status, **manifest}, indent=2), flush=True)
        raise


if __name__ == "__main__":
    main()
