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

import librosa
import numpy as np
import torch


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


V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
V4_AUDIT_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4_audit.json"
CHECKPOINT_PATH = REPO_ROOT / (
    "ai-training/experiments/r4_4c2_bigru_ctc_seed42/"
    "R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt"
)
R4C2_MANIFEST = REPO_ROOT / "ai-training/experiments/r4_4c2_bigru_ctc_seed42/artifact_hashes.json"
R4C2_DATASET = REPO_ROOT / "ai-training/experiments/r4_4c2_bigru_ctc_seed42/dataset_summary.json"
R4C2_SELECTED = REPO_ROOT / "ai-training/experiments/r4_4c2_bigru_ctc_seed42/selected_checkpoint.json"
R4C2_TRAINING = REPO_ROOT / "ai-training/experiments/r4_4c2_bigru_ctc_seed42/training_config.json"
R4A_ADDITION = REPO_ROOT / "ai-training/experiments/r4_4a_ctc_sequence_feasibility/addition_audit.json"
R4_CLOSURE_MANIFEST = REPO_ROOT / "ai-training/experiments/r4_deletion_research_closure/artifact_hashes.json"
R4_MFA_RUNTIME = REPO_ROOT / "ai-training/experiments/r4_2c_mfa_missing_phone_anchor/runtime_alignment_path.json"
R4_MFA_SEMANTICS = REPO_ROOT / "ai-training/experiments/r4_2c_mfa_missing_phone_anchor/manual_vs_mfa_semantics.json"
R4_MFA_STATUS = REPO_ROOT / "ai-training/experiments/r4_2c_mfa_missing_phone_anchor/final_status.json"
MFA_ALIGNER_SOURCE = REPO_ROOT / "ai-worker/app/alignment/mfa_aligner.py"
MFA_PARSER_SOURCE = REPO_ROOT / "ai-worker/app/alignment/textgrid_parser.py"
PREREG_PATH = EXPERIMENT_DIR / "R5_0_PREREGISTRATION.json"

EXPECTED = {
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "checkpoint": "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085",
    "r4_closure_manifest": "3E21936C6175F9DEA5FAE3346E96EECD3AEF18732072C42B0B2FCAE4161D6174",
    "preregistration": "14CBFADAC0BE35D53C01DC966A030A71F24EC4D86EA88384BC449544CE12AEF7",
}
TRAIN = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
VALIDATION = ("ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI")
TEST = ("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK")
TRAIN_SET, VALIDATION_SET, TEST_SET = map(frozenset, (TRAIN, VALIDATION, TEST))
PHONE_VOCAB = tuple(r3.PHONE_VOCAB)
PHONE_TO_ID = dict(r3.PHONE_TO_ID)
ELIGIBLE_RELATIONS = frozenset(("correct", "substitution", "deletion", "addition"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * precision * recall, precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def binary_word_metrics(truth: list[bool], predicted: list[bool]) -> dict[str, Any]:
    tp = sum(t and p for t, p in zip(truth, predicted))
    fp = sum((not t) and p for t, p in zip(truth, predicted))
    fn = sum(t and (not p) for t, p in zip(truth, predicted))
    tn = sum((not t) and (not p) for t, p in zip(truth, predicted))
    positive = prf(tp, fp, fn)
    negative = prf(tn, fn, fp)
    tpr, tnr = positive["recall"], negative["recall"]
    return {
        "words": len(truth), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": positive["precision"], "recall": positive["recall"], "f1": positive["f1"],
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "binary_macro_f1": (positive["f1"] + negative["f1"]) / 2.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def verify_sources() -> dict[str, Any]:
    paths = {
        "v4": V4_PATH,
        "checkpoint": CHECKPOINT_PATH,
        "r4_closure_manifest": R4_CLOSURE_MANIFEST,
        "preregistration": PREREG_PATH,
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    mismatch = {
        name: {"expected": EXPECTED[name], "actual": actual[name]}
        for name in EXPECTED if actual[name] != EXPECTED[name]
    }
    if mismatch:
        raise RuntimeError(f"R5_0_BLOCKED_SOURCE_IDENTITY: {mismatch}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    checkpoint_contract = {
        "epoch": int(checkpoint["epoch"]),
        "validation_per": float(checkpoint["validation_per"]),
        "blank_index": int(checkpoint["blank_index"]),
        "vocabulary": list(checkpoint["vocabulary"]),
    }
    if checkpoint_contract["blank_index"] != 40 or tuple(checkpoint_contract["vocabulary"]) != PHONE_VOCAB:
        raise RuntimeError("R5_0_BLOCKED_SOURCE_IDENTITY: checkpoint vocabulary/blank mismatch")
    return {
        "status": "PASS", "paths": {name: str(path) for name, path in paths.items()},
        "expected_sha256": EXPECTED, "actual_sha256": actual,
        "bytes": {name: paths[name].stat().st_size for name in paths},
        "checkpoint_contract": checkpoint_contract,
        "r4_modified": False,
    }


def scan_source() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    relation_global: Counter[str] = Counter()
    relation_split: dict[str, Counter[str]] = {name: Counter() for name in ("train", "validation", "test")}
    clean_addition_speakers: dict[str, Counter[str]] = {name: Counter() for name in ("train", "validation", "test")}
    detail_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tagged_addition_non_test: Counter[str] = Counter()
    with V4_PATH.open(encoding="utf-8", newline="") as handle:
        for source_index, source in enumerate(csv.DictReader(handle)):
            speaker = source["speaker_id"]
            relation = source["relation"]
            if speaker in TRAIN_SET:
                split = "train"
            elif speaker in VALIDATION_SET:
                split = "validation"
            elif speaker in TEST_SET:
                split = "test"
            else:
                raise RuntimeError(f"Unexpected speaker: {speaker}")
            # TEST policy: after speaker and frozen relation, do not inspect any other source field.
            relation_global[relation] += 1
            relation_split[split][relation] += 1
            if relation == "addition":
                clean_addition_speakers[split][speaker] += 1
            if split == "test":
                continue
            tagged = source["tagged_relation"]
            if tagged == "addition":
                tagged_addition_non_test[source.get("exclusion_reason") or "resolved_clean"] += 1
            detail_rows[(speaker, source["utterance_id"])].append({
                "source_index": source_index,
                "speaker_id": speaker,
                "utterance_id": source["utterance_id"],
                "interval_index": int(source["interval_index"]),
                "start": float(source["start_time"]),
                "end": float(source["end_time"]),
                "raw_label": source["raw_label"],
                "tagged_relation": tagged,
                "relation": relation,
                "expected": source["expected_phone_canonical"],
                "observed": source["observed_phone_canonical"],
                "label_quality": source["label_quality"],
                "exclusion_reason": source["exclusion_reason"],
                "research_subset": source["research_subset"],
                "is_addition_audit": source["is_addition_audit"],
            })
    v4_audit = json.loads(V4_AUDIT_PATH.read_text(encoding="utf-8"))
    source_summary = {
        "source_rows": sum(relation_global.values()),
        "relation_counts_global": dict(relation_global),
        "eligible_relation_observations_global": sum(relation_global[name] for name in ELIGIBLE_RELATIONS),
        "resolved_clean_additions_global": relation_global["addition"],
        "tagged_addition_annotations_global": (
            int(v4_audit["relation_counts"]["addition"])
            + int(v4_audit["unresolved_by_tagged_relation"]["addition"])
        ),
        "excluded_or_ambiguous_tagged_additions_global": int(v4_audit["unresolved_by_tagged_relation"]["addition"]),
        "excluded_reason_global": {"addition_invalid_observed_phone": int(v4_audit["unresolved_reasons"]["addition_invalid_observed_phone"])},
        "global_tagged_counts_source": "pre-existing frozen V4 audit aggregate; no TEST detail inspected",
        "speakers_with_clean_addition_global": sum(len(counts) for counts in clean_addition_speakers.values()),
        "split_relation_counts": {split: dict(values) for split, values in relation_split.items()},
        "clean_addition_counts_by_split": {split: sum(values.values()) for split, values in clean_addition_speakers.items()},
        "clean_addition_speaker_counts_internal": clean_addition_speakers,
        "tagged_addition_non_test_by_resolution": dict(tagged_addition_non_test),
        "test_fields_inspected": ["speaker_id", "relation"],
        "test_detail_exported": False,
    }
    return source_summary, detail_rows, v4_audit


def map_additions(
    detail_rows: dict[tuple[str, str], list[dict[str, Any]]], audio_root: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    events_by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
    position_counts = {"train": Counter(), "validation": Counter()}
    phone_counts = {"train": Counter(), "validation": Counter()}
    word_counts = {"train": Counter(), "validation": Counter()}
    addition_words: dict[str, Counter[int]] = {"train": Counter(), "validation": Counter()}
    mixed_counts = {"train": Counter(), "validation": Counter()}
    mapping = {"train": Counter(), "validation": Counter()}
    for (speaker, utterance), rows in sorted(detail_rows.items()):
        split = "train" if speaker in TRAIN_SET else "validation"
        if not any(row["relation"] == "addition" for row in rows):
            continue
        manual_path = audio_root / speaker / "annotation" / f"{utterance}.TextGrid"
        manual = r42c.parse_textgrid(manual_path)
        rows_by_word: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            word_index = r42c.containing_word(manual["words"], row["start"], row["end"])
            if word_index is not None:
                rows_by_word[word_index].append(row)
            elif row["relation"] == "addition":
                mapping[split]["clean_addition_without_manual_word"] += 1
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
            word_id = f"{speaker}/{utterance}/{word_index}"
            relation_set = {row["relation"] for row in ordered if row["relation"] in ELIGIBLE_RELATIONS}
            mixed = bool(relation_set & {"substitution", "deletion"})
            if mixed:
                mixed_counts[split]["addition_words_with_substitution_or_deletion"] += 1
            addition_words[split][len(additions)] += 1
            for addition in additions:
                phone_counts[split][addition["observed"]] += 1
                word_label = manual["words"][word_index]["label"]
                word_counts[split][word_label.casefold()] += 1
                if not expected_rows:
                    mapping[split]["clean_addition_without_expected_sequence"] += 1
                    continue
                boundary = sum(row["source_index"] < addition["source_index"] for row in expected_rows)
                n = len(expected_rows)
                if boundary == 0:
                    position = "BEFORE_FIRST"
                elif boundary == n:
                    position = "AFTER_FINAL"
                else:
                    position = "BETWEEN"
                event = {
                    "word_id": word_id, "speaker_id": speaker, "utterance_id": utterance,
                    "manual_word_index": word_index, "word": word_label,
                    "source_index": addition["source_index"], "phone": addition["observed"],
                    "boundary": boundary, "expected_length": n, "position": position,
                    "mixed_error_word": mixed,
                }
                events_by_word[word_id].append(event)
                position_counts[split][position] += 1
                mapping[split]["position_mapped"] += 1
    report = {
        "mapping": {split: dict(values) for split, values in mapping.items()},
        "position_counts": {split: dict(values) for split, values in position_counts.items()},
        "phone_counts": {split: dict(sorted(values.items())) for split, values in phone_counts.items()},
        "word_counts": {split: dict(values.most_common()) for split, values in word_counts.items()},
        "addition_word_multiplicity": {
            split: {str(k): v for k, v in sorted(values.items())} for split, values in addition_words.items()
        },
        "mixed_error_counts": {split: dict(values) for split, values in mixed_counts.items()},
    }
    return events_by_word, report


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
    return words, {
        "candidate_records": len(records), "eligible_r5_words": len(words),
        "addition_positive_words": sum(bool(word["true_addition_events"]) for word in words),
        "mapped_true_events": sum(len(word["true_addition_events"]) for word in words),
        "reconstruction": reconstruction,
    }


def infer_train(words: list[dict[str, Any]], device: torch.device) -> tuple[list[list[int]], dict[str, Any]]:
    started = time.perf_counter()
    features, feature_report = r4b.materialize_features(words, device, "r5_train_only")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = r4c2.WordBiGRUCTCModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    decoded: list[list[int] | None] = [None] * len(words)
    inference_started = time.perf_counter()
    with torch.no_grad():
        for batch_number, indexes in enumerate(r4b.make_evaluation_batches([item.shape[-1] for item in features]), start=1):
            maximum = max(features[index].shape[-1] for index in indexes)
            batch = torch.zeros((len(indexes), 1, r4b.N_MELS, maximum), dtype=torch.float32)
            frame_lengths = []
            for position, index in enumerate(indexes):
                item = features[index]
                batch[position, 0, :, : item.shape[-1]] = item
                frame_lengths.append(item.shape[-1])
            frame_tensor = torch.tensor(frame_lengths, dtype=torch.long, device=device)
            logits, output_lengths = model(batch.to(device, non_blocking=True), frame_tensor)
            batch_decoded = r4b.greedy_decode(logits, output_lengths)
            for index, sequence in zip(indexes, batch_decoded):
                decoded[index] = sequence
            if batch_number % 250 == 0:
                print(f"r5_train_inference batches={batch_number}", flush=True)
    if any(value is None for value in decoded):
        raise RuntimeError("Incomplete TRAIN inference")
    inference_seconds = time.perf_counter() - inference_started
    return [list(value) for value in decoded if value is not None], {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "feature_report": feature_report,
        "feature_seconds": feature_report["seconds"],
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "words_per_second": ratio(len(words), time.perf_counter() - started),
        "training": False, "validation_inference": False, "test_inference": False,
    }


def insertion_events(reference: list[int], hypothesis: list[int]) -> list[dict[str, Any]]:
    operations = r4b.sequence_alignment(reference, hypothesis)
    cursor = 0
    events: list[dict[str, Any]] = []
    for operation in operations:
        name = operation["operation"]
        if name == "INSERT_IN_DECODED":
            hyp_index = int(operation["hypothesis_index"])
            boundary = cursor
            n = len(reference)
            position = "BEFORE_FIRST" if boundary == 0 else ("AFTER_FINAL" if boundary == n else "BETWEEN")
            events.append({"phone": PHONE_VOCAB[hypothesis[hyp_index]], "boundary": boundary, "position": position})
        else:
            cursor += 1
    return events


def event_metrics(true_events: list[dict[str, Any]], predicted_events: list[dict[str, Any]]) -> dict[str, Any]:
    true_counter = Counter((event.get("word_id"), event["boundary"], event["phone"]) for event in true_events)
    pred_counter = Counter((event.get("word_id"), event["boundary"], event["phone"]) for event in predicted_events)
    tp = sum((true_counter & pred_counter).values())
    return {**prf(tp, sum(pred_counter.values()) - tp, sum(true_counter.values()) - tp),
            "true_events": sum(true_counter.values()), "predicted_events": sum(pred_counter.values())}


def run_ctc_audit(words: list[dict[str, Any]], decoded: list[list[int]]) -> tuple[dict[str, Any], dict[str, Any]]:
    truth_word: list[bool] = []
    predicted_word: list[bool] = []
    all_true: list[dict[str, Any]] = []
    all_pred: list[dict[str, Any]] = []
    exact_by_position: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"true": [], "pred": []} for name in ("BEFORE_FIRST", "BETWEEN", "AFTER_FINAL")
    }
    true_phone = Counter()
    matched_phone = Counter()
    hallucinated = Counter()
    correct_words = 0
    correct_positive = 0
    correct_insertions = 0
    addition_words = 0
    addition_positive = 0
    word_rows: list[dict[str, Any]] = []
    for word, hypothesis in zip(words, decoded):
        true_events = word["true_addition_events"]
        predicted_events = insertion_events(word["expected_ids"], hypothesis)
        for event in predicted_events:
            event["word_id"] = word["word_id"]
        is_true = bool(true_events)
        is_predicted = bool(predicted_events)
        truth_word.append(is_true); predicted_word.append(is_predicted)
        all_true.extend(true_events); all_pred.extend(predicted_events)
        for position in exact_by_position:
            exact_by_position[position]["true"].extend(event for event in true_events if event["position"] == position)
            exact_by_position[position]["pred"].extend(event for event in predicted_events if event["position"] == position)
        true_counter = Counter((event["boundary"], event["phone"]) for event in true_events)
        pred_counter = Counter((event["boundary"], event["phone"]) for event in predicted_events)
        matched = true_counter & pred_counter
        for (_, phone), count in true_counter.items():
            true_phone[phone] += count
        for (_, phone), count in matched.items():
            matched_phone[phone] += count
        for event in predicted_events:
            hallucinated[event["phone"]] += 1
        if is_true:
            addition_words += 1; addition_positive += is_predicted
        relations = Counter(word["relations"])
        is_correct_only = (
            not is_true and word["substitution"] == 0 and word["deletion"] == 0
            and all(name in {"correct", "non_speech"} or count == 0 for name, count in relations.items())
        )
        if is_correct_only:
            correct_words += 1; correct_positive += is_predicted; correct_insertions += len(predicted_events)
        word_rows.append({
            "word_id": word["word_id"], "speaker_id": word["speaker_id"],
            "truth_addition": is_true, "predicted_addition": is_predicted,
            "true_events": len(true_events), "predicted_events": len(predicted_events),
        })
    word_metrics = binary_word_metrics(truth_word, predicted_word)
    exact = event_metrics(all_true, all_pred)
    position = {
        name: event_metrics(values["true"], values["pred"])
        for name, values in exact_by_position.items()
    }
    phone = {
        name: {"support": true_phone[name], "exact_matches": matched_phone[name], "exact_match_recall": ratio(matched_phone[name], true_phone[name])}
        for name in sorted(true_phone)
    }
    correct_rate = ratio(correct_positive, correct_words)
    addition_rate = ratio(addition_positive, addition_words)
    baseline = {
        "scope": "TRAIN only", "score_or_threshold_tuned": False,
        "eligible_words": len(words), "word_level": word_metrics,
        "exact_insertion_event": exact, "position_exact_event": position,
        "phone_exact_recall": phone,
        "alignment": "unit-cost Levenshtein with frozen MATCH > SUBSTITUTION > DELETE > INSERT backtrace",
        "true_event_mapping_coverage": {"events_in_baseline_words": len(all_true)},
    }
    hallucination = {
        "scope": "TRAIN correct-only words",
        "correct_only_words": correct_words,
        "words_with_at_least_one_greedy_insert": correct_positive,
        "false_insertion_word_rate": correct_rate,
        "insertion_edits": correct_insertions,
        "insertion_edits_per_word": ratio(correct_insertions, correct_words),
        "most_common_hallucinated_inserted_phones_all_train": hallucinated.most_common(20),
        "true_addition_words": addition_words,
        "greedy_insertion_positive_true_addition_words": addition_positive,
        "greedy_insertion_positive_rate_true_addition_words": addition_rate,
        "greedy_insertion_positive_rate_correct_only_words": correct_rate,
        "ADDITION_INSERTION_RATE_DELTA": addition_rate - correct_rate,
        "directional_signal": addition_rate - correct_rate > 0,
    }
    return baseline, hallucination


def support_and_distribution(source: dict[str, Any], mapping: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    split_counts = source["clean_addition_counts_by_split"]
    speaker_counts = source.pop("clean_addition_speaker_counts_internal")
    positions = mapping["position_counts"]
    support = {}
    for split in ("train", "validation", "test"):
        counts = speaker_counts[split]
        support[split] = {
            "clean_additions": split_counts[split],
            "speakers_with_at_least_1": sum(value >= 1 for value in counts.values()),
            "speakers_with_at_least_5": sum(value >= 5 for value in counts.values()),
        }
        if split != "test":
            support[split]["per_speaker"] = {speaker: counts.get(speaker, 0) for speaker in (TRAIN if split == "train" else VALIDATION)}
            support[split]["position_counts"] = positions[split]
    eligible = int(source["eligible_relation_observations_global"])
    additions = int(source["resolved_clean_additions_global"])
    global_relations = source["relation_counts_global"]
    prevalence = ratio(additions, eligible)
    correct_ratio = ratio(global_relations.get("correct", 0), additions)
    imbalance = "not severe" if correct_ratio <= 10 else ("severe" if correct_ratio <= 100 else "extremely severe")
    data_support = {
        "global": {
            "eligible_relation_observations": eligible,
            "total_tagged_addition_annotations": source["tagged_addition_annotations_global"],
            "clean_additions": additions,
            "excluded_or_ambiguous_additions": source["excluded_or_ambiguous_tagged_additions_global"],
            "excluded_reasons": source["excluded_reason_global"],
            "addition_prevalence": prevalence,
            "speakers_with_at_least_1_clean_addition": source["speakers_with_clean_addition_global"],
        },
        "splits": support,
        "class_imbalance": {
            "classification": imbalance,
            "correct_to_addition": ratio(global_relations.get("correct", 0), additions),
            "substitution_to_addition": ratio(global_relations.get("substitution", 0), additions),
            "deletion_to_addition": ratio(global_relations.get("deletion", 0), additions),
            "exact_relation_counts": {name: global_relations.get(name, 0) for name in ("correct", "substitution", "deletion", "addition")},
        },
        "test_detail_policy": "Only aggregate clean support and aggregate speaker-support counts are exported.",
    }
    train_distribution = {
        "clean_addition_count": support["train"]["clean_additions"],
        "speakers_with_additions": support["train"]["speakers_with_at_least_1"],
        "per_speaker": support["train"]["per_speaker"],
        "by_added_phone": mapping["phone_counts"]["train"],
        "by_word": mapping["word_counts"]["train"],
        "by_insertion_position": positions["train"],
        "addition_word_multiplicity": mapping["addition_word_multiplicity"]["train"],
        "mixed_error_counts": mapping["mixed_error_counts"]["train"],
        "addition_prevalence_global_contract": prevalence,
    }
    phone_items = sorted(train_distribution["by_added_phone"].items(), key=lambda item: (item[1], item[0]))
    train_distribution["bottom_supported_phones"] = phone_items[:10]
    train_distribution["top_supported_phones"] = list(reversed(phone_items[-10:]))
    return data_support, train_distribution, support


def evaluate_gates(support: dict[str, Any], hallucination: dict[str, Any], reuse_class: str) -> tuple[dict[str, Any], str]:
    train_classes = sum(value > 0 for value in support["train"]["position_counts"].values())
    gates = {
        "TRAIN": {
            "clean_additions_ge_100": support["train"]["clean_additions"] >= 100,
            "speakers_ge_5_at_least_6": support["train"]["speakers_with_at_least_5"] >= 6,
            "position_classes_at_least_2": train_classes >= 2,
        },
        "VALIDATION": {
            "clean_additions_ge_43": support["validation"]["clean_additions"] >= 43,
            "speakers_ge_5_at_least_3": support["validation"]["speakers_with_at_least_5"] >= 3,
        },
        "TEST": {
            "clean_additions_ge_43": support["test"]["clean_additions"] >= 43,
            "speakers_ge_5_at_least_3": support["test"]["speakers_with_at_least_5"] >= 3,
        },
    }
    for values in gates.values():
        values["PASS"] = all(values.values())
    all_support = all(values["PASS"] for values in gates.values())
    runtime_viable = True  # deterministic CTC insertion phone+boundary representation was computed
    directional = bool(hallucination["directional_signal"])
    reuse_justified = reuse_class in {
        "R5_CTC_REUSE_TRAIN_DIAGNOSTIC_ONLY",
        "R5_CTC_REUSE_DEVELOPMENT_ALLOWED_CONFIRMATION_REQUIRES_NEW_PROTOCOL",
        "R5_CTC_REUSE_CONFIRMATION_ACCEPTABLE",
    }
    if not all_support:
        status = "R5_0_BLOCKED_INSUFFICIENT_DATA"
    elif not runtime_viable:
        status = "R5_0_BLOCKED_RUNTIME_REPRESENTATION"
    elif directional and reuse_justified:
        status = "R5_0_PASS_EXISTING_CTC_FEASIBLE"
    else:
        status = "R5_0_PASS_DATA_FEASIBLE_CTC_WEAK"
    return {
        "support_gates": gates, "all_data_support_gates_pass": all_support,
        "ctc_sequence_representation_viable": runtime_viable,
        "directional_ctc_insertion_signal": directional,
        "reuse_justified_for_next_development_stage": reuse_justified,
        "final_status": status,
    }, status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", required=True, type=Path)
    args = parser.parse_args()
    audio_root = args.audio_root.resolve()
    if audio_root.name.casefold() != "l2arctic_release_v5.0" or not audio_root.is_dir():
        raise RuntimeError("Invalid L2-ARCTIC root")
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started = time.perf_counter()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    source_identity = verify_sources()
    source_summary, detail_rows, v4_audit = scan_source()
    events_by_word, mapping = map_additions(detail_rows, audio_root)
    data_support, train_distribution, split_support = support_and_distribution(source_summary, mapping)

    annotation = {
        "status": "UNAMBIGUOUS",
        "source_fields": ["raw_label", "tagged_relation", "relation", "expected_phone_raw", "observed_phone_raw", "expected_phone_canonical", "observed_phone_canonical", "start_time", "end_time", "label_quality", "exclusion_reason", "research_subset"],
        "representation": "phones-tier interval raw triplet sil,<observed_phone>,a; resolved relation addition; expected canonical placeholder <SIL>; observed canonical is the added phone",
        "has_own_interval": True, "timestamps_present": True,
        "expected_phone_identity_present": "placeholder sil/<SIL>, not a canonical expected phone",
        "observed_phone_identity_present": True,
        "multiple_additions_supported": True, "mixed_with_substitution_or_deletion_supported": True,
        "clean_rule": json.loads(PREREG_PATH.read_text(encoding="utf-8"))["clean_addition_definition"],
        "malformed_unknown": {
            "global_tagged_addition_annotations": data_support["global"]["total_tagged_addition_annotations"],
            "global_clean_additions": data_support["global"]["clean_additions"],
            "addition_invalid_observed_phone": data_support["global"]["excluded_reasons"]["addition_invalid_observed_phone"],
        },
        "position_contract": json.loads(PREREG_PATH.read_text(encoding="utf-8"))["addition_position_contract"],
        "position_mapping": mapping["mapping"],
    }

    words, word_reconstruction = build_train_words(audio_root, events_by_word)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoded, compute = infer_train(words, device)
    ctc_baseline, hallucination = run_ctc_audit(words, decoded)

    mfa_runtime = {
        "explicit_extra_phone_localization_supported": False,
        "empirical_rerun_required": False,
        "empirical_rerun_performed": False,
        "evidence": [
            "Phoenix runtime writes the canonical prompt transcript to MFA as a .lab file and invokes transcript-constrained forced alignment.",
            "The runtime API supplies prompt text and does not supply observed/addition phone labels; no code path requests arbitrary extra-phone hypotheses.",
            "The TextGrid parser only returns intervals emitted by MFA and cannot create an additional phone label.",
            "Frozen R4-2C evidence classified MFA behavior as C_FORCED_ALIGNMENT_RETAINS_EXPECTED_LABEL; 2454 deletion slots retained the expected label and only 10 were missing without a clean anchor."
        ],
        "normal_runtime_output": "canonical-transcript alignment intervals; extra acoustic material can be absorbed into expected/silence spans or boundary placement, but is not exposed as a labeled extra-phone slot",
        "limitation": "MFA can supply a word span and expected-phone timing context, but not a reliable explicit addition interval or added-phone identity under the current runtime contract.",
        "ctc_alternative": "The frozen CTC decoded sequence independently represents insertions by canonical phone identity and expected-sequence boundary.",
        "sources": {
            "runtime_aligner": str(MFA_ALIGNER_SOURCE), "runtime_parser": str(MFA_PARSER_SOURCE),
            "r4_runtime_path": str(R4_MFA_RUNTIME), "r4_semantics": str(R4_MFA_SEMANTICS), "r4_status": str(R4_MFA_STATUS),
        },
        "test_mfa_run": False, "validation_mfa_run_for_model_evaluation": False,
    }

    r4_dataset = json.loads(R4C2_DATASET.read_text(encoding="utf-8"))
    r4_selected = json.loads(R4C2_SELECTED.read_text(encoding="utf-8"))
    r4_addition = json.loads(R4A_ADDITION.read_text(encoding="utf-8"))
    reuse_class = "R5_CTC_REUSE_DEVELOPMENT_ALLOWED_CONFIRMATION_REQUIRES_NEW_PROTOCOL"
    reuse = {
        "classification": reuse_class,
        "checkpoint_train_speakers": list(TRAIN),
        "addition_containing_words_excluded_from_r4": True,
        "r4_addition_word_counts": r4_addition["word_counts"],
        "r5_addition_labels_directly_used_in_r4_training": False,
        "r4_validation_speakers_influenced_checkpoint_selection": True,
        "checkpoint_selection": r4_selected["selection"],
        "selected_epoch": r4_selected["selected_epoch"],
        "selected_validation_metric": "lowest validation PER, then deletion F1, then earlier epoch",
        "test_used": False,
        "train_only_r5_diagnostic_reuse": "scientifically acceptable because the checkpoint is frozen, TRAIN speakers are its development population, and addition-containing words/labels were excluded from R4 training",
        "future_current_validation_use": "acceptable only as iterative development, not clean confirmatory evidence, because the same validation speakers selected epoch 35",
        "clean_confirmation_requirement": "a new preregistered protocol with an independent untouched evaluation population and leakage controls; the untouched R4/R5 TEST must remain closed until a development candidate is frozen",
        "reason": "No addition labels entered training, so diagnostic reuse is informative; however validation-based checkpoint selection prevents treating future results on the same validation speakers as independent confirmation.",
        "provenance_paths": [str(R4C2_DATASET), str(R4C2_SELECTED), str(R4C2_TRAINING), str(R4A_ADDITION)],
    }

    gates, status = evaluate_gates(split_support, hallucination, reuse_class)
    final_status = {
        "status": status,
        "training": False,
        "threshold_tuning": False,
        "validation_performance_consumed": False,
        "validation_model_inference": False,
        "test_audio_accessed": False,
        "test_paths_resolved": False,
        "test_inference_run": False,
        "test_performance_consumed": False,
        "r4_modified": False,
        "r5_1_started": False,
        "next_action": (
            "Freeze a separate R5 development-stage insertion scorer/detector contract using TRAIN only; do not implement or evaluate it in R5-0."
            if status == "R5_0_PASS_DATA_FEASIBLE_CTC_WEAK" else
            "Freeze an R5-1 development contract for the frozen CTC insertion representation, with independent confirmation protocol explicitly separated; do not implement it in R5-0."
            if status == "R5_0_PASS_EXISTING_CTC_FEASIBLE" else
            "Stop addition-model development and resolve the recorded blocking condition before any new R5 authorization."
        ),
    }

    source_identity.update({
        "v4_audit_path": str(V4_AUDIT_PATH), "v4_audit_sha256": sha256(V4_AUDIT_PATH),
        "branch": os.popen(f'git -C "{REPO_ROOT}" branch --show-current').read().strip(),
        "audio_root_used_for_train_validation_data_and_train_inference": str(audio_root),
        "test_audio_root_paths_constructed": False,
    })
    write_json(EXPERIMENT_DIR / "r5_0_source_identity.json", source_identity)
    write_json(EXPERIMENT_DIR / "r5_0_annotation_semantics.json", annotation)
    write_json(EXPERIMENT_DIR / "r5_0_data_support.json", data_support)
    write_json(EXPERIMENT_DIR / "r5_0_addition_distribution_train.json", train_distribution)
    write_json(EXPERIMENT_DIR / "r5_0_split_support.json", split_support)
    ctc_baseline["word_reconstruction"] = word_reconstruction
    ctc_baseline["compute"] = compute
    write_json(EXPERIMENT_DIR / "r5_0_ctc_greedy_train_baseline.json", ctc_baseline)
    write_json(EXPERIMENT_DIR / "r5_0_ctc_hallucination_audit.json", hallucination)
    write_json(EXPERIMENT_DIR / "r5_0_mfa_runtime_audit.json", mfa_runtime)
    write_json(EXPERIMENT_DIR / "r5_0_ctc_reuse_provenance.json", reuse)
    write_json(EXPERIMENT_DIR / "r5_0_feasibility_gates.json", gates)
    write_json(EXPERIMENT_DIR / "r5_0_final_status.json", final_status)

    train_positions = data_support["splits"]["train"]["position_counts"]
    report = f"""# R5-0 Addition Data & Runtime Feasibility Audit

Research-only feasibility audit. No neural training, threshold tuning, VALIDATION model inference, or TEST audio/inference occurred.

## Frozen identities

- V4: `{V4_PATH.relative_to(REPO_ROOT)}` — `{EXPECTED['v4']}`
- R4-4C2 checkpoint: `{CHECKPOINT_PATH.relative_to(REPO_ROOT)}` — `{EXPECTED['checkpoint']}`
- R5-0 preregistration: `{EXPECTED['preregistration']}`
- R4 closure manifest: `{EXPECTED['r4_closure_manifest']}`

## Annotation semantics

An addition is a dedicated manual `phones` IntervalTier row encoded as `sil,<observed_phone>,a`. `sil` is a placeholder rather than an expected canonical phone; the observed canonical label is the added phone. The interval has start/end timestamps. Multiple additions and additions mixed with substitution/deletion are retained. Malformed added-phone labels remain unresolved and are reported, not silently repaired.

The insertion boundary is the number of expected-sequence phones preceding the addition interval: 0 is BEFORE_FIRST, N is AFTER_FINAL, and an interior boundary is BETWEEN.

## Data support

- Global clean additions: **{data_support['global']['clean_additions']}** / {data_support['global']['eligible_relation_observations']} eligible relation rows ({data_support['global']['addition_prevalence']:.6%}).
- Tagged addition annotations: {data_support['global']['total_tagged_addition_annotations']}; excluded/invalid: {data_support['global']['excluded_or_ambiguous_additions']}.
- TRAIN: {split_support['train']['clean_additions']} additions; {split_support['train']['speakers_with_at_least_1']} speakers with additions; {split_support['train']['speakers_with_at_least_5']} speakers with >=5.
- VALIDATION support only: {split_support['validation']['clean_additions']} additions; {split_support['validation']['speakers_with_at_least_1']} speakers with additions; {split_support['validation']['speakers_with_at_least_5']} speakers with >=5.
- TEST aggregate support only: {split_support['test']['clean_additions']} additions; {split_support['test']['speakers_with_at_least_1']} speakers with additions; {split_support['test']['speakers_with_at_least_5']} speakers with >=5.
- TRAIN mapped positions: BEFORE_FIRST={train_positions.get('BEFORE_FIRST', 0)}, BETWEEN={train_positions.get('BETWEEN', 0)}, AFTER_FINAL={train_positions.get('AFTER_FINAL', 0)}.
- Imbalance: **{data_support['class_imbalance']['classification']}**; correct:addition={data_support['class_imbalance']['correct_to_addition']:.3f}:1, substitution:addition={data_support['class_imbalance']['substitution_to_addition']:.3f}:1, deletion:addition={data_support['class_imbalance']['deletion_to_addition']:.3f}:1.

## Feasibility gates

- TRAIN support: **{'PASS' if gates['support_gates']['TRAIN']['PASS'] else 'FAIL'}**
- VALIDATION support: **{'PASS' if gates['support_gates']['VALIDATION']['PASS'] else 'FAIL'}**
- TEST aggregate support: **{'PASS' if gates['support_gates']['TEST']['PASS'] else 'FAIL'}**

## Frozen TRAIN CTC greedy insertion baseline

- Word precision/recall/F1: {ctc_baseline['word_level']['precision']:.6f} / {ctc_baseline['word_level']['recall']:.6f} / {ctc_baseline['word_level']['f1']:.6f}
- Binary Macro-F1: {ctc_baseline['word_level']['binary_macro_f1']:.6f}
- Exact event precision/recall/F1: {ctc_baseline['exact_insertion_event']['precision']:.6f} / {ctc_baseline['exact_insertion_event']['recall']:.6f} / {ctc_baseline['exact_insertion_event']['f1']:.6f}
- Correct-only false insertion word rate: {hallucination['false_insertion_word_rate']:.6f}
- ADDITION_INSERTION_RATE_DELTA: {hallucination['ADDITION_INSERTION_RATE_DELTA']:.6f}

## Runtime MFA and CTC provenance

Current runtime MFA does not expose an explicit arbitrary extra-phone slot: it aligns the canonical prompt transcript, and its parser only returns MFA-emitted intervals. It may still provide word context, but explicit added-phone identity/location requires a separate acoustic sequence mechanism. The frozen CTC insertion alignment does provide a deterministic phone-plus-boundary representation.

Reuse classification: **{reuse_class}**. R4 excluded addition-containing words and did not use addition labels, so TRAIN-only diagnostic reuse is acceptable. Epoch 35 was nevertheless selected on these VALIDATION speakers by PER, so future work on the same VALIDATION split is iterative development rather than independent confirmation.

## Final decision

**{status}**

This is a feasibility outcome, not confirmation that Phoenix detects additions.

## Protocol closure

- Training: NO
- Threshold tuning: NO
- VALIDATION performance consumed: NO
- TEST audio accessed: NO
- TEST inference: NO
- TEST performance consumed: NO
- R4 modified: NO
"""
    (EXPERIMENT_DIR / "R5_0_ADDITION_FEASIBILITY_REPORT.md").write_text(report, encoding="utf-8")

    completed_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    compute.update({
        "started_at": started_wall, "completed_at": completed_wall,
        "wall_seconds": time.perf_counter() - started,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
    })
    write_json(EXPERIMENT_DIR / "r5_0_compute_report.json", compute)

    files = sorted(path for path in EXPERIMENT_DIR.iterdir() if path.is_file() and path.name != "artifact_hashes.json")
    entries = [{
        "relative_path": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)
    } for path in files]
    manifest = {
        "schema_version": 1, "artifact_count": len(entries), "artifacts": entries,
        "self_excluded": "artifact_hashes.json cannot contain a stable hash of itself",
        "verification": "PENDING_REOPEN",
    }
    write_json(EXPERIMENT_DIR / "artifact_hashes.json", manifest)
    reopened = json.loads((EXPERIMENT_DIR / "artifact_hashes.json").read_text(encoding="utf-8"))
    failures = []
    for entry in reopened["artifacts"]:
        path = EXPERIMENT_DIR / entry["relative_path"]
        if path.stat().st_size != entry["byte_size"] or sha256(path) != entry["sha256"]:
            failures.append(entry["relative_path"])
    manifest["verification"] = "HASH_AUDIT_PASS" if not failures else "HASH_AUDIT_FAIL"
    manifest["failures"] = failures
    write_json(EXPERIMENT_DIR / "artifact_hashes.json", manifest)
    print(json.dumps({
        "status": status, "preregistration_sha256": EXPECTED["preregistration"],
        "data_support": {split: split_support[split]["clean_additions"] for split in split_support},
        "word_f1": ctc_baseline["word_level"]["f1"], "event_f1": ctc_baseline["exact_insertion_event"]["f1"],
        "delta": hallucination["ADDITION_INSERTION_RATE_DELTA"], "hash_audit": manifest["verification"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
