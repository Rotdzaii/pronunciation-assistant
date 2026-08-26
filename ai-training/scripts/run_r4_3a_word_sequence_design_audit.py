from __future__ import annotations

import csv
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

import run_l2_arctic_observed_phone_r3_1a as r3  # noqa: E402
import run_r4_2c_mfa_missing_phone_anchor_audit as r42c  # noqa: E402


REPO_ROOT = r3.REPO_ROOT
V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
EXPECTED_V4_SHA = "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D"
CHECKPOINT_PATH = (
    REPO_ROOT / "ai-training/experiments/r3_1d_observed_phone_seed42_48epochs/"
    "R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt"
)
EXPECTED_CHECKPOINT_SHA = "5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E"
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r4_3a_word_sequence_design_audit"
TRAIN_SPEAKERS = frozenset(r3.TRAIN_SPEAKERS)
VALIDATION_SPEAKERS = frozenset(r3.VALIDATION_SPEAKERS)
AUDIT_SPEAKERS = TRAIN_SPEAKERS | VALIDATION_SPEAKERS
STRIDES_MS = (20, 40, 50)
RECOMMENDED_STRIDE_MS = 40
MAX_ORACLE_PATHS = 4096
SEED = 42


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)), "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90)),
    }


def probe_centers(start: float, end: float, stride_seconds: float) -> list[float]:
    if end <= start:
        return [(start + end) / 2.0]
    centers = [start]
    while centers[-1] + stride_seconds < end - 1e-9:
        centers.append(centers[-1] + stride_seconds)
    if end - centers[-1] > 1e-9:
        centers.append(end)
    return centers


def oracle_alignment(expected: list[str], observed: list[str]) -> dict[str, Any]:
    n, m = len(expected), len(observed)
    cost = np.zeros((n + 1, m + 1), dtype=np.int16)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    back: list[list[list[str]]] = [[[] for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        back[i][0] = ["D"]
    for j in range(1, m + 1):
        back[0][j] = ["I"]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = int(cost[i - 1, j - 1]) + (expected[i - 1] != observed[j - 1])
            deletion = int(cost[i - 1, j]) + 1
            insertion = int(cost[i, j - 1]) + 1
            best = min(diagonal, deletion, insertion)
            cost[i, j] = best
            if diagonal == best:
                back[i][j].append("M" if expected[i - 1] == observed[j - 1] else "S")
            if deletion == best:
                back[i][j].append("D")
            if insertion == best:
                back[i][j].append("I")

    pending: list[tuple[list[tuple[str, int | None, int | None]], int, int]] = [([], n, m)]
    signatures: set[tuple[str, ...]] = set()
    chosen_path: list[tuple[str, int | None, int | None]] | None = None
    capped = False
    examined = 0
    while pending:
        operations, i, j = pending.pop()
        if i == 0 and j == 0:
            path = list(reversed(operations))
            if chosen_path is None:
                chosen_path = path
            signature = ["" for _ in range(n)]
            for operation, expected_index, _ in path:
                if expected_index is not None:
                    signature[expected_index] = operation
            signatures.add(tuple(signature))
            examined += 1
            if examined >= MAX_ORACLE_PATHS:
                capped = bool(pending)
                break
            continue
        for operation in reversed(back[i][j]):
            if operation in {"M", "S"}:
                pending.append((operations + [(operation, i - 1, j - 1)], i - 1, j - 1))
            elif operation == "D":
                pending.append((operations + [(operation, i - 1, None)], i - 1, j))
            else:
                pending.append((operations + [(operation, None, j - 1)], i, j - 1))
    if chosen_path is None:
        raise RuntimeError("Oracle alignment produced no path")
    chosen_signature = next(iter(signatures)) if len(signatures) == 1 else tuple(
        operation for operation, expected_index, _ in chosen_path if expected_index is not None
    )
    operation_options = [sorted({signature[index] for signature in signatures}) for index in range(n)]
    return {
        "distance": int(cost[n, m]), "chosen_signature": chosen_signature,
        "signatures": signatures, "operation_options": operation_options,
        "ambiguous": len(signatures) > 1, "paths_examined": examined, "paths_capped": capped,
    }


def compress_runs(phones: list[str], offsets_ms: list[float]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for phone, offset in zip(phones, offsets_ms):
        if runs and runs[-1]["phone"] == phone:
            runs[-1]["end_offset_ms"] = offset
            runs[-1]["probes"] += 1
        else:
            runs.append({"phone": phone, "start_offset_ms": offset, "end_offset_ms": offset, "probes": 1})
    return runs


def build_word_records(audio_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    utterance_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with V4_PATH.open(encoding="utf-8", newline="") as handle:
        for source_index, source in enumerate(csv.DictReader(handle)):
            speaker = source["speaker_id"]
            if speaker not in AUDIT_SPEAKERS:
                continue
            utterance_rows[(speaker, source["utterance_id"])].append({
                "source_index": source_index,
                "speaker_id": speaker,
                "utterance_id": source["utterance_id"],
                "audio_path": source["audio_path"],
                "start": float(source["start_time"]),
                "end": float(source["end_time"]),
                "relation": source["relation"],
                "label_quality": source["label_quality"],
                "expected": source["expected_phone_canonical"],
                "observed": source["observed_phone_canonical"],
                "raw_label": source["raw_label"],
            })

    records: list[dict[str, Any]] = []
    accounting: Counter[str] = Counter()
    invalid_aligned: Counter[str] = Counter()
    boundary_start_errors: list[float] = []
    boundary_end_errors: list[float] = []
    target_rows_without_manual_word: Counter[str] = Counter()
    for (speaker, utterance), rows in sorted(utterance_rows.items()):
        manual = r42c.parse_textgrid(audio_root / speaker / "annotation" / f"{utterance}.TextGrid")
        aligned = r42c.parse_textgrid(audio_root / speaker / "textgrid" / f"{utterance}.TextGrid")
        manual_labels = [r42c.normalize_word(word["label"]) for word in manual["words"]]
        aligned_labels = [r42c.normalize_word(word["label"]) for word in aligned["words"]]
        word_alignment = r42c.align_tokens(manual_labels, aligned_labels)
        ambiguous_manual_words = set(word_alignment["ambiguous_mapping_indices"])
        manual_to_aligned = word_alignment["mapping"]
        for phone in aligned["phones"]:
            if phone["label"] not in r42c.NON_SPEECH and r42c.canonical_phone(phone["label"]) is None:
                invalid_aligned[phone["label"]] += 1

        rows_by_word: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            word_index = r42c.containing_word(manual["words"], row["start"], row["end"])
            if word_index is None:
                if row["relation"] in {"correct", "substitution", "deletion", "addition"}:
                    target_rows_without_manual_word[row["relation"]] += 1
                continue
            rows_by_word[word_index].append(row)

        for manual_index, word_rows in sorted(rows_by_word.items()):
            expected_source_rows = [
                row for row in sorted(word_rows, key=lambda item: item["source_index"])
                if row["expected"] in r3.PHONE_TO_ID
            ]
            if not expected_source_rows:
                continue
            relations = Counter(row["relation"] for row in word_rows)
            has_unresolved = any(
                row["label_quality"] == "unresolved" or row["relation"] == "unknown"
                or (
                    row["relation"] in {"correct", "substitution", "deletion"}
                    and (
                        row["label_quality"] != "clean"
                        or row["expected"] not in r3.PHONE_TO_ID
                        or (row["relation"] in {"correct", "substitution"} and row["observed"] not in r3.PHONE_TO_ID)
                        or (row["relation"] == "deletion" and row["observed"] != "<SIL>")
                    )
                )
                for row in word_rows
            )
            has_addition = relations["addition"] > 0
            clean_rows = [
                row for row in expected_source_rows
                if row["label_quality"] == "clean" and row["relation"] in {"correct", "substitution", "deletion"}
                and (row["relation"] == "deletion" or row["observed"] in r3.PHONE_TO_ID)
            ]
            expected = [row["expected"] for row in clean_rows]
            observed: list[str] = []
            ground_operations: list[str] = []
            for row in clean_rows:
                if row["relation"] == "deletion":
                    ground_operations.append("D")
                else:
                    observed.append(row["observed"])
                    ground_operations.append("M" if row["relation"] == "correct" else "S")

            aligned_index = manual_to_aligned.get(manual_index)
            boundary_available = (
                aligned_index is not None
                and manual_index not in ambiguous_manual_words
                and 0 <= aligned_index < len(aligned["words"])
            )
            if boundary_available:
                aligned_word = aligned["words"][aligned_index]
                word_start, word_end = float(aligned_word["start"]), float(aligned_word["end"])
                boundary_start_errors.append(abs(word_start - float(manual["words"][manual_index]["start"])))
                boundary_end_errors.append(abs(word_end - float(manual["words"][manual_index]["end"])))
            else:
                word_start = word_end = None

            usable = bool(expected) and not has_unresolved and not has_addition and boundary_available
            split = "train" if speaker in TRAIN_SPEAKERS else "validation"
            record = {
                "word_id": f"{speaker}/{utterance}/{manual_index}",
                "split": split, "speaker_id": speaker, "utterance_id": utterance,
                "word": manual["words"][manual_index]["label"], "manual_word_index": manual_index,
                "manual_start": float(manual["words"][manual_index]["start"]),
                "manual_end": float(manual["words"][manual_index]["end"]),
                "mfa_start": word_start, "mfa_end": word_end, "utterance_end": float(aligned["xmax"]),
                "audio_path": r3.resolve_audio_path(expected_source_rows[0]["audio_path"], audio_root),
                "expected": expected, "observed": observed, "ground_operations": ground_operations,
                "clean_rows": clean_rows, "relations": dict(relations),
                "correct": relations["correct"], "substitution": relations["substitution"],
                "deletion": relations["deletion"], "addition": relations["addition"],
                "has_unresolved": has_unresolved, "has_addition": has_addition,
                "boundary_available": boundary_available, "usable": usable,
            }
            records.append(record)
            accounting[f"{split}_candidate"] += 1
            if usable:
                accounting[f"{split}_usable"] += 1
            if relations["deletion"] > 1:
                accounting["multiple_deletion_words"] += 1
            if relations["substitution"] and relations["deletion"]:
                accounting["substitution_plus_deletion_words"] += 1
            if relations["addition"] and relations["deletion"]:
                accounting["addition_plus_deletion_words"] += 1
            if has_unresolved:
                accounting["malformed_or_unresolved_words"] += 1
            if not boundary_available:
                accounting["unavailable_word_boundary"] += 1
            if has_addition:
                accounting["addition_words_excluded"] += 1

    return records, {
        "counts": dict(accounting),
        "target_rows_without_manual_word": {
            "total": sum(target_rows_without_manual_word.values()),
            "by_relation": dict(target_rows_without_manual_word),
        },
        "invalid_forced_alignment_phone_labels": dict(invalid_aligned),
        "mfa_vs_manual_boundary_start_error_seconds": quantiles(boundary_start_errors),
        "mfa_vs_manual_boundary_end_error_seconds": quantiles(boundary_end_errors),
    }


def audit_oracle(words: list[dict[str, Any]]) -> dict[str, Any]:
    relation = {
        name: {"support": 0, "chosen_recovered": 0, "optimal_path_representable": 0, "operation_ambiguous": 0}
        for name in ("correct", "substitution", "deletion")
    }
    exact_words = 0
    representable_words = 0
    ambiguous_words = 0
    capped_words = 0
    canonical_equal_substitutions = 0
    examples: list[dict[str, Any]] = []
    for word in words:
        result = oracle_alignment(word["expected"], word["observed"])
        ground_signature = tuple(word["ground_operations"])
        chosen_signature = tuple(result["chosen_signature"])
        exact_words += chosen_signature == ground_signature
        representable_words += ground_signature in result["signatures"]
        ambiguous_words += result["ambiguous"]
        capped_words += result["paths_capped"]
        for index, row in enumerate(word["clean_rows"]):
            name = row["relation"]
            relation[name]["support"] += 1
            relation[name]["chosen_recovered"] += chosen_signature[index] == ground_signature[index]
            relation[name]["optimal_path_representable"] += ground_signature[index] in result["operation_options"][index]
            relation[name]["operation_ambiguous"] += len(result["operation_options"][index]) > 1
            if name == "substitution" and row["expected"] == row["observed"]:
                canonical_equal_substitutions += 1
        if chosen_signature != ground_signature and len(examples) < 25:
            examples.append({
                "word_id": word["word_id"], "word": word["word"], "expected": word["expected"],
                "observed": word["observed"], "ground_operations": ground_signature,
                "chosen_operations": chosen_signature, "ambiguous": result["ambiguous"],
            })
    for values in relation.values():
        values["chosen_recovery_rate"] = values["chosen_recovered"] / values["support"] if values["support"] else None
        values["optimal_path_representable_rate"] = (
            values["optimal_path_representable"] / values["support"] if values["support"] else None
        )
        values["ambiguity_rate"] = values["operation_ambiguous"] / values["support"] if values["support"] else None
    return {
        "words": len(words),
        "exact_deterministic_word_recovery": exact_words,
        "exact_deterministic_word_recovery_rate": exact_words / len(words),
        "ground_truth_signature_optimal_path_representable": representable_words,
        "ground_truth_signature_representable_rate": representable_words / len(words),
        "ambiguous_optimal_alignment_words": ambiguous_words,
        "ambiguous_word_rate": ambiguous_words / len(words),
        "path_enumeration_capped_words": capped_words,
        "canonical_equal_substitutions_not_identifiable_after_stress_removal": canonical_equal_substitutions,
        "relation_recovery": relation,
        "first_mismatch_examples": examples,
    }


def audit_strides(words: list[dict[str, Any]]) -> dict[str, Any]:
    phone_durations = [
        row["end"] - row["start"]
        for word in words for row in word["clean_rows"] if row["end"] > row["start"]
    ]
    word_durations = [word["mfa_end"] - word["mfa_start"] for word in words]
    output: dict[str, Any] = {
        "word_duration_seconds": quantiles(word_durations),
        "manual_phone_duration_seconds": quantiles(phone_durations),
        "candidates": {},
    }
    for stride_ms in STRIDES_MS:
        stride = stride_ms / 1000.0
        counts: list[int] = []
        padded = 0
        total = 0
        split_probes: Counter[str] = Counter()
        for word in words:
            centers = probe_centers(word["mfa_start"], word["mfa_end"], stride)
            counts.append(len(centers))
            split_probes[word["split"]] += len(centers)
            total += len(centers)
            padded += sum(center - 0.25 < 0 or center + 0.25 > word["utterance_end"] for center in centers)
        output["candidates"][str(stride_ms)] = {
            "stride_ms": stride_ms,
            "crop_ms": 500,
            "adjacent_crop_overlap": 1.0 - stride / 0.5,
            "probes_per_word": quantiles([float(value) for value in counts]),
            "total_probes": total,
            "train_probes": split_probes["train"],
            "validation_probes": split_probes["validation"],
            "edge_padded_probes": padded,
            "edge_padding_rate": padded / total,
            "phone_duration_at_least_one_stride_rate": float(np.mean(np.asarray(phone_durations) >= stride)),
            "phone_duration_at_least_two_strides_rate": float(np.mean(np.asarray(phone_durations) >= 2 * stride)),
            "temporal_redundancy": "HIGH" if stride_ms <= 20 else ("MODERATE_HIGH" if stride_ms == 40 else "MODERATE"),
        }
    return output


def choose_samples(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(words, key=lambda word: (word["split"] != "validation", word["speaker_id"], word["utterance_id"], word["manual_word_index"]))
    categories = {
        "all_correct": lambda word: word["substitution"] == 0 and word["deletion"] == 0,
        "one_substitution": lambda word: word["substitution"] == 1 and word["deletion"] == 0,
        "one_deletion": lambda word: word["deletion"] == 1 and word["substitution"] == 0,
        "substitution_plus_deletion": lambda word: word["substitution"] > 0 and word["deletion"] > 0,
        "initial_deletion": lambda word: bool(word["ground_operations"]) and word["ground_operations"][0] == "D",
        "medial_deletion": lambda word: any(op == "D" for op in word["ground_operations"][1:-1]),
        "final_deletion": lambda word: bool(word["ground_operations"]) and word["ground_operations"][-1] == "D",
    }
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for category, predicate in categories.items():
        candidate = next((word for word in ordered if word["word_id"] not in used and predicate(word)), None)
        if candidate is None:
            candidate = next((word for word in ordered if predicate(word)), None)
        if candidate is None:
            selected.append({"category": category, "available": False})
            continue
        used.add(candidate["word_id"])
        selected.append({**candidate, "category": category, "available": True})
    return selected


def run_sample_evidence(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    available = [sample for sample in samples if sample.get("available")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = r3.SmallPronunciationCNNAttention().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    extractor = r3.FixedLogMel().to(device).eval()
    store = r3.SequentialWaveStore()
    waveforms: list[np.ndarray] = []
    probe_metadata: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(available):
        audio = store.load(sample["audio_path"])
        centers = probe_centers(sample["mfa_start"], sample["mfa_end"], RECOMMENDED_STRIDE_MS / 1000.0)
        for center in centers:
            waveforms.append(r3.centered_window(audio, center, center))
            probe_metadata.append({
                "sample_index": sample_index,
                "center": center,
                "offset_ms": (center - sample["mfa_start"]) * 1000.0,
            })
    waveform_tensor = torch.from_numpy(np.stack(waveforms)).float()
    logits_batches: list[torch.Tensor] = []
    feature_shape: tuple[int, ...] | None = None
    internal_shape: tuple[int, ...] | None = None
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(waveform_tensor), r3.PREPROCESS_BATCH_SIZE):
            feature = extractor(waveform_tensor[start:start + r3.PREPROCESS_BATCH_SIZE].to(device))
            feature_shape = tuple(feature.shape[1:])
            for model_start in range(0, len(feature), r3.BATCH_SIZE):
                batch = feature[model_start:model_start + r3.BATCH_SIZE]
                if internal_shape is None:
                    internal_shape = tuple(model.features(batch).shape[1:])
                logits_batches.append(model(batch).cpu())
    inference_seconds = time.perf_counter() - started
    logits = torch.cat(logits_batches).numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    top1 = np.argmax(probabilities, axis=1)
    top3 = np.argsort(-probabilities, axis=1)[:, :3]

    csv_rows: list[dict[str, Any]] = []
    by_sample: dict[int, list[int]] = defaultdict(list)
    for probe_index, metadata in enumerate(probe_metadata):
        by_sample[metadata["sample_index"]].append(probe_index)
        row = {
            "category": available[metadata["sample_index"]]["category"],
            "word_id": available[metadata["sample_index"]]["word_id"],
            "center_seconds": metadata["center"], "offset_from_word_start_ms": metadata["offset_ms"],
            "top1_phone": r3.PHONE_VOCAB[int(top1[probe_index])],
            "top1_probability": float(probabilities[probe_index, top1[probe_index]]),
            "top3_phones": "|".join(r3.PHONE_VOCAB[int(value)] for value in top3[probe_index]),
            "top3_probabilities": "|".join(str(float(probabilities[probe_index, value])) for value in top3[probe_index]),
        }
        for phone_index, phone in enumerate(r3.PHONE_VOCAB):
            row[f"logit_{phone}"] = float(logits[probe_index, phone_index])
            row[f"prob_{phone}"] = float(probabilities[probe_index, phone_index])
        csv_rows.append(row)

    summaries: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(available):
        indexes = by_sample[sample_index]
        sample_top1 = [r3.PHONE_VOCAB[int(top1[index])] for index in indexes]
        offsets = [probe_metadata[index]["offset_ms"] for index in indexes]
        expected_peaks = {
            f"{position}:{phone}": float(np.max(probabilities[indexes, r3.PHONE_TO_ID[phone]]))
            for position, phone in enumerate(sample["expected"])
        }
        error_diagnostics: list[dict[str, Any]] = []
        for position, row in enumerate(sample["clean_rows"]):
            if row["relation"] not in {"substitution", "deletion"}:
                continue
            expected_phone = row["expected"]
            diagnostic: dict[str, Any] = {
                "position": position, "relation": row["relation"], "expected": expected_phone,
                "observed": row["observed"],
                "expected_peak_probability": expected_peaks[f"{position}:{expected_phone}"],
                "expected_top1_probe_count": sample_top1.count(expected_phone),
            }
            if row["relation"] == "substitution":
                observed_phone = row["observed"]
                diagnostic["observed_peak_probability"] = float(np.max(probabilities[indexes, r3.PHONE_TO_ID[observed_phone]]))
                diagnostic["observed_top1_probe_count"] = sample_top1.count(observed_phone)
                diagnostic["distinct_alternative_observable"] = sample_top1.count(observed_phone) > 0
            else:
                if 0 < position < len(sample["expected"]) - 1:
                    left, right = sample["expected"][position - 1], sample["expected"][position + 1]
                    left_positions = [index for index, phone in enumerate(sample_top1) if phone == left]
                    right_positions = [index for index, phone in enumerate(sample_top1) if phone == right]
                    plausible = (
                        sample_top1.count(expected_phone) == 0
                        and bool(left_positions) and bool(right_positions)
                        and min(left_positions) < max(right_positions)
                    )
                    diagnostic.update({
                        "left_expected": left, "right_expected": right,
                        "left_top1_probe_count": len(left_positions), "right_top1_probe_count": len(right_positions),
                        "plausible_skip_pattern": plausible,
                    })
                else:
                    diagnostic["plausible_skip_pattern"] = "NOT_ASSESSABLE_WORD_EDGE"
            error_diagnostics.append(diagnostic)
        summaries.append({
            "category": sample["category"], "word_id": sample["word_id"], "split": sample["split"],
            "word": sample["word"], "expected": sample["expected"], "observed_oracle": sample["observed"],
            "ground_operations": sample["ground_operations"], "probes": len(indexes),
            "top1_runs": compress_runs(sample_top1, offsets), "expected_peak_probabilities": expected_peaks,
            "error_diagnostics": error_diagnostics,
        })
    return {
        "checkpoint_epoch": int(checkpoint["epoch"]), "device": str(device),
        "probes": len(probe_metadata), "inference_seconds": inference_seconds,
        "probes_per_second": len(probe_metadata) / inference_seconds,
        "feature_shape": list(feature_shape or ()), "internal_feature_map_shape": list(internal_shape or ()),
        "samples": summaries,
    }, csv_rows


def main() -> int:
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite audit directory: {EXPERIMENT_DIR}")
    if r3.sha256_file(V4_PATH) != EXPECTED_V4_SHA:
        raise RuntimeError("V4 SHA mismatch")
    if r3.sha256_file(CHECKPOINT_PATH) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen R3 checkpoint SHA mismatch")
    audio_root = r3.require_audio_root()
    r3.set_seed(SEED)

    words, reconstruction = build_word_records(audio_root)
    usable_words = [word for word in words if word["usable"]]
    oracle = audit_oracle(usable_words)
    strides = audit_strides(usable_words)
    samples = choose_samples(usable_words)
    sample_evidence, evidence_rows = run_sample_evidence(samples)

    evidence_options = {
        "A_frozen_r3_sliding_logits": {
            "available": True, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "output": "fixed-center timeline x 40 logits; exact frozen 0.50 s preprocessing",
            "recommended": True,
        },
        "B_mfa_kaldi_frame_posteriors": {
            "available_in_current_project_output": False,
            "evidence": "run_mfa_alignment retains parsed TextGrid word/phone intervals only; no lattice/posterior matrix contract",
        },
        "C_r3_internal_temporal_map": {
            "technically_accessible": True,
            "shape_for_0_50s_input": sample_evidence["internal_feature_map_shape"],
            "validated_phone_posterior_sequence": False,
            "limitation": "classifier was trained after temporal attention pooling, not at the four internal time positions",
        },
    }
    alignment_contract = {
        "state": "DP(i,t): expected-phone index i and acoustic probe index t",
        "emission": "negative log-softmax evidence for expected phone over a contiguous probe span",
        "substitution": "consume a contiguous span whose sustained best competing real phone differs from expected",
        "deletion": "advance expected index without assigning an acoustic span; global skip cost only",
        "advance_time": "consume another probe in the current contiguous emission span",
        "optional_insertion": "disabled in first R4-3B because addition words are excluded",
        "cost_policy": (
            "No costs selected in R4-3A. For R4-3B, use frozen log-softmax emissions and global Laplace-smoothed "
            "MATCH/SUBSTITUTION/DELETION priors estimated once from TRAIN only; no phone-, speaker-, duration-, or validation-tuned costs."
        ),
        "decision": (
            "DELETE only when the best monotonic path skips Ei; SUBSTITUTION only when Ei consumes a span dominated "
            "by a sustained competing phone. Low P(Ei) alone is insufficient."
        ),
    }
    runtime = {
        "verdict": "WORD_SEQUENCE_RUNTIME_READY_WITH_SMALL_LAYER",
        "available": ["target word", "expected canonical phones", "MFA word boundary", "word audio", "frozen acoustic checkpoint"],
        "missing": ["batched sliding-window evidence service", "posterior-aware monotonic DP layer", "sequence reliability contract"],
        "mfa_phone_labels_used_as_observed_truth": False,
    }
    oracle_ok = (
        oracle["ground_truth_signature_representable_rate"] >= 0.99
        and oracle["relation_recovery"]["deletion"]["optimal_path_representable_rate"] >= 0.99
    )
    if not usable_words or not oracle_ok:
        final_status = "R4_3A_SEQUENCE_DESIGN_BLOCKED"
    else:
        final_status = "R4_3A_SEQUENCE_DESIGN_READY_WITH_WARNINGS"

    experiment = {
        "id": "R4-3B", "scope": "future validation-only; do not run in R4-3A",
        "acoustic_evidence": "frozen R3-1D 40-phone logits on 0.50 s crops",
        "stride_ms": RECOMMENDED_STRIDE_MS,
        "alignment": alignment_contract,
        "split": {"train": list(r3.TRAIN_SPEAKERS), "validation": list(r3.VALIDATION_SPEAKERS), "test": "CLOSED"},
        "selection": "single pre-registered deterministic run; transition priors from TRAIN only; no validation cost tuning",
        "metrics": [
            "phone-operation Macro-F1", "deletion precision/recall/F1", "substitution false-deletion rate",
            "word exact operation-sequence accuracy", "per-speaker deletion metrics", "per-phone deletion metrics",
        ],
        "anti_duration_controls": [
            "no manual phone center/duration input", "fixed 0.50 s crop and fixed 40 ms stride",
            "word-duration and expected-length matched validation diagnostic",
            "compare against frozen duration-only and R4-1 baselines",
        ],
        "hard_gates": {
            "full_validation_macro_f1": ">=0.70",
            "deletion_f1": ">=0.40", "deletion_recall": ">=0.45",
            "macro_f1_gain_over_duration_baseline": ">=+0.03",
            "matched_macro_f1": ">=0.60", "matched_deletion_f1": ">=0.55",
            "substitution_false_deletion_rate": "<=0.25",
            "speaker_rule": "for speakers with >=30 deletions, deletion recall >=0.25",
        },
    }

    EXPERIMENT_DIR.mkdir(parents=True)
    write_json(EXPERIMENT_DIR / "preflight.json", {
        "v4_sha256": EXPECTED_V4_SHA, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "train_validation_only": True, "r4_test_paths_resolved": False,
        "r4_test_audio_accessed": False, "r4_test_inference": False, "training_performed": False,
    })
    write_json(EXPERIMENT_DIR / "word_reconstruction.json", reconstruction)
    write_json(EXPERIMENT_DIR / "oracle_alignment.json", oracle)
    write_json(EXPERIMENT_DIR / "stride_audit.json", strides)
    write_json(EXPERIMENT_DIR / "acoustic_evidence_options.json", evidence_options)
    write_json(EXPERIMENT_DIR / "representative_sample_report.json", sample_evidence)
    write_json(EXPERIMENT_DIR / "sequence_alignment_contract.json", alignment_contract)
    write_json(EXPERIMENT_DIR / "runtime_compatibility.json", runtime)
    median_probes = strides["candidates"][str(RECOMMENDED_STRIDE_MS)]["probes_per_word"]["median"]
    write_json(EXPERIMENT_DIR / "compute_estimate.json", {
        "recommended_stride_ms": RECOMMENDED_STRIDE_MS,
        "median_probes_per_word": median_probes,
        "unbatched_forward_equivalents_per_word": median_probes,
        "model_batches_per_median_word_at_batch8": math.ceil(float(median_probes) / r3.BATCH_SIZE),
        "small_sample_gpu_throughput_probes_per_second": sample_evidence["probes_per_second"],
        "small_sample_estimated_median_word_seconds": float(median_probes) / sample_evidence["probes_per_second"],
        "prior_r4_2b_bulk_reference": {
            "probes": 56_245, "seconds": 16.1848403,
            "probes_per_second": 56_245 / 16.1848403,
            "estimated_median_word_seconds_when_large_batches_are_amortized": float(median_probes) / (56_245 / 16.1848403),
        },
        "cpu": "not benchmarked; expected materially slower than CUDA and must be measured before runtime integration",
    })
    write_json(EXPERIMENT_DIR / "r4_3b_preregistered_design.json", experiment)
    evidence_fields = list(evidence_rows[0])
    with (EXPERIMENT_DIR / "representative_acoustic_evidence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evidence_fields)
        writer.writeheader()
        writer.writerows(evidence_rows)
    final = {
        "status": final_status,
        "usable_train_words": sum(word["split"] == "train" for word in usable_words),
        "usable_validation_words": sum(word["split"] == "validation" for word in usable_words),
        "oracle_deletion_deterministic_tiebreak_recovery": oracle["relation_recovery"]["deletion"]["chosen_recovery_rate"],
        "oracle_deletion_optimal_path_representable": oracle["relation_recovery"]["deletion"]["optimal_path_representable_rate"],
        "recommended_evidence": "FROZEN_R3_SLIDING_LOGITS",
        "recommended_stride_ms": RECOMMENDED_STRIDE_MS,
        "runtime": runtime["verdict"],
        "main_unresolved_risk": "global skip/substitution transition costs and broad-context R3 posterior localization",
        "training_performed": False, "r4_test_accessed": False,
    }
    write_json(EXPERIMENT_DIR / "final_status.json", final)
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
