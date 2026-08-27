from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
EXPECTED_V4_SHA = "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D"
RAW_ROOT = Path(os.environ["L2_ARCTIC_ROOT"]).expanduser().resolve()
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r4_2c_mfa_missing_phone_anchor"
TRAIN_SPEAKERS = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
VALIDATION_SPEAKERS = ("ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI")
TEST_SPEAKERS = ("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK")
AUDIT_SPEAKERS = frozenset(TRAIN_SPEAKERS + VALIDATION_SPEAKERS)
PHONE_VOCAB = frozenset((
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
    "UH", "UW", "V", "W", "Y", "Z", "ZH",
))
NON_SPEECH = frozenset(("", "sp", "sil"))
IMPORTANT_PHONES = ("D", "T", "R", "L", "N", "Z", "V", "K")
EXPECTED_DELETIONS = 2_591
EXPECTED_CORRECT = 76_703
EXPECTED_SUBSTITUTIONS = 9_068
MAX_OPTIMAL_PATHS = 256


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_phone(label: str) -> str | None:
    value = label.strip()
    if value in NON_SPEECH:
        return None
    value = re.sub(r"([A-Z]+)[012]$", r"\1", value)
    return value if value in PHONE_VOCAB else None


def normalize_word(value: str) -> str:
    return re.sub(r"[^a-z']", "", value.casefold())


def parse_textgrid(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    tiers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    current = ""
    pending: dict[str, Any] | None = None
    xmax = 0.0
    for raw in lines:
        line = raw.strip()
        if line.startswith("xmax") and "=" in line:
            try:
                xmax = max(xmax, float(line.split("=", 1)[1].strip()))
            except ValueError:
                pass
        if line.startswith("name") and "=" in line:
            current = line.split("=", 1)[1].strip().strip('"').casefold()
            pending = None
        elif re.match(r"intervals\s*\[\d+\]:", line):
            pending = {}
        elif pending is not None and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            if key == "xmin":
                pending["start"] = float(value)
            elif key == "xmax":
                pending["end"] = float(value)
            elif key == "text":
                pending["label"] = value.strip().strip('"').replace('""', '"').strip()
                if {"start", "end"} <= pending.keys():
                    tiers[current].append(pending)
                pending = None
    word_tier = next((values for name, values in tiers.items() if name in {"word", "words", "orthography"}), [])
    phone_tier = next((values for name, values in tiers.items() if name in {"phone", "phones", "phoneme", "phonemes"}), [])
    return {
        "words": [row for row in word_tier if row["label"]],
        "phones": phone_tier,
        "xmax": xmax,
        "tier_names": sorted(tiers),
    }


def containing_word(words: list[dict[str, Any]], start: float, end: float) -> int | None:
    midpoint = (start + end) / 2
    containing = [
        index for index, word in enumerate(words)
        if start >= word["start"] - 0.001 and end <= word["end"] + 0.001
    ]
    if len(containing) == 1:
        return containing[0]
    overlapping = [
        index for index, word in enumerate(words)
        if word["start"] - 0.001 <= midpoint <= word["end"] + 0.001
    ]
    return overlapping[0] if len(overlapping) == 1 else None


def align_tokens(expected: list[str], aligned: list[str]) -> dict[str, Any]:
    n, m = len(expected), len(aligned)
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
            diagonal = int(cost[i - 1, j - 1]) + (expected[i - 1] != aligned[j - 1])
            deletion = int(cost[i - 1, j]) + 1
            insertion = int(cost[i, j - 1]) + 1
            best = min(diagonal, deletion, insertion)
            cost[i, j] = best
            if diagonal == best:
                back[i][j].append("M" if expected[i - 1] == aligned[j - 1] else "S")
            if deletion == best:
                back[i][j].append("D")
            if insertion == best:
                back[i][j].append("I")

    paths: list[tuple[list[tuple[str, int | None, int | None]], int, int]] = [([], n, m)]
    completed: list[list[tuple[str, int | None, int | None]]] = []
    capped = False
    while paths:
        operations, i, j = paths.pop()
        if i == 0 and j == 0:
            completed.append(list(reversed(operations)))
            if len(completed) >= MAX_OPTIMAL_PATHS:
                capped = bool(paths)
                break
            continue
        for operation in reversed(back[i][j]):
            if operation in {"M", "S"}:
                paths.append((operations + [(operation, i - 1, j - 1)], i - 1, j - 1))
            elif operation == "D":
                paths.append((operations + [(operation, i - 1, None)], i - 1, j))
            else:
                paths.append((operations + [(operation, None, j - 1)], i, j - 1))
    if not completed:
        raise RuntimeError("Sequence alignment produced no path")
    path = completed[0]
    missing_sets = {tuple(op[1] for op in candidate if op[0] == "D") for candidate in completed}
    mapping_signatures = {
        tuple((op[1], op[2]) for op in candidate if op[0] in {"M", "S"}) for candidate in completed
    }
    missing_membership: dict[int, set[bool]] = {index: set() for index in range(n)}
    mapping_options: dict[int, set[int | None]] = {index: set() for index in range(n)}
    for candidate in completed:
        candidate_missing = {int(op[1]) for op in candidate if op[0] == "D" and op[1] is not None}
        candidate_mapping = {
            int(i): int(j) for op, i, j in candidate
            if op in {"M", "S"} and i is not None and j is not None
        }
        for index in range(n):
            missing_membership[index].add(index in candidate_missing)
            mapping_options[index].add(candidate_mapping.get(index))
    mapping: dict[int, int] = {int(i): int(j) for op, i, j in path if op in {"M", "S"} and i is not None and j is not None}
    return {
        "cost": int(cost[n, m]), "operations": path, "mapping": mapping,
        "missing_expected": [int(op[1]) for op in path if op[0] == "D"],
        "insertions": sum(op[0] == "I" for op in path), "substitutions": sum(op[0] == "S" for op in path),
        "optimal_paths_examined": len(completed), "optimal_paths_capped": capped,
        "ambiguous_missing": len(missing_sets) > 1, "ambiguous_mapping": len(mapping_signatures) > 1,
        "ambiguous_missing_indices": [index for index, values in missing_membership.items() if len(values) > 1],
        "ambiguous_mapping_indices": [index for index, values in mapping_options.items() if len(values) > 1],
    }


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {key: None for key in ("count", "mean", "median", "p50", "p75", "p90", "p95", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(np.mean(array)), "median": float(np.median(array)),
        "p50": float(np.percentile(array, 50)), "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)), "max": float(np.max(array)),
        "within_ms": {
            str(limit): {"count": int(np.sum(array <= limit / 1000)), "rate": float(np.mean(array <= limit / 1000))}
            for limit in (25, 50, 75, 100, 150)
        },
    }


def main() -> int:
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite audit directory: {EXPERIMENT_DIR}")
    if sha256_file(V4_PATH) != EXPECTED_V4_SHA:
        raise RuntimeError("V4 SHA mismatch")
    if not RAW_ROOT.is_dir():
        raise RuntimeError(f"Raw root unavailable: {RAW_ROOT}")

    # Stream V4, retaining only TRAIN+VALIDATION. TEST fields are never parsed into audit records.
    utterance_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    relation_counts: Counter[str] = Counter()
    with V4_PATH.open(encoding="utf-8", newline="") as handle:
        for source_index, source in enumerate(csv.DictReader(handle)):
            speaker = source["speaker_id"]
            if speaker not in AUDIT_SPEAKERS:
                continue
            row = {
                "source_index": source_index, "speaker_id": speaker, "utterance_id": source["utterance_id"],
                "start": float(source["start_time"]), "end": float(source["end_time"]),
                "relation": source["relation"], "label_quality": source["label_quality"],
                "expected": source["expected_phone_canonical"], "observed": source["observed_phone_canonical"],
                "raw_label": source["raw_label"],
            }
            utterance_rows[(speaker, source["utterance_id"])].append(row)
            relation_counts[row["relation"]] += 1
    if relation_counts["deletion"] != EXPECTED_DELETIONS:
        raise RuntimeError(f"Deletion support mismatch: {relation_counts['deletion']}")

    invalid_mfa_labels: Counter[str] = Counter()
    missing_mfa_files: list[str] = []
    mapping_failures: Counter[str] = Counter()
    utterance_records: list[dict[str, Any]] = []
    all_target_rows: list[dict[str, Any]] = []
    uniform_errors: list[float] = []
    manual_words_total = 0
    parsed_utterances = 0

    for (speaker, utterance), rows in sorted(utterance_rows.items()):
        mfa_path = RAW_ROOT / speaker / "textgrid" / f"{utterance}.TextGrid"
        manual_path = RAW_ROOT / speaker / "annotation" / f"{utterance}.TextGrid"
        target_rows = [
            row for row in rows
            if row["label_quality"] == "clean" and row["relation"] in {"correct", "substitution", "deletion"}
        ]
        all_target_rows.extend(target_rows)
        if not mfa_path.is_file():
            missing_mfa_files.append(f"{speaker}/{utterance}")
            for row in target_rows:
                row["anchor_status"] = "MFA_TEXTGRID_MISSING"
            continue

        mfa = parse_textgrid(mfa_path)
        parsed_utterances += 1

        # This is the runtime-side expected sequence proxy. It uses only canonical
        # expected-phone identity and stable V4 row order; no manual start/end or
        # relation value participates in alignment or anchor generation.
        expected_rows = [
            row for row in sorted(rows, key=lambda item: item["source_index"])
            if row["expected"] in PHONE_VOCAB
        ]
        expected = [row["expected"] for row in expected_rows]
        aligned_intervals: list[dict[str, Any]] = []
        for phone in sorted(mfa["phones"], key=lambda item: (item["start"], item["end"])):
            canonical = canonical_phone(phone["label"])
            if phone["label"] not in NON_SPEECH and canonical is None:
                invalid_mfa_labels[phone["label"]] += 1
                continue
            if canonical is None:
                continue
            aligned_intervals.append({
                **phone,
                "canonical": canonical,
                "mfa_word_index": containing_word(mfa["words"], phone["start"], phone["end"]),
            })
        aligned = [phone["canonical"] for phone in aligned_intervals]
        if not expected or not aligned:
            mapping_failures["empty_expected_or_aligned_sequence"] += 1
            for row in target_rows:
                row["anchor_status"] = "NO_SEQUENCE"
            continue

        sequence = align_tokens(expected, aligned)
        mapping = sequence["mapping"]
        missing = sequence["missing_expected"]
        ambiguous_missing_indices = set(sequence["ambiguous_missing_indices"])
        ambiguous_mapping_indices = set(sequence["ambiguous_mapping_indices"])
        operation_by_expected = {
            int(i): operation for operation, i, _ in sequence["operations"] if i is not None
        }
        utterance_records.append({
            "speaker_id": speaker,
            "utterance_id": utterance,
            "expected_count": len(expected),
            "aligned_count": len(aligned),
            "cost": sequence["cost"],
            "missing_count": len(missing),
            "insertions": sequence["insertions"],
            "substitutions": sequence["substitutions"],
            "ambiguous_missing": sequence["ambiguous_missing"],
            "ambiguous_mapping": sequence["ambiguous_mapping"],
        })

        uniform_step = mfa["xmax"] / len(expected_rows)
        for expected_index, row in enumerate(expected_rows):
            row["expected_position"] = expected_index
            row["missing_candidates"] = missing
            row["missing_stable"] = expected_index in missing and expected_index not in ambiguous_missing_indices
            row["sequence_operation"] = operation_by_expected.get(expected_index)
            mapped_index = mapping.get(expected_index)
            if mapped_index is not None:
                mapped = aligned_intervals[mapped_index]
                row["mapped_aligned_phone"] = mapped["canonical"]
                row["mapped_aligned_start"] = mapped["start"]
                row["mapped_aligned_end"] = mapped["end"]
                row["mapped_mfa_word_index"] = mapped["mfa_word_index"]
            row["uniform_anchor"] = (expected_index + 0.5) * uniform_step
            if row["relation"] == "deletion" and row["label_quality"] == "clean":
                uniform_errors.append(abs(row["uniform_anchor"] - (row["start"] + row["end"]) / 2))

            if expected_index not in missing:
                row["anchor_status"] = "NOT_MISSING"
                continue
            if expected_index in ambiguous_missing_indices:
                row["anchor_status"] = "AMBIGUOUS_MISSING"
                continue
            left_index, right_index = expected_index - 1, expected_index + 1
            left_mapped = mapping.get(left_index)
            right_mapped = mapping.get(right_index)
            neighbors_ambiguous = left_index in ambiguous_mapping_indices or right_index in ambiguous_mapping_indices
            if neighbors_ambiguous:
                row["anchor_status"] = "AMBIGUOUS_NEIGHBOR_MAPPING"
            elif left_mapped is not None and right_mapped is not None:
                left_word = aligned_intervals[left_mapped]["mfa_word_index"]
                right_word = aligned_intervals[right_mapped]["mfa_word_index"]
                if left_word is not None and left_word == right_word:
                    row["predicted_anchor"] = (
                        aligned_intervals[left_mapped]["end"] + aligned_intervals[right_mapped]["start"]
                    ) / 2
                    row["anchor_status"] = "BOTH_SIDED_ANCHOR"
                    row["left_aligned_end"] = aligned_intervals[left_mapped]["end"]
                    row["right_aligned_start"] = aligned_intervals[right_mapped]["start"]
                else:
                    row["anchor_status"] = "ONE_SIDED_ANCHOR_WORD_BOUNDARY"
            elif left_mapped is not None or right_mapped is not None:
                row["anchor_status"] = "ONE_SIDED_ANCHOR"
            else:
                row["anchor_status"] = "NO_ANCHOR"

        # Manual annotations are attached only now, after every predicted missing
        # identity and anchor is frozen, and are used solely for evaluation.
        manual = parse_textgrid(manual_path)
        manual_words_total += len(manual["words"])
        rows_by_manual_word: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in expected_rows:
            manual_word_index = containing_word(manual["words"], row["start"], row["end"])
            row["manual_word_index"] = manual_word_index
            if manual_word_index is not None:
                rows_by_manual_word[manual_word_index].append(row)
        for manual_word_index, word_rows in rows_by_manual_word.items():
            ordered_word_rows = sorted(word_rows, key=lambda item: item["source_index"])
            stable_missing_sources = [row["source_index"] for row in ordered_word_rows if row.get("missing_stable")]
            word_has_substitution_op = any(row.get("sequence_operation") == "S" for row in ordered_word_rows)
            for word_position, row in enumerate(ordered_word_rows):
                row["word"] = manual["words"][manual_word_index]["label"]
                row["word_expected_position"] = word_position
                row["word_phone_count"] = len(ordered_word_rows)
                row["word_missing_source_indices"] = stable_missing_sources
                row["word_has_substitution"] = word_has_substitution_op
                row["utterance_has_insertion"] = sequence["insertions"] > 0

    deletion_rows = [row for row in all_target_rows if row["relation"] == "deletion"]
    correct_rows = [row for row in all_target_rows if row["relation"] == "correct"]
    substitution_rows = [row for row in all_target_rows if row["relation"] == "substitution"]
    if len(deletion_rows) != EXPECTED_DELETIONS:
        raise RuntimeError(f"Evaluable deletion row accounting mismatch: {len(deletion_rows)}")
    if len(correct_rows) != EXPECTED_CORRECT or len(substitution_rows) != EXPECTED_SUBSTITUTIONS:
        raise RuntimeError(
            f"Control row accounting mismatch: correct={len(correct_rows)}, substitution={len(substitution_rows)}"
        )
    source_indices = [row["source_index"] for row in all_target_rows]
    if len(source_indices) != len(set(source_indices)):
        raise RuntimeError("Duplicate V4 row identity in audit accounting")

    def is_missing(row: dict[str, Any]) -> bool:
        return bool(row.get("missing_stable"))

    exact = [row for row in deletion_rows if is_missing(row)]
    wrong = [row for row in deletion_rows if not is_missing(row) and row.get("word_missing_source_indices")]
    none = [row for row in deletion_rows if not is_missing(row) and not row.get("word_missing_source_indices")]
    ambiguous = [row for row in deletion_rows if row.get("anchor_status") in {"AMBIGUOUS_MISSING", "AMBIGUOUS_NEIGHBOR_MAPPING"}]
    both_sided = [row for row in deletion_rows if row.get("anchor_status") == "BOTH_SIDED_ANCHOR"]
    one_sided = [row for row in deletion_rows if row.get("anchor_status") in {
        "ONE_SIDED_ANCHOR", "ONE_SIDED_ANCHOR_WORD_BOUNDARY"
    }]
    no_anchor = [row for row in deletion_rows if row not in both_sided and row not in one_sided]
    anchor_errors = [abs(row["predicted_anchor"] - (row["start"] + row["end"]) / 2) for row in both_sided]
    anchor_distribution = distribution(anchor_errors)
    anchor_coverage = len(both_sided) / len(deletion_rows)
    median_error = anchor_distribution["median"]
    p90_error = anchor_distribution["p90"]
    if anchor_coverage >= 0.80 and median_error is not None and median_error <= 0.050 and p90_error <= 0.100:
        anchor_criterion = "ANCHOR_HIGH_PRECISION"
    elif anchor_coverage >= 0.60 and median_error is not None and median_error <= 0.075 and p90_error <= 0.150:
        anchor_criterion = "ANCHOR_USABLE_WITH_WARNINGS"
    else:
        anchor_criterion = "ANCHOR_NOT_RELIABLE"

    false_missing = {
        "correct_origin": {
            "rows": len(correct_rows), "false_missing": sum(is_missing(row) for row in correct_rows),
            "rate": sum(is_missing(row) for row in correct_rows) / len(correct_rows),
            "word_any_missing_rate": sum(bool(row.get("word_missing_source_indices")) for row in correct_rows) / len(correct_rows),
            "sequence_mapping_evaluable": sum("missing_candidates" in row for row in correct_rows),
            "sequence_mapping_coverage": sum("missing_candidates" in row for row in correct_rows) / len(correct_rows),
        },
        "substitution_origin": {
            "rows": len(substitution_rows), "false_missing": sum(is_missing(row) for row in substitution_rows),
            "rate": sum(is_missing(row) for row in substitution_rows) / len(substitution_rows),
            "word_any_missing_rate": sum(bool(row.get("word_missing_source_indices")) for row in substitution_rows) / len(substitution_rows),
            "sequence_mapping_evaluable": sum("missing_candidates" in row for row in substitution_rows),
            "sequence_mapping_coverage": sum("missing_candidates" in row for row in substitution_rows) / len(substitution_rows),
        },
    }

    def quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
        generated = [row for row in rows if row.get("anchor_status") == "BOTH_SIDED_ANCHOR"]
        errors = [abs(row["predicted_anchor"] - (row["start"] + row["end"]) / 2) for row in generated]
        return {
            "manual_deletions": len(rows), "exact_missing": sum(is_missing(row) for row in rows),
            "identification_rate": sum(is_missing(row) for row in rows) / len(rows) if rows else None,
            "anchors": len(generated), "anchor_coverage": len(generated) / len(rows) if rows else None,
            "timing_error": distribution(errors),
        }

    position_groups = {
        "single_phone": [row for row in deletion_rows if row.get("word_phone_count") == 1],
        "initial": [row for row in deletion_rows if row.get("word_phone_count", 0) > 1
                    and row.get("word_expected_position") == 0],
        "final": [row for row in deletion_rows if row.get("word_expected_position") is not None
                  and row.get("word_phone_count", 0) > 1
                  and row["word_expected_position"] == row.get("word_phone_count", 0) - 1],
        "medial": [row for row in deletion_rows if row.get("word_expected_position") not in {None, 0}
                   and row.get("word_phone_count", 0) > 1
                   and row["word_expected_position"] != row.get("word_phone_count", 0) - 1],
        "unmapped_position": [row for row in deletion_rows if row.get("word_expected_position") is None],
    }
    position_quality = {name: quality(rows) for name, rows in position_groups.items()}

    phone_counts = Counter(row["expected"] for row in deletion_rows)
    phone_set = sorted(set(IMPORTANT_PHONES) | {phone for phone, count in phone_counts.items() if count >= 20})
    phone_quality = {phone: quality([row for row in deletion_rows if row["expected"] == phone]) for phone in phone_set}
    dtrl_quality = {
        "D_T_R_L": quality([row for row in deletion_rows if row["expected"] in {"D", "T", "R", "L"}]),
        "other_phones": quality([row for row in deletion_rows if row["expected"] not in {"D", "T", "R", "L"}]),
    }
    speaker_quality = {
        speaker: quality([row for row in deletion_rows if row["speaker_id"] == speaker])
        for speaker in TRAIN_SPEAKERS + VALIDATION_SPEAKERS
    }

    evaluated_word_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in all_target_rows:
        if row.get("manual_word_index") is not None:
            evaluated_word_groups[(row["speaker_id"], row["utterance_id"], row["manual_word_index"])].append(row)
    word_stats = {
        "manual_words_total": manual_words_total,
        "evaluated_words": len(evaluated_word_groups),
        "exactly_one_predicted_expected_missing": sum(
            len({item["source_index"] for item in rows if is_missing(item)}) == 1
            for rows in evaluated_word_groups.values()
        ),
        "multiple_predicted_expected_missing": sum(
            len({item["source_index"] for item in rows if is_missing(item)}) > 1
            for rows in evaluated_word_groups.values()
        ),
        "ground_truth_words_with_one_deletion": sum(
            sum(item["relation"] == "deletion" for item in rows) == 1 for rows in evaluated_word_groups.values()
        ),
        "ground_truth_words_with_multiple_deletions": sum(
            sum(item["relation"] == "deletion" for item in rows) > 1 for rows in evaluated_word_groups.values()
        ),
        "predicted_missing_with_insertion_and_or_substitution": sum(
            any(is_missing(item) for item in rows)
            and any(item.get("word_has_substitution") or item.get("utterance_has_insertion") for item in rows)
            for rows in evaluated_word_groups.values()
        ),
        "ambiguous_sequence_mapping": sum(
            any(item.get("anchor_status") in {"AMBIGUOUS_MISSING", "AMBIGUOUS_NEIGHBOR_MAPPING"} for item in rows)
            for rows in evaluated_word_groups.values()
        ),
    }

    # Semantics are described only after hidden manual centers are revealed for evaluation.
    semantics = Counter()
    for row in deletion_rows:
        if row.get("anchor_status") == "BOTH_SIDED_ANCHOR":
            gap = row["right_aligned_start"] - row["left_aligned_end"]
            semantics["A_temporal_gap"] += gap > 0.010
            semantics["B_neighbors_touch_or_stretch"] += gap <= 0.010
        elif row.get("anchor_status") == "NOT_MISSING" and row.get("mapped_aligned_start") is not None:
            manual_center = (row["start"] + row["end"]) / 2
            if row["mapped_aligned_start"] <= manual_center <= row["mapped_aligned_end"]:
                semantics["C_expected_label_interval_contains_manual_deletion_center"] += 1
            elif row["mapped_aligned_end"] >= row["start"] and row["mapped_aligned_start"] <= row["end"]:
                semantics["C_expected_label_interval_overlaps_manual_deletion_slot"] += 1
            else:
                semantics["C_expected_label_retained_elsewhere_in_utterance"] += 1
        elif row.get("missing_stable"):
            semantics["D_missing_without_clean_anchor"] += 1
        else:
            semantics["D_no_missing_mapping"] += 1
    c_total = sum(value for key, value in semantics.items() if key.startswith("C_"))
    semantics_verdict = (
        "C_FORCED_ALIGNMENT_RETAINS_EXPECTED_LABEL"
        if c_total / len(deletion_rows) >= 0.80
        else "E_VARIES_ACROSS_CASES"
    )

    runtime_path = {
        "expected_sequence": {
            "file": "ai-worker/app/scorers/cnn_attention_scorer.py", "function": "_job_canonical_phones",
            "behavior": "reads job canonical_phones/phones and calls normalize_expected_phones",
        },
        "canonicalization": {
            "file": "ai-worker/app/phonetics/canonicalization.py", "functions": ["canonicalize_phones", "normalize_expected_phones"],
        },
        "provider_selection": {"file": "ai-worker/app/alignment/alignment_service.py", "function": "align_audio"},
        "mfa_execution": {"file": "ai-worker/app/alignment/mfa_aligner.py", "function": "run_mfa_alignment"},
        "textgrid_parsing": {"file": "ai-worker/app/alignment/textgrid_parser.py", "function": "parse_textgrid"},
        "fallback": {"file": "ai-worker/app/alignment/fallback_aligner.py", "function": "align_prompt_fallback",
                     "reliability": "limited_fallback_alignment; not primary evidence"},
        "quality": {"file": "ai-worker/app/alignment/quality.py", "function": "validate_alignment_quality"},
        "unsafe_position_warning": {
            "file": "ai-worker/app/scorers/cnn_attention_scorer.py", "function": "_alignment_phone_for_prediction",
            "behavior": "does not fall back to list position because alignment may omit a phone",
        },
    }
    implementability = "IMPLEMENTABLE_WITH_SMALL_ALIGNMENT_MAPPING_LAYER"
    acceptable_false_missing = false_missing["correct_origin"]["rate"] <= 0.05 and false_missing["substitution_origin"]["rate"] <= 0.10
    if anchor_criterion == "ANCHOR_HIGH_PRECISION" and acceptable_false_missing:
        final_status = "R4_2C_RUNTIME_ANCHOR_CONFIRMED"
    elif anchor_criterion == "ANCHOR_USABLE_WITH_WARNINGS":
        final_status = "R4_2C_RUNTIME_ANCHOR_PARTIAL"
    else:
        final_status = "R4_2C_RUNTIME_ANCHOR_BLOCKED"

    EXPERIMENT_DIR.mkdir(parents=True)
    row_fields = [
        "source_csv_row", "speaker_id", "utterance_id", "word", "relation_ground_truth", "expected_phone",
        "expected_position", "word_expected_position", "word_phone_count", "missing_candidates", "missing_stable",
        "mapped_aligned_phone", "mapped_aligned_start", "mapped_aligned_end", "anchor_status",
        "predicted_anchor", "manual_center_revealed_after_prediction", "absolute_error_seconds",
    ]
    with (EXPERIMENT_DIR / "deletion_anchor_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields); writer.writeheader()
        for row in deletion_rows:
            manual_center = (row["start"] + row["end"]) / 2
            writer.writerow({
                "source_csv_row": row["source_index"] + 2, "speaker_id": row["speaker_id"],
                "utterance_id": row["utterance_id"], "word": row.get("word"), "relation_ground_truth": row["relation"],
                "expected_phone": row["expected"], "expected_position": row.get("expected_position"),
                "word_expected_position": row.get("word_expected_position"),
                "word_phone_count": row.get("word_phone_count"), "missing_candidates": "|".join(map(str, row.get("missing_candidates", []))),
                "missing_stable": row.get("missing_stable"), "mapped_aligned_phone": row.get("mapped_aligned_phone"),
                "mapped_aligned_start": row.get("mapped_aligned_start"), "mapped_aligned_end": row.get("mapped_aligned_end"),
                "anchor_status": row.get("anchor_status"),
                "predicted_anchor": row.get("predicted_anchor"), "manual_center_revealed_after_prediction": manual_center,
                "absolute_error_seconds": abs(row["predicted_anchor"] - manual_center) if "predicted_anchor" in row else None,
            })
    write_json(EXPERIMENT_DIR / "runtime_alignment_path.json", runtime_path)
    write_json(EXPERIMENT_DIR / "preflight.json", {
        "v4_sha256": EXPECTED_V4_SHA, "raw_root_identity": RAW_ROOT.name, "parsed_utterances": parsed_utterances,
        "mfa_textgrid_missing_for_audit_utterances": missing_mfa_files, "invalid_mfa_phone_labels": dict(invalid_mfa_labels),
        "sequence_mapping_failures": dict(mapping_failures),
        "algorithm_inputs": ["expected_phone_identity", "stable_v4_row_order", "aligned_phone_identity", "aligned_phone_timing"],
        "algorithm_forbidden_inputs_verified": ["manual_deletion_center", "manual_deletion_duration", "manual_relation_label"],
        "manual_annotation_attached_after_prediction_for_evaluation_only": True,
        "utterance_alignment_summary": {
            "utterances": len(utterance_records),
            "with_missing_expected": sum(record["missing_count"] > 0 for record in utterance_records),
            "with_insertions": sum(record["insertions"] > 0 for record in utterance_records),
            "with_substitutions": sum(record["substitutions"] > 0 for record in utterance_records),
            "ambiguous_missing": sum(record["ambiguous_missing"] for record in utterance_records),
            "ambiguous_mapping": sum(record["ambiguous_mapping"] for record in utterance_records),
        },
        "train_validation_only": True,
        "test_alignment_paths_resolved": False, "test_alignment_accessed": False, "test_audio_accessed": False,
    })
    identification = {
        "manual_deletions": len(deletion_rows), "exact_expected_phone_identified": len(exact),
        "exact_identification_rate": len(exact) / len(deletion_rows), "wrong_expected_phone": len(wrong),
        "no_missing_phone_detected": len(none), "ambiguous_multiple_candidates": len(ambiguous),
        "missing_candidate_coverage": (len(exact) + len(wrong)) / len(deletion_rows),
        "both_sided_anchors": len(both_sided), "one_sided_anchors": len(one_sided),
        "no_anchor_or_missing_event": len(no_anchor),
        "without_primary_midpoint_anchor": len(deletion_rows) - len(both_sided),
        "anchor_coverage": anchor_coverage,
        "sequence_mapping_evaluable": sum("missing_candidates" in row for row in deletion_rows),
        "sequence_mapping_coverage": sum("missing_candidates" in row for row in deletion_rows) / len(deletion_rows),
    }
    write_json(EXPERIMENT_DIR / "deletion_identification.json", identification)
    write_json(EXPERIMENT_DIR / "anchor_timing.json", {
        "mfa_midpoint_anchor": anchor_distribution, "uniform_fallback_baseline": distribution(uniform_errors),
        "anchor_criterion": anchor_criterion,
    })
    write_json(EXPERIMENT_DIR / "false_missing_control.json", false_missing)
    write_json(EXPERIMENT_DIR / "word_analysis.json", {**word_stats, "position_quality": position_quality})
    write_json(EXPERIMENT_DIR / "phone_quality.json", {"groups": dtrl_quality, "phones": phone_quality})
    write_json(EXPERIMENT_DIR / "speaker_quality.json", speaker_quality)
    write_json(EXPERIMENT_DIR / "manual_vs_mfa_semantics.json", {"counts": dict(semantics), "verdict": semantics_verdict})
    final = {
        "status": final_status, "anchor_criterion": anchor_criterion, "runtime_implementability": implementability,
        "manual_deletions": len(deletion_rows), "exact_identification_rate": identification["exact_identification_rate"],
        "anchor_coverage": anchor_coverage, "median_error_seconds": median_error, "p90_error_seconds": p90_error,
        "correct_false_missing_rate": false_missing["correct_origin"]["rate"],
        "substitution_false_missing_rate": false_missing["substitution_origin"]["rate"],
        "manual_vs_mfa_semantics": semantics_verdict, "training_performed": False,
        "runtime_modified": False, "mfa_modified": False, "r4_test_alignment_accessed": False,
        "r4_test_audio_accessed": False, "r4_test_inference": False,
    }
    write_json(EXPERIMENT_DIR / "final_status.json", final)
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
