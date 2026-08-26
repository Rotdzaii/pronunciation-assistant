from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_l2_arctic_deletion_r4_1 as r4  # noqa: E402
import run_l2_arctic_observed_phone_r3_1a as r3  # noqa: E402


REPO_ROOT = r3.REPO_ROOT
R3D_DIR = REPO_ROOT / "ai-training/experiments/r3_1d_observed_phone_seed42_48epochs"
CHECKPOINT_PATH = R3D_DIR / "R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt"
EXPECTED_CHECKPOINT_SHA256 = "5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E"
R4_2A_DIR = REPO_ROOT / "ai-training/experiments/r4_2a_frozen_r3_expected_evidence"
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r4_2b_sequence_localization_feasibility"
OFFSETS_MS = (-100, -50, 0, 50, 100)
OFFSETS_SECONDS = tuple(value / 1000.0 for value in OFFSETS_MS)
CENTER_PROBE = 2
SEED = 42
MIN_SPEAKER_PAIRS = 20
DTRL = frozenset(("D", "T", "R", "L"))
IMPORTANT_PHONES = ("D", "T", "R", "L", "N", "Z", "V", "K", "AH", "HH", "S")
# Float32/CUDA tolerance for subset-batched probes versus the all-row R4-2A
# materialization. This guards model/preprocessing identity without requiring
# identical unrelated-row membership in each mel batch.
REPRODUCTION_TOLERANCE = 1e-4


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)), "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)), "p90": float(np.percentile(values, 90)),
    }


def auc_pr(truth: np.ndarray, deletion_oriented_score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc_deletion": float(roc_auc_score(truth, deletion_oriented_score)),
        "pr_auc_deletion": float(average_precision_score(truth, deletion_oriented_score)),
    }


def usable_neighbor(source: dict[str, str] | None) -> tuple[bool, str, str | None]:
    if source is None:
        return False, "utterance_edge", None
    relation = source.get("relation", "")
    if relation == "addition":
        return False, "addition_neighbor", None
    if relation == "non_speech":
        return False, "non_speech_neighbor", None
    if source.get("label_quality") != "clean" or relation not in {"correct", "substitution", "deletion"}:
        return False, "unresolved_neighbor", None
    phone = source.get("expected_phone_canonical", "")
    if phone not in r3.PHONE_TO_ID:
        return False, "invalid_expected_neighbor", None
    return True, "usable", phone


def attach_neighbors(split_rows: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    with r4.V4_PATH.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(source_rows):
        groups[(row["speaker_id"], row["utterance_id"])].append((index, row))
    adjacency: dict[int, tuple[dict[str, str] | None, dict[str, str] | None]] = {}
    for items in groups.values():
        items.sort(key=lambda item: (float(item[1]["start_time"]), float(item[1]["end_time"]), item[0]))
        for position, (source_index, _) in enumerate(items):
            adjacency[source_index] = (
                items[position - 1][1] if position else None,
                items[position + 1][1] if position + 1 < len(items) else None,
            )

    accounting: dict[str, Any] = {}
    excluded_records: list[dict[str, str]] = []
    for split in ("train", "validation"):
        relation_counts: dict[str, Any] = {}
        for relation in ("correct", "substitution", "deletion"):
            candidates = [row for row in split_rows[split] if row["relation"] == relation]
            reasons: Counter[str] = Counter()
            retained = 0
            for row in candidates:
                left, right = adjacency[row["_source_index"]]
                left_ok, left_reason, left_phone = usable_neighbor(left)
                right_ok, right_reason, right_phone = usable_neighbor(right)
                row["_left_expected_phone"] = left_phone
                row["_right_expected_phone"] = right_phone
                row["_usable_neighbors"] = left_ok and right_ok
                if row["_usable_neighbors"]:
                    retained += 1
                else:
                    if not left_ok:
                        reasons[f"left_{left_reason}"] += 1
                    if not right_ok:
                        reasons[f"right_{right_reason}"] += 1
                    excluded_records.append({
                        "source_csv_row": str(row["_source_index"] + 2), "split": split,
                        "speaker_id": row["speaker_id"], "relation": relation,
                        "left_reason": left_reason, "right_reason": right_reason,
                    })
            relation_counts[relation] = {
                "candidates": len(candidates), "retained": retained, "excluded": len(candidates) - retained,
                "exclusion_reasons_nonexclusive": dict(sorted(reasons.items())),
            }
        accounting[split] = relation_counts
    return accounting, excluded_records


def sample_validation_matched(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(row["speaker_id"], row["expected_phone_canonical"])][row["relation"]].append(row)
    rng = np.random.default_rng(SEED)
    selected: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    excluded_strata: list[dict[str, Any]] = []
    pair_id = 0
    for key in sorted(groups):
        substitutions = sorted(groups[key].get("substitution", []), key=lambda row: row["_source_index"])
        deletions = sorted(groups[key].get("deletion", []), key=lambda row: row["_source_index"])
        n = min(len(substitutions), len(deletions))
        if not n:
            excluded_strata.append({
                "speaker_id": key[0], "expected_phone": key[1],
                "substitutions": len(substitutions), "deletions": len(deletions), "reason": "one_class_absent",
            })
            continue
        sub_indexes = np.sort(rng.choice(len(substitutions), n, replace=False))
        del_indexes = np.sort(rng.choice(len(deletions), n, replace=False))
        for sub_index, del_index in zip(sub_indexes, del_indexes):
            pair_id += 1
            for row in (substitutions[int(sub_index)], deletions[int(del_index)]):
                row["_sequence_pair_id"] = pair_id
                selected.append(row)
                pair_records.append({
                    "pair_id": pair_id, "source_csv_row": row["_source_index"] + 2,
                    "speaker_id": row["speaker_id"], "expected_phone": row["expected_phone_canonical"],
                    "relation": row["relation"],
                })
        if len(substitutions) != len(deletions):
            excluded_strata.append({
                "speaker_id": key[0], "expected_phone": key[1],
                "substitutions": len(substitutions), "deletions": len(deletions),
                "matched_each": n, "reason": "unmatched_excess_rows",
            })
    source_ids = [row["_source_index"] for row in selected]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Duplicate row in sequence matched subset")
    summary = {
        "seed": SEED, "without_replacement": True, "matching": "validation speaker + expected phone; no duration",
        "pairs": pair_id, "rows": len(selected), "substitutions": sum(row["relation"] == "substitution" for row in selected),
        "deletions": sum(row["relation"] == "deletion" for row in selected),
        "phones": sorted({row["expected_phone_canonical"] for row in selected}),
        "speakers": sorted({row["speaker_id"] for row in selected}),
        "excluded_strata": excluded_strata, "records": pair_records,
    }
    return selected, summary


def sample_correct_reference(validation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in validation_rows:
        if row.get("_usable_neighbors"):
            groups[(row["speaker_id"], row["expected_phone_canonical"])][row["relation"]].append(row)
    rng = np.random.default_rng(SEED)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        correct = sorted(groups[key].get("correct", []), key=lambda row: row["_source_index"])
        primary_count = len(groups[key].get("substitution", [])) + len(groups[key].get("deletion", []))
        n = min(len(correct), primary_count)
        if n:
            indexes = np.sort(rng.choice(len(correct), n, replace=False))
            selected.extend(correct[int(index)] for index in indexes)
    return selected


class ProbeDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.store = r3.SequentialWaveStore()

    def __len__(self) -> int:
        return len(self.rows) * len(OFFSETS_SECONDS)

    def __getitem__(self, index: int):
        row_index, probe_index = divmod(index, len(OFFSETS_SECONDS))
        row = self.rows[row_index]
        audio = self.store.load(row["_audio_path"])
        center = (row["_start"] + row["_end"]) / 2 + OFFSETS_SECONDS[probe_index]
        waveform = r3.centered_window(audio, center, center)
        return torch.from_numpy(waveform), row_index, probe_index


def run_probes(rows: list[dict[str, Any]], model: torch.nn.Module, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    dataset = ProbeDataset(rows)
    # Reproduce the frozen R3 pipeline exactly: materialize log-mel at 128,
    # then run the acoustic model over the fixed features at batch size 8.
    loader = DataLoader(dataset, batch_size=r3.PREPROCESS_BATCH_SIZE, shuffle=False, num_workers=0)
    extractor = r3.FixedLogMel().to(device).eval()
    features = torch.empty((len(dataset), *r3.EXPECTED_FEATURE_SHAPE), dtype=torch.float32)
    cursor = 0
    with torch.no_grad():
        for waveforms, row_indexes, probe_indexes in tqdm(loader, desc="R4-2B materialize temporal probes"):
            batch_features = extractor(waveforms.to(device, non_blocking=True)).cpu()
            if tuple(batch_features.shape[1:]) != r3.EXPECTED_FEATURE_SHAPE:
                raise RuntimeError(f"Probe feature shape mismatch: {tuple(batch_features.shape[1:])}")
            expected_flat = row_indexes * 5 + probe_indexes
            actual_flat = torch.arange(cursor, cursor + len(waveforms))
            if not torch.equal(expected_flat, actual_flat):
                raise RuntimeError("Non-sequential temporal feature order")
            features[cursor:cursor + len(waveforms)].copy_(batch_features)
            cursor += len(waveforms)
    if cursor != len(dataset):
        raise RuntimeError("Incomplete temporal features")
    flat_logits = np.empty((len(dataset), 40), dtype=np.float32)
    feature_loader = DataLoader(features, batch_size=r3.BATCH_SIZE, shuffle=False, num_workers=0,
                                pin_memory=device.type == "cuda")
    cursor = 0
    with torch.no_grad():
        for batch_features in tqdm(feature_loader, desc="R4-2B frozen R3 inference"):
            output = model(batch_features.to(device, non_blocking=True)).cpu().numpy().astype(np.float32)
            flat_logits[cursor:cursor + len(output)] = output
            cursor += len(output)
    logits = flat_logits.reshape(len(rows), 5, 40)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy().astype(np.float32)
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite probe evidence")
    return logits, probabilities


def derive_scores(rows: list[dict[str, Any]], logits: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    count = len(rows)
    top1 = np.argmax(logits, axis=2)
    top3 = np.argsort(-logits, axis=2)[:, :, :3]
    expected_ids = np.asarray([r3.PHONE_TO_ID[row["expected_phone_canonical"]] for row in rows])
    left_ids = np.asarray([r3.PHONE_TO_ID[row["_left_expected_phone"]] for row in rows])
    right_ids = np.asarray([r3.PHONE_TO_ID[row["_right_expected_phone"]] for row in rows])
    row_indexes = np.arange(count)
    neighbor_prob = np.maximum(probabilities[row_indexes, :, left_ids], probabilities[row_indexes, :, right_ids])
    neighbor_capture = np.mean(neighbor_prob, axis=1)

    center_probabilities = probabilities[:, CENTER_PROBE, :].copy()
    center_probabilities[row_indexes, expected_ids] = -np.inf
    center_probabilities[row_indexes, left_ids] = -np.inf
    center_probabilities[row_indexes, right_ids] = -np.inf
    center_distinctness = np.max(center_probabilities, axis=1)

    without_expected = logits.copy()
    without_expected[row_indexes, :, expected_ids] = -np.inf
    alternative_top1 = np.argmax(without_expected, axis=2)
    modal_alternative = np.empty(count, dtype=np.int64)
    persistence = np.empty(count, dtype=np.float64)
    for index in range(count):
        values, counts = np.unique(alternative_top1[index], return_counts=True)
        maximum = np.max(counts)
        # Stable deterministic tie break: earliest occurrence among tied modes.
        tied = set(values[counts == maximum].tolist())
        modal = next(value for value in alternative_top1[index] if value in tied)
        modal_alternative[index] = modal
        persistence[index] = maximum / 5.0
    modal_category = np.where(modal_alternative == left_ids, "left", np.where(modal_alternative == right_ids, "right", "neither"))

    left_match = np.mean(top1[:, :2] == left_ids[:, None], axis=1)
    right_match = np.mean(top1[:, 3:] == right_ids[:, None], axis=1)
    transition = (left_match + right_match) / 2.0
    center_top1 = top1[:, CENTER_PROBE]
    central_novel = ~((center_top1 == expected_ids) | (center_top1 == left_ids) | (center_top1 == right_ids))
    expected_logits = logits[np.arange(count)[:, None], np.arange(5)[None, :], expected_ids[:, None]]
    alternatives = logits.copy()
    alternatives[np.arange(count)[:, None], np.arange(5)[None, :], expected_ids[:, None]] = -np.inf
    best_alt_ids = np.argmax(alternatives, axis=2)
    best_alt_logits = np.max(alternatives, axis=2)
    expected_margin = expected_logits - best_alt_logits
    return {
        "top1": top1, "top3": top3, "expected_ids": expected_ids, "left_ids": left_ids, "right_ids": right_ids,
        "neighbor_capture": neighbor_capture, "center_distinctness": center_distinctness,
        "alternative_persistence": persistence, "modal_alternative": modal_alternative,
        "modal_category": modal_category, "left_match": left_match, "right_match": right_match,
        "left_right_transition": transition, "central_novel": central_novel,
        "best_alt_ids": best_alt_ids, "expected_margin": expected_margin,
    }


def score_report(raw: np.ndarray, truth: np.ndarray, deletion_high: bool) -> dict[str, Any]:
    oriented = raw if deletion_high else -raw
    return {
        **auc_pr(truth, oriented), "deletion_high_orientation": deletion_high,
        "substitution": distribution(raw[truth == 0]), "deletion": distribution(raw[truth == 1]),
    }


def pattern_counts(scores: dict[str, Any], positions: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    top1 = scores["top1"][positions]
    expected = scores["expected_ids"][positions]
    left = scores["left_ids"][positions]
    right = scores["right_ids"][positions]
    modal_category = scores["modal_category"][positions]
    persistence = scores["alternative_persistence"][positions]
    patterns = {
        "center_top1_target": top1[:, 2] == expected,
        "center_top1_left": top1[:, 2] == left,
        "center_top1_right": top1[:, 2] == right,
        "center_top1_novel": scores["central_novel"][positions],
        "same_novel_alternative_persists_ge_3_of_5": (modal_category == "neither") & (persistence >= 0.6),
        "left_to_right_neighbor_transition": (scores["left_match"][positions] > 0) & (scores["right_match"][positions] > 0),
        "single_neighbor_dominates_ge_3_of_5": np.maximum(np.sum(top1 == left[:, None], axis=1),
                                                          np.sum(top1 == right[:, None], axis=1)) >= 3,
        "expected_phone_dominates_ge_3_of_5": np.sum(top1 == expected[:, None], axis=1) >= 3,
    }
    result: dict[str, Any] = {}
    for name, mask in patterns.items():
        result[name] = {}
        for label, relation in ((0, "substitution"), (1, "deletion")):
            class_mask = truth == label
            count = int(np.sum(mask[class_mask]))
            support = int(np.sum(class_mask))
            result[name][relation] = {"count": count, "support": support, "rate": count / support if support else None}
    return result


def write_probe_summary(path: Path, rows: list[dict[str, Any]], logits: np.ndarray,
                        probabilities: np.ndarray, scores: dict[str, Any]) -> None:
    fields = [
        "source_csv_row", "speaker_id", "relation", "offset_ms", "top1_phone", "top1_probability",
        "top3_phones", "top3_probabilities", "expected_phone", "expected_logit", "expected_margin",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            expected_id = scores["expected_ids"][index]
            for probe, offset_ms in enumerate(OFFSETS_MS):
                top3_ids = scores["top3"][index, probe]
                writer.writerow({
                    "source_csv_row": row["_source_index"] + 2, "speaker_id": row["speaker_id"],
                    "relation": row["relation"], "offset_ms": offset_ms,
                    "top1_phone": r3.PHONE_VOCAB[scores["top1"][index, probe]],
                    "top1_probability": probabilities[index, probe, scores["top1"][index, probe]],
                    "top3_phones": "|".join(r3.PHONE_VOCAB[value] for value in top3_ids),
                    "top3_probabilities": "|".join(str(float(probabilities[index, probe, value])) for value in top3_ids),
                    "expected_phone": row["expected_phone_canonical"],
                    "expected_logit": logits[index, probe, expected_id],
                    "expected_margin": scores["expected_margin"][index, probe],
                })


def main() -> int:
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite audit directory: {EXPERIMENT_DIR}")
    checkpoint_sha = r3.sha256_file(CHECKPOINT_PATH)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA mismatch: {checkpoint_sha}")
    r3.set_seed(SEED)
    audio_root = r3.require_audio_root()
    split_rows, source_summary, _ = r4.load_and_verify(audio_root)
    adjacency_accounting, excluded_records = attach_neighbors(split_rows)
    train_primary = [row for row in split_rows["train"] if row["relation"] in {"substitution", "deletion"} and row["_usable_neighbors"]]
    validation_primary = [row for row in split_rows["validation"] if row["relation"] in {"substitution", "deletion"} and row["_usable_neighbors"]]
    matched_rows, matched_summary = sample_validation_matched(validation_primary)
    correct_reference = sample_correct_reference(split_rows["validation"])
    audit_rows = train_primary + validation_primary + correct_reference
    if len({row["_source_index"] for row in audit_rows}) != len(audit_rows):
        raise RuntimeError("Audit row duplication")
    for index, row in enumerate(audit_rows):
        row["_audit_index"] = index
    audio_preflight = r3.preflight_audio(audit_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if int(checkpoint["epoch"]) != 47:
        raise RuntimeError(f"Frozen checkpoint epoch mismatch: {checkpoint['epoch']}")
    model = r3.SmallPronunciationCNNAttention().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    started = time.perf_counter()
    logits, probabilities = run_probes(audit_rows, model, device)
    inference_seconds = time.perf_counter() - started
    scores = derive_scores(audit_rows, logits, probabilities)

    # Center probes must exactly reproduce the already audited frozen R4-2A margins.
    reference_by_source: dict[int, float] = {}
    for filename in ("train_phone_evidence.csv", "validation_phone_evidence.csv"):
        with (R4_2A_DIR / filename).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                reference_by_source[int(row["source_csv_row"]) - 2] = float(row["expected_margin"])
    primary_count = len(train_primary) + len(validation_primary)
    center_margins = scores["expected_margin"][:primary_count, CENTER_PROBE]
    reference_margins = np.asarray([reference_by_source[row["_source_index"]] for row in audit_rows[:primary_count]])
    reproduction_max_delta = float(np.max(np.abs(center_margins - reference_margins)))
    if reproduction_max_delta > REPRODUCTION_TOLERANCE:
        raise RuntimeError(f"Center probe reproduction failed: max delta {reproduction_max_delta}")

    train_positions = np.arange(len(train_primary))
    validation_start = len(train_primary)
    validation_positions = np.arange(validation_start, validation_start + len(validation_primary))
    correct_positions = np.arange(validation_start + len(validation_primary), len(audit_rows))
    matched_positions = np.asarray([row["_audit_index"] for row in matched_rows])
    matched_truth = np.asarray([1 if audit_rows[index]["relation"] == "deletion" else 0 for index in matched_positions])
    if len(matched_truth) != matched_summary["rows"] or np.sum(matched_truth) != matched_summary["pairs"]:
        raise RuntimeError("Matched sequence position mismatch")

    train_matrix = np.column_stack([
        scores["neighbor_capture"][train_positions], scores["left_right_transition"][train_positions],
        scores["center_distinctness"][train_positions],
    ])
    standard_mean = train_matrix.mean(axis=0)
    standard_std = train_matrix.std(axis=0)
    if np.any(standard_std <= 0):
        raise RuntimeError("Degenerate TRAIN standardization")
    combined = (
        (scores["neighbor_capture"] - standard_mean[0]) / standard_std[0]
        + (scores["left_right_transition"] - standard_mean[1]) / standard_std[1]
        - (scores["center_distinctness"] - standard_mean[2]) / standard_std[2]
    )
    score_specs: dict[str, tuple[np.ndarray, bool]] = {
        "neighbor_capture": (scores["neighbor_capture"], True),
        "center_distinctness": (scores["center_distinctness"], False),
        "alternative_persistence": (scores["alternative_persistence"], False),
        "left_right_transition": (scores["left_right_transition"], True),
    }
    primary_reports = {
        name: score_report(values[matched_positions], matched_truth, deletion_high)
        for name, (values, deletion_high) in score_specs.items()
    }
    combined_report = score_report(combined[matched_positions], matched_truth, True)

    speaker_results: dict[str, Any] = {}
    for speaker in r4.VALIDATION_SPEAKERS:
        positions = np.asarray([index for index in matched_positions if audit_rows[index]["speaker_id"] == speaker])
        truth = np.asarray([1 if audit_rows[index]["relation"] == "deletion" else 0 for index in positions])
        pairs = int(np.sum(truth))
        item: dict[str, Any] = {"substitutions": pairs, "deletions": pairs, "sufficient_support": pairs >= MIN_SPEAKER_PAIRS}
        if pairs:
            item["scores"] = {
                name: auc_pr(truth, values[positions] if deletion_high else -values[positions])
                for name, (values, deletion_high) in score_specs.items()
            }
            item["combined"] = auc_pr(truth, combined[positions])
        speaker_results[speaker] = item
    for name, report in primary_reports.items():
        consistent = sum(
            item.get("scores", {}).get(name, {}).get("roc_auc_deletion", 0) >= 0.50
            for item in speaker_results.values() if item["sufficient_support"]
        )
        eligible = sum(item["sufficient_support"] for item in speaker_results.values())
        report["speaker_direction_consistency"] = {"consistent": consistent, "eligible": eligible,
                                                     "minimum_pairs_per_class": MIN_SPEAKER_PAIRS}

    strong_scores = [
        name for name, report in primary_reports.items()
        if report["roc_auc_deletion"] >= 0.70 and report["pr_auc_deletion"] >= 0.65
        and report["speaker_direction_consistency"]["consistent"] >= 4
    ]
    best_name = max(primary_reports, key=lambda name: primary_reports[name]["roc_auc_deletion"])
    best_auc = primary_reports[best_name]["roc_auc_deletion"]
    if strong_scores:
        signal_gate = "SEQUENCE_SIGNAL_STRONG"
    elif best_auc >= 0.62:
        signal_gate = "SEQUENCE_SIGNAL_MODERATE"
    else:
        signal_gate = "SEQUENCE_SIGNAL_WEAK"

    # Substitution observed-phone diagnostics use labels only after all scores are fixed.
    validation_sub_positions = np.asarray([
        index for index in validation_positions if audit_rows[index]["relation"] == "substitution"
    ])
    observed_ids = np.asarray([r3.PHONE_TO_ID[audit_rows[index]["observed_phone_canonical"]] for index in validation_sub_positions])
    substitution_diagnostic = {
        "rows": int(len(validation_sub_positions)),
        "center_top1_observed_accuracy": float(np.mean(scores["top1"][validation_sub_positions, 2] == observed_ids)),
        "center_top3_observed_accuracy": float(np.mean([
            observed in guesses for observed, guesses in zip(observed_ids, scores["top3"][validation_sub_positions, 2])
        ])),
        "modal_alternative_observed_accuracy": float(np.mean(scores["modal_alternative"][validation_sub_positions] == observed_ids)),
    }

    relation_comparison: dict[str, Any] = {}
    for relation, positions in {
        "correct_reference": correct_positions,
        "substitution": np.asarray([index for index in validation_positions if audit_rows[index]["relation"] == "substitution"]),
        "deletion": np.asarray([index for index in validation_positions if audit_rows[index]["relation"] == "deletion"]),
    }.items():
        relation_comparison[relation] = {
            name: distribution(values[positions]) for name, (values, _) in score_specs.items()
        }
        relation_comparison[relation]["combined"] = distribution(combined[positions])

    group_results: dict[str, Any] = {}
    for group, predicate in {
        "D_T_R_L": lambda phone: phone in DTRL,
        "other_phones": lambda phone: phone not in DTRL,
    }.items():
        positions = np.asarray([index for index in matched_positions if predicate(audit_rows[index]["expected_phone_canonical"])])
        truth = np.asarray([1 if audit_rows[index]["relation"] == "deletion" else 0 for index in positions])
        group_results[group] = {
            "rows": int(len(positions)), "pairs": int(np.sum(truth)),
            "scores": {name: auc_pr(truth, values[positions] if deletion_high else -values[positions])
                       for name, (values, deletion_high) in score_specs.items()} if len(np.unique(truth)) == 2 else {},
            "combined": auc_pr(truth, combined[positions]) if len(np.unique(truth)) == 2 else None,
        }

    phone_results: dict[str, Any] = {}
    matched_phone_counts = Counter(audit_rows[index]["expected_phone_canonical"] for index in matched_positions
                                   if audit_rows[index]["relation"] == "deletion")
    phones_to_report = sorted(set(IMPORTANT_PHONES) | {phone for phone, count in matched_phone_counts.items() if count >= 10})
    for phone in phones_to_report:
        positions = np.asarray([index for index in matched_positions if audit_rows[index]["expected_phone_canonical"] == phone])
        truth = np.asarray([1 if audit_rows[index]["relation"] == "deletion" else 0 for index in positions])
        item: dict[str, Any] = {"pairs": int(np.sum(truth)), "rows": int(len(positions)), "sufficient": int(np.sum(truth)) >= 10}
        if len(np.unique(truth)) == 2:
            item["scores"] = {name: auc_pr(truth, values[positions] if deletion_high else -values[positions])
                              for name, (values, deletion_high) in score_specs.items()}
            item["combined"] = auc_pr(truth, combined[positions])
        phone_results[phone] = item

    temporal_patterns = pattern_counts(scores, matched_positions, matched_truth)
    novel_rates = temporal_patterns["center_top1_novel"]
    best_group_aucs = {
        group: payload.get("scores", {}).get(best_name, {}).get("roc_auc_deletion")
        for group, payload in group_results.items()
    }
    runtime_anchor = {
        "status": "RUNTIME_ANCHOR_UNCERTAIN",
        "evidence": [
            "MFA aligns known transcript and can emit phone intervals, so an approximate canonical slot may exist.",
            "Current scorer explicitly handles MFA output that may omit a phone and refuses positional association.",
            "Current contract has no explicit missing-expected-phone slot or gap-to-canonical-index mapping.",
            "Fallback evenly splits canonical phones but project documentation marks phone timing reliability as limited.",
        ],
        "source_paths": [
            "ai-training/docs/AI_PHASE4A_FORCED_ALIGNMENT_PLAN.md",
            "ai-worker/app/alignment/fallback_aligner.py",
            "ai-worker/app/scorers/cnn_attention_scorer.py",
        ],
    }
    if signal_gate == "SEQUENCE_SIGNAL_WEAK":
        verdict = "SEQUENCE_LOCALIZATION_NOT_SUPPORTED"
    elif (
        signal_gate == "SEQUENCE_SIGNAL_STRONG"
        or (best_auc >= 0.65 and combined_report["roc_auc_deletion"] >= 0.65
            and all(value is not None and value >= 0.60 for value in best_group_aucs.values()))
    ) and runtime_anchor["status"] != "RUNTIME_ANCHOR_BLOCKER":
        verdict = "SEQUENCE_LOCALIZATION_PROMISING"
    else:
        verdict = "SEQUENCE_LOCALIZATION_NEEDS_MORE_EVIDENCE"

    EXPERIMENT_DIR.mkdir(parents=True)
    with (EXPERIMENT_DIR / "excluded_neighbor_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(excluded_records[0]) if excluded_records else ["source_csv_row"])
        writer.writeheader(); writer.writerows(excluded_records)
    with (EXPERIMENT_DIR / "matched_substitution_deletion_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matched_summary["records"][0]))
        writer.writeheader(); writer.writerows(matched_summary["records"])
    np.savez_compressed(
        EXPERIMENT_DIR / "temporal_probe_evidence.npz", source_csv_rows=np.asarray([row["_source_index"] + 2 for row in audit_rows]),
        offsets_ms=np.asarray(OFFSETS_MS), logits=logits, probabilities=probabilities,
    )
    write_probe_summary(EXPERIMENT_DIR / "temporal_probe_summary.csv", audit_rows, logits, probabilities, scores)
    summary_fields = [
        "source_csv_row", "partition", "speaker_id", "relation", "left_expected_phone", "target_expected_phone",
        "right_expected_phone", "neighbor_capture", "center_distinctness", "alternative_persistence",
        "modal_alternative_phone", "modal_alternative_category", "left_match", "right_match",
        "left_right_transition", "central_novel", "combined_sequence_score", "is_primary_matched",
    ]
    matched_sources = {row["_source_index"] for row in matched_rows}
    with (EXPERIMENT_DIR / "row_sequence_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields); writer.writeheader()
        for index, row in enumerate(audit_rows):
            partition = "train_primary" if index < validation_start else ("validation_primary" if index < correct_positions[0] else "validation_correct_reference")
            writer.writerow({
                "source_csv_row": row["_source_index"] + 2, "partition": partition, "speaker_id": row["speaker_id"],
                "relation": row["relation"], "left_expected_phone": row["_left_expected_phone"],
                "target_expected_phone": row["expected_phone_canonical"], "right_expected_phone": row["_right_expected_phone"],
                "neighbor_capture": scores["neighbor_capture"][index], "center_distinctness": scores["center_distinctness"][index],
                "alternative_persistence": scores["alternative_persistence"][index],
                "modal_alternative_phone": r3.PHONE_VOCAB[scores["modal_alternative"][index]],
                "modal_alternative_category": scores["modal_category"][index], "left_match": scores["left_match"][index],
                "right_match": scores["right_match"][index], "left_right_transition": scores["left_right_transition"][index],
                "central_novel": bool(scores["central_novel"][index]), "combined_sequence_score": combined[index],
                "is_primary_matched": row["_source_index"] in matched_sources,
            })

    config = {
        "experiment": "R4-2B Sequence/Localization Feasibility Audit",
        "markers": ["RESEARCH_ONLY", "AUDIT_ONLY", "NO_TRAINING", "R4_TEST_CLOSED"],
        "checkpoint_sha256": checkpoint_sha, "offsets_ms": list(OFFSETS_MS),
        "crop_seconds_each_probe": 0.50, "feature_shape": list(r3.EXPECTED_FEATURE_SHAPE),
        "matched_speaker_sufficient_pairs": MIN_SPEAKER_PAIRS,
        "score_orientations": {"neighbor_capture": "deletion_high", "center_distinctness": "substitution_high",
                               "alternative_persistence": "substitution_high", "left_right_transition": "deletion_high"},
        "combined": "+z_train(neighbor_capture)+z_train(transition)-z_train(center_distinctness); weights fixed 1",
        "verdict_rule": "PROMISING if STRONG, or MODERATE with best and combined AUC>=.65 and both phone groups best-score AUC>=.60; runtime not blocker; WEAK=>NOT_SUPPORTED",
        "test_audio_paths_resolved": False, "test_audio_accessed": False, "training_performed": False,
    }
    write_json(EXPERIMENT_DIR / "config.json", config)
    write_json(EXPERIMENT_DIR / "preflight.json", {
        "checkpoint_sha256": checkpoint_sha, "source": source_summary, "audio": audio_preflight,
        "probe_shape": [len(audit_rows), 5, 40], "feature_shape": list(r3.EXPECTED_FEATURE_SHAPE),
        "center_reproduction_max_abs_delta": reproduction_max_delta, "center_reproduction_status": "PASS",
        "inference_seconds": inference_seconds, "test_audio_paths_resolved": False, "test_audio_accessed": False,
    })
    write_json(EXPERIMENT_DIR / "neighbor_accounting.json", adjacency_accounting)
    write_json(EXPERIMENT_DIR / "matched_subset.json", {key: value for key, value in matched_summary.items() if key != "records"})
    write_json(EXPERIMENT_DIR / "train_standardization.json", {
        "features": ["neighbor_capture", "left_right_transition", "center_distinctness"],
        "mean": standard_mean.tolist(), "std": standard_std.tolist(), "train_rows": len(train_primary),
    })
    write_json(EXPERIMENT_DIR / "primary_score_diagnostics.json", primary_reports)
    write_json(EXPERIMENT_DIR / "combined_score_diagnostic.json", combined_report)
    write_json(EXPERIMENT_DIR / "substitution_observed_phone_diagnostic.json", substitution_diagnostic)
    write_json(EXPERIMENT_DIR / "relation_reference_comparison.json", relation_comparison)
    write_json(EXPERIMENT_DIR / "phone_group_diagnostics.json", group_results)
    write_json(EXPERIMENT_DIR / "per_phone_diagnostics.json", phone_results)
    write_json(EXPERIMENT_DIR / "per_speaker_diagnostics.json", speaker_results)
    write_json(EXPERIMENT_DIR / "temporal_pattern_counts.json", temporal_patterns)
    write_json(EXPERIMENT_DIR / "runtime_anchor_assessment.json", runtime_anchor)
    final = {
        "signal_gate": signal_gate, "best_predefined_score": best_name, "best_matched_roc_auc": best_auc,
        "strong_scores": strong_scores, "combined_roc_auc": combined_report["roc_auc_deletion"],
        "combined_pr_auc": combined_report["pr_auc_deletion"], "central_novel_rates": novel_rates,
        "runtime_anchor": runtime_anchor["status"], "formulation_verdict": verdict,
        "training_performed": False, "r4_test_audio_paths_resolved": False,
        "r4_test_audio_accessed": False, "r4_test_inference": False,
    }
    write_json(EXPERIMENT_DIR / "final_status.json", final)
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
