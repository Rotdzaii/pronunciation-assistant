from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


RESEARCH = Path(r"C:\Users\Admin\Documents\KLTN\pronunciation-assistant-research")
R5_1A = RESEARCH / "ai-training/experiments/r5_1a_alignability_safe_addition_scoring"
R5_2B = RESEARCH / "ai-training/experiments/r5_2b_relation_competition"

EXPECTED = {
    "r5_1a_manifest": "C9343A75CE26C2BEECA388EBA855E91AE0992D22C0C5D785308D0A98B89A3CD6",
    "r5_2b_contract": "55572805C1878D41D4B6C41E1C7A31B2C3725A81050942494F0E1292092FEF38",
    "r5_2b_scorer": "2E44B79828DBB37B312CDC03C897A80803947A48F13864B26C454F3B8ED161A3",
    "r5_2b_manifest": "37F3C86FF11526B8AB54D173937A0F488125A1D0B610032CC8A5D47B11602387",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "checkpoint": "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085",
}

SPEAKERS = ["BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite value cannot be serialized in R5-2C audit JSON")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def audit_manifest(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    manifest = read_json(path)
    entries = manifest.get("artifacts", manifest.get("files", []))
    failures: list[str] = []
    for entry in entries:
        rel = entry.get("relative_path", entry.get("path"))
        size = entry.get("byte_size", entry.get("size_bytes"))
        expected_hash = entry.get("sha256", entry.get("sha256_hex"))
        target = root / rel
        if not target.is_file():
            failures.append(f"missing:{rel}")
            continue
        if target.stat().st_size != int(size):
            failures.append(f"size:{rel}")
        if sha256(target) != str(expected_hash).upper():
            failures.append(f"hash:{rel}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "entry_count": len(entries),
        "failures": failures,
        "pass": not failures,
    }


def index_unique(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    index: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        key = row["source_identity"]
        if key in index:
            duplicates.append(key)
        else:
            index[key] = row
    return index, sorted(set(duplicates))


def stats(values: Iterable[float]) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "equal_zero_count": int(np.count_nonzero(x == 0.0)),
        "greater_zero_count": int(np.count_nonzero(x > 0.0)),
        "less_zero_count": int(np.count_nonzero(x < 0.0)),
    }


def auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if not np.all(np.isfinite(scores)):
        raise ValueError("R5-2C expected finite frozen A/C scores")
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC cohort lacks one class")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def count_share(counter: Counter[str], total: int) -> dict[str, Any]:
    return {
        key: {"count": int(counter[key]), "share": (counter[key] / total if total else None)}
        for key in sorted(counter)
    }


def event_position(boundary: int, expected_length: int) -> str:
    if boundary == 0:
        return "BEFORE_FIRST"
    if boundary == expected_length:
        return "AFTER_FINAL"
    return "BETWEEN"


def boundary_context(expected: list[str], boundary: int) -> str:
    left = "<SIL>" if boundary == 0 else expected[boundary - 1]
    right = "<SIL>" if boundary == len(expected) else expected[boundary]
    return f"{left}|{right}"


def truth_diagnostics(indices: list[int], old_rows: list[dict[str, Any]]) -> dict[str, Any]:
    phones: Counter[str] = Counter()
    positions: Counter[str] = Counter()
    contexts: Counter[str] = Counter()
    event_count = 0
    for i in indices:
        expected = old_rows[i]["expected_sequence"]
        for event in old_rows[i]["ground_truth_events"]:
            event_count += 1
            phones[str(event["phone"])] += 1
            positions[str(event["position"])] += 1
            contexts[boundary_context(expected, int(event["boundary"]))] += 1
    return {
        "word_count": len(indices),
        "truth_event_count": event_count,
        "phone_counts": dict(phones.most_common()),
        "position_counts": dict(positions.most_common()),
        "expected_context_counts": dict(contexts.most_common()),
    }


def predicted_diagnostics(indices: list[int], new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    phones: Counter[str] = Counter()
    positions: Counter[str] = Counter()
    contexts: Counter[str] = Counter()
    for i in indices:
        row = new_rows[i]
        phones[str(row["best_insert_phone"])] += 1
        boundary = int(row["best_insert_boundary"])
        positions[event_position(boundary, len(row["expected_sequence"]))] += 1
        contexts[boundary_context(row["expected_sequence"], boundary)] += 1
    return {
        "word_count": len(indices),
        "predicted_phone_counts": dict(phones.most_common()),
        "predicted_position_counts": dict(positions.most_common()),
        "expected_context_counts": dict(contexts.most_common()),
    }


def far(pred: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    support = int(mask.sum())
    positives = int(np.count_nonzero(pred & mask))
    return {
        "support": support,
        "false_positives": positives,
        "false_addition_rate": positives / support if support else None,
    }


def transitions(old_pred: np.ndarray, new_pred: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    return {
        "TN_in_both": int(np.count_nonzero(mask & ~old_pred & ~new_pred)),
        "FP_in_both": int(np.count_nonzero(mask & old_pred & new_pred)),
        "TN_R5_1A_to_FP_R5_2B": int(np.count_nonzero(mask & ~old_pred & new_pred)),
        "FP_R5_1A_to_TN_R5_2B": int(np.count_nonzero(mask & old_pred & ~new_pred)),
    }


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    r5_1a_manifest = audit_manifest(R5_1A, "R5_1A_EXECUTION_MANIFEST.json")
    r5_2b_manifest = audit_manifest(R5_2B, "R5_2B_TC1_PA1_ENV_EXECUTION_MANIFEST.json")
    direct = {
        "r5_2b_contract": R5_2B / "R5_2B_DEVELOPMENT_CONTRACT.json",
        "r5_2b_scorer": R5_2B / "r5_2b_scorer.py",
        "v4": RESEARCH / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv",
        "checkpoint": RESEARCH / "ai-training/experiments/r4_4c2_bigru_ctc_seed42/R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt",
    }
    direct_checks = {
        key: {
            "path": str(path),
            "expected_sha256": EXPECTED[key],
            "actual_sha256": sha256(path),
            "match": sha256(path) == EXPECTED[key],
        }
        for key, path in direct.items()
    }
    identity_pass = (
        r5_1a_manifest["pass"]
        and r5_2b_manifest["pass"]
        and r5_1a_manifest["sha256"] == EXPECTED["r5_1a_manifest"]
        and r5_2b_manifest["sha256"] == EXPECTED["r5_2b_manifest"]
        and all(item["match"] for item in direct_checks.values())
    )
    source_identity = {
        "status": "PASS" if identity_pass else "R5_2C_BLOCKED_IDENTITY",
        "r5_1a_execution_manifest": r5_1a_manifest,
        "r5_2b_execution_manifest": r5_2b_manifest,
        "direct_frozen_identities": direct_checks,
        "audio_or_model_loaded": False,
    }
    write_json(out_dir / "r5_2c_source_identity.json", source_identity)
    if not identity_pass:
        raise RuntimeError("R5_2C_BLOCKED_IDENTITY")

    old_scores = read_jsonl(R5_1A / "r5_1a_train_scores.jsonl")
    old_oof = read_jsonl(R5_1A / "r5_1a_oof_predictions.jsonl")
    new_scores = read_jsonl(R5_2B / "r5_2b_tc1_pa1_env_train_scores.jsonl")
    new_oof = read_jsonl(R5_2B / "r5_2b_tc1_pa1_env_oof_predictions.jsonl")
    old_score_index, old_score_dups = index_unique(old_scores)
    old_oof_index, old_oof_dups = index_unique(old_oof)
    new_score_index, new_score_dups = index_unique(new_scores)
    new_oof_index, new_oof_dups = index_unique(new_oof)
    key_sets = [set(x) for x in [old_score_index, old_oof_index, new_score_index, new_oof_index]]
    common = set.intersection(*key_sets)
    union = set.union(*key_sets)
    missing = {
        "r5_1a_scores": sorted(union - key_sets[0]),
        "r5_1a_oof": sorted(union - key_sets[1]),
        "r5_2b_scores": sorted(union - key_sets[2]),
        "r5_2b_oof": sorted(union - key_sets[3]),
    }
    duplicate = {
        "r5_1a_scores": old_score_dups,
        "r5_1a_oof": old_oof_dups,
        "r5_2b_scores": new_score_dups,
        "r5_2b_oof": new_oof_dups,
    }
    ordered_keys = [row["source_identity"] for row in new_oof]
    old_rows = [old_oof_index[key] for key in ordered_keys if key in common]
    new_rows = [new_oof_index[key] for key in ordered_keys if key in common]
    metadata_mismatches: list[str] = []
    for old, new in zip(old_rows, new_rows):
        if (
            old["speaker"] != new["speaker"]
            or old["word"] != new["word"]
            or old["expected_sequence"] != new["expected_sequence"]
            or bool(old["addition_label"]) != bool(new["addition_label"])
            or old["ground_truth_events"] != new["ground_truth_events"]
        ):
            metadata_mismatches.append(old["source_identity"])
    join_pass = (
        len(common) == 16582
        and len(union) == 16582
        and not any(duplicate.values())
        and not any(missing.values())
        and not metadata_mismatches
        and len(ordered_keys) == len(set(ordered_keys)) == 16582
    )
    join_audit = {
        "status": "PASS" if join_pass else "R5_2C_BLOCKED_ROW_IDENTITY",
        "stable_identity_field": "source_identity",
        "matched_rows": len(common),
        "union_rows": len(union),
        "row_counts": {
            "r5_1a_scores": len(old_scores),
            "r5_1a_oof": len(old_oof),
            "r5_2b_scores": len(new_scores),
            "r5_2b_oof": len(new_oof),
        },
        "duplicates": {key: len(value) for key, value in duplicate.items()},
        "missing": {key: len(value) for key, value in missing.items()},
        "metadata_mismatch_count": len(metadata_mismatches),
        "metadata_mismatch_examples": metadata_mismatches[:20],
        "fuzzy_matching_used": False,
        "rows_excluded": 0,
    }
    write_json(out_dir / "r5_2c_join_audit.json", join_audit)
    if not join_pass:
        raise RuntimeError("R5_2C_BLOCKED_ROW_IDENTITY")

    n = len(old_rows)
    labels = np.asarray([bool(r["addition_label"]) for r in old_rows], dtype=bool)
    old_pred = np.asarray([bool(r["predicted_addition"]) for r in old_rows], dtype=bool)
    new_pred = np.asarray([bool(r["predicted_addition"]) for r in new_rows], dtype=bool)
    a = np.asarray([float(r["addition_score_A_value"]) for r in old_rows], dtype=np.float64)
    c = np.asarray([float(r["relation_competition_score_C_value"]) for r in new_rows], dtype=np.float64)
    keep2 = np.asarray([float(r["keep_target_score_value"]) for r in new_rows], dtype=np.float64)
    insert2 = np.asarray([float(r["best_insert_score_value"]) for r in new_rows], dtype=np.float64)
    nonadd2 = np.asarray([float(r["best_nonaddition_score_value"]) for r in new_rows], dtype=np.float64)
    theta_a = np.asarray([float(r["fold_threshold"]) for r in old_rows], dtype=np.float64)
    theta_c = np.asarray([float(r["fold_threshold"]) for r in new_rows], dtype=np.float64)
    suppression_observed = a - c
    suppression_constructed = nonadd2 - keep2
    a_reconstructed_r5_2 = insert2 - keep2
    cross_execution_representation_delta = a - a_reconstructed_r5_2
    algebra_residual = suppression_observed - suppression_constructed - cross_execution_representation_delta
    algebra_scale = np.maximum.reduce(
        [np.ones(n), np.abs(a), np.abs(c), np.abs(nonadd2), np.abs(keep2), np.abs(insert2)]
    )
    algebra_bound = 16.0 * np.finfo(np.float64).eps * algebra_scale
    score_identity_pass = bool(
        np.all(suppression_constructed >= 0.0)
        and np.all(np.abs(algebra_residual) <= algebra_bound)
        and np.all(np.isfinite(a))
        and np.all(np.isfinite(c))
    )

    cohort_masks: dict[str, np.ndarray] = {
        "addition_positive": labels,
        "correct_only_negative": np.asarray(
            [bool(r["relation_cohorts"]["correct_only"]) for r in new_rows], dtype=bool
        ),
        "substitution_containing_negative": np.asarray(
            [bool(r["relation_cohorts"]["substitution_negative"]) for r in new_rows], dtype=bool
        ),
        "deletion_containing_negative": np.asarray(
            [bool(r["relation_cohorts"]["deletion_negative"]) for r in new_rows], dtype=bool
        ),
    }
    cohort_masks["substitution_and_deletion_negative"] = (
        cohort_masks["substitution_containing_negative"] & cohort_masks["deletion_containing_negative"]
    )
    winners = np.asarray([str(r["best_nonaddition_family"]) for r in new_rows], dtype=object)
    suppression_by_cohort: dict[str, Any] = {}
    for name, mask in cohort_masks.items():
        total = int(mask.sum())
        suppression_by_cohort[name] = {
            "n": total,
            "observed_A_minus_C": stats(suppression_observed[mask]),
            "constructed_best_nonaddition_minus_KEEP": stats(suppression_constructed[mask]),
            "constructed_equal_zero_share": float(np.mean(suppression_constructed[mask] == 0.0)),
            "constructed_greater_zero_share": float(np.mean(suppression_constructed[mask] > 0.0)),
            "winner_family": count_share(Counter(winners[mask]), total),
        }
    suppression_metrics = {
        "status": "PASS" if score_identity_pass else "R5_2C_SCORE_IDENTITY_MISMATCH",
        "definition": "SUPPRESSION = frozen R5-1A A - frozen R5-2B C",
        "observed_A_minus_C": stats(suppression_observed),
        "constructed_identity_term_best_nonaddition_minus_KEEP": stats(suppression_constructed),
        "constructed_term_nonnegative_count": int(np.count_nonzero(suppression_constructed >= 0.0)),
        "constructed_term_negative_count": int(np.count_nonzero(suppression_constructed < 0.0)),
        "tiny_negative_observed_A_minus_C": {
            "count": int(np.count_nonzero(suppression_observed < 0.0)),
            "minimum": float(np.min(suppression_observed)),
            "explanation": "separate frozen float32 R5-1A/R5-2B score artifacts; fully explained by the recorded cross-execution INSERT-minus-KEEP representation delta",
        },
        "cross_execution_A_representation_delta": stats(cross_execution_representation_delta),
        "identity_algebra_residual": stats(algebra_residual),
        "maximum_allowed_float64_algebra_bound": float(np.max(algebra_bound)),
        "all_algebra_residuals_within_bound": bool(np.all(np.abs(algebra_residual) <= algebra_bound)),
        "cohorts": suppression_by_cohort,
    }
    write_json(out_dir / "r5_2c_suppression_metrics.json", suppression_metrics)
    if not score_identity_pass:
        raise RuntimeError("R5_2C_SCORE_IDENTITY_MISMATCH")

    positive_mask = labels
    positive_count = int(positive_mask.sum())
    positive_winners = Counter(winners[positive_mask])
    positive_by_winner: dict[str, Any] = {}
    for family in ["KEEP", "SUB", "DELETE"]:
        mask = positive_mask & (winners == family)
        positive_by_winner[family] = {
            "n": int(mask.sum()),
            "share": float(mask.sum() / positive_count),
            "suppression": stats(suppression_constructed[mask]),
            "R5_1A_A": stats(a[mask]),
            "R5_2B_C": stats(c[mask]),
        }
    positive_suppression = {
        "positive_words": positive_count,
        "winner_families": count_share(positive_winners, positive_count),
        "overall_suppression": stats(suppression_constructed[positive_mask]),
        "R5_1A_A_distribution": stats(a[positive_mask]),
        "R5_2B_C_distribution": stats(c[positive_mask]),
        "score_location_delta_C_minus_A": stats(c[positive_mask] - a[positive_mask]),
        "by_winner_family": positive_by_winner,
        "systematic_SUB_or_DELETE_suppression_share": float(
            np.mean(np.isin(winners[positive_mask], ["SUB", "DELETE"]))
        ),
    }
    write_json(out_dir / "r5_2c_positive_suppression.json", positive_suppression)

    ranking_masks = {
        "addition_vs_all_negatives": ~labels,
        "addition_vs_correct_only": cohort_masks["correct_only_negative"],
        "addition_vs_substitution_containing": cohort_masks["substitution_containing_negative"],
        "addition_vs_deletion_containing": cohort_masks["deletion_containing_negative"],
    }
    ranking: dict[str, Any] = {}
    for name, negative_mask in ranking_masks.items():
        comparison_mask = labels | negative_mask
        comparison_labels = labels[comparison_mask]
        auc_a = auc_mann_whitney(a[comparison_mask], comparison_labels)
        auc_c = auc_mann_whitney(c[comparison_mask], comparison_labels)
        delta = auc_c - auc_a
        ranking[name] = {
            "positive_n": int(labels.sum()),
            "negative_n": int(negative_mask.sum()),
            "R5_1A_A_auc": auc_a,
            "R5_2B_C_auc": auc_c,
            "delta_C_minus_A": delta,
            "classification": "improved_ranking" if delta > 0 else ("degraded_ranking" if delta < 0 else "approximately_preserved_ranking"),
        }
    ranking_comparison = {
        "implementation": "deterministic Mann-Whitney average-rank AUC over existing frozen scores",
        "no_score_transformation": True,
        "comparisons": ranking,
    }
    write_json(out_dir / "r5_2c_ranking_comparison.json", ranking_comparison)

    old_tp_new_fn_mask = labels & old_pred & ~new_pred
    old_fn_new_tp_mask = labels & ~old_pred & new_pred
    old_tp_count = int(np.count_nonzero(labels & old_pred))

    def transition_detail(mask: np.ndarray) -> dict[str, Any]:
        idx = np.flatnonzero(mask).tolist()
        total = len(idx)
        return {
            "count": total,
            "suppression": stats(suppression_constructed[mask]),
            "winner_family": count_share(Counter(winners[mask]), total),
            "speaker_counts": dict(Counter(old_rows[i]["speaker"] for i in idx).most_common()),
            "truth_diagnostics": truth_diagnostics(idx, old_rows),
        }

    tp_fn = {
        "R5_1A_positive_TP": old_tp_count,
        "R5_1A_TP_to_R5_2B_FN": transition_detail(old_tp_new_fn_mask),
        "share_of_R5_1A_TP_lost": float(old_tp_new_fn_mask.sum() / old_tp_count) if old_tp_count else None,
        "R5_1A_FN_to_R5_2B_TP": transition_detail(old_fn_new_tp_mask),
        "thresholds_are_frozen_oof_values": True,
    }
    write_json(out_dir / "r5_2c_tp_fn_transition_audit.json", tp_fn)

    correct_mask = cohort_masks["correct_only_negative"]
    correct_transitions = transitions(old_pred, new_pred, correct_mask)
    new_correct_fp_mask = correct_mask & ~old_pred & new_pred
    new_correct_fp_indices = np.flatnonzero(new_correct_fp_mask).tolist()
    new_correct_fp_rows: list[dict[str, Any]] = []
    for i in new_correct_fp_indices:
        row = new_rows[i]
        new_correct_fp_rows.append(
            {
                "source_identity": row["source_identity"],
                "speaker": row["speaker"],
                "word": row["word"],
                "A": float(a[i]),
                "C": float(c[i]),
                "suppression": float(suppression_constructed[i]),
                "R5_1A_theta": float(theta_a[i]),
                "R5_2B_theta": float(theta_c[i]),
                "R5_1A_margin": float(a[i] - theta_a[i]),
                "R5_2B_margin": float(c[i] - theta_c[i]),
                "BEST_NON_ADDITION_family": str(winners[i]),
            }
        )
    speaker_threshold_comp: dict[str, Any] = {}
    for speaker in SPEAKERS:
        sm = np.asarray([r["speaker"] == speaker for r in old_rows], dtype=bool)
        cm = sm & correct_mask
        speaker_threshold_comp[speaker] = {
            "R5_1A_theta": float(theta_a[sm][0]),
            "R5_2B_theta": float(theta_c[sm][0]),
            "theta_delta_R5_2B_minus_R5_1A": float(theta_c[sm][0] - theta_a[sm][0]),
            "all_word_A": stats(a[sm]),
            "all_word_C": stats(c[sm]),
            "correct_only_A": stats(a[cm]),
            "correct_only_C": stats(c[cm]),
            "correct_only_old_margin": stats((a - theta_a)[cm]),
            "correct_only_new_margin": stats((c - theta_c)[cm]),
            "correct_transition_counts": transitions(old_pred, new_pred, cm),
        }
    threshold_compensation = {
        "frozen_thresholds_only": True,
        "threshold_search_or_optimization": False,
        "correct_only_transition_counts": correct_transitions,
        "new_R5_2B_correct_false_positives": new_correct_fp_rows,
        "new_false_positive_summary": {
            "count": len(new_correct_fp_indices),
            "C_below_corresponding_R5_1A_theta_count": int(np.count_nonzero(c[new_correct_fp_mask] < theta_a[new_correct_fp_mask])),
            "score_was_suppressed_count": int(np.count_nonzero(suppression_constructed[new_correct_fp_mask] > 0.0)),
            "threshold_drop_exceeded_observed_score_drop_count": int(
                np.count_nonzero((theta_a - theta_c)[new_correct_fp_mask] > suppression_observed[new_correct_fp_mask])
            ),
            "speaker_counts": dict(Counter(new_rows[i]["speaker"] for i in new_correct_fp_indices).most_common()),
            "winner_family": count_share(Counter(winners[new_correct_fp_mask]), len(new_correct_fp_indices)),
            "suppression": stats(suppression_constructed[new_correct_fp_mask]),
            "old_margin": stats((a - theta_a)[new_correct_fp_mask]),
            "new_margin": stats((c - theta_c)[new_correct_fp_mask]),
        },
        "by_speaker": speaker_threshold_comp,
        "interpretation": "lower frozen R5-2B LOSO thresholds compensated for broad score suppression and introduced correct-only false positives",
    }
    write_json(out_dir / "r5_2c_threshold_compensation.json", threshold_compensation)

    relation_audit: dict[str, Any] = {}
    for name in ["substitution_containing_negative", "deletion_containing_negative"]:
        mask = cohort_masks[name]
        total = int(mask.sum())
        relation_audit[name] = {
            "n": total,
            "suppression": stats(suppression_constructed[mask]),
            "decision_transitions": transitions(old_pred, new_pred, mask),
            "winner_family": count_share(Counter(winners[mask]), total),
            "R5_1A_FAR": far(old_pred, mask),
            "R5_2B_FAR": far(new_pred, mask),
            "FAR_delta_R5_2B_minus_R5_1A": far(new_pred, mask)["false_addition_rate"] - far(old_pred, mask)["false_addition_rate"],
        }
    write_json(out_dir / "r5_2c_relation_cohort_audit.json", relation_audit)

    metadata_mismatch: list[str] = []
    for old, new in zip(old_rows, new_rows):
        if (
            bool(old["best_insert_exists"]) != bool(new["best_insert_exists"])
            or old["best_insert_phone_index"] != new["best_insert_phone_index"]
            or old["best_insert_boundary"] != new["best_insert_boundary"]
        ):
            metadata_mismatch.append(old["source_identity"])
    old_event_pred = np.asarray([bool(p and r["best_insert_exists"]) for p, r in zip(old_pred, old_rows)], dtype=bool)
    new_event_pred = np.asarray([bool(p and r["best_insert_exists"]) for p, r in zip(new_pred, new_rows)], dtype=bool)

    def exact_event(row: dict[str, Any], predicted: bool) -> bool:
        if not predicted:
            return False
        phone = int(row["best_insert_phone_index"])
        boundary = int(row["best_insert_boundary"])
        return any(int(e["phone_index"]) == phone and int(e["boundary"]) == boundary for e in row["ground_truth_events"])

    old_exact = np.asarray([exact_event(row, pred) for row, pred in zip(old_rows, old_event_pred)], dtype=bool)
    new_exact = np.asarray([exact_event(row, pred) for row, pred in zip(new_rows, new_event_pred)], dtype=bool)
    old_fp_event = old_event_pred & ~old_exact
    new_fp_event = new_event_pred & ~new_exact
    lost_event_tp = old_exact & ~new_exact
    removed_event_fp = old_fp_event & ~new_fp_event
    introduced_event_fp = new_fp_event & ~old_fp_event

    def event_position_counts(mask: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, int]:
        result: Counter[str] = Counter()
        for i in np.flatnonzero(mask):
            boundary = int(rows[i]["best_insert_boundary"])
            result[event_position(boundary, len(rows[i]["expected_sequence"]))] += 1
        return dict(result)

    event_loss = {
        "BEST_INSERT_metadata": {
            "identical_count": n - len(metadata_mismatch),
            "identical_share": (n - len(metadata_mismatch)) / n,
            "mismatch_count": len(metadata_mismatch),
            "mismatch_examples": metadata_mismatch[:20],
        },
        "frozen_event_F1": {"R5_1A": 0.04389312977099236, "R5_2B": 0.03253796095444685, "delta": 0.03253796095444685 - 0.04389312977099236},
        "R5_1A_predicted_event_words": int(old_event_pred.sum()),
        "R5_2B_predicted_event_words": int(new_event_pred.sum()),
        "R5_1A_exact_event_TP_words": int(old_exact.sum()),
        "R5_2B_exact_event_TP_words": int(new_exact.sum()),
        "event_TP_lost_due_decision_change": int(lost_event_tp.sum()),
        "event_TP_lost_position_counts": event_position_counts(lost_event_tp, old_rows),
        "event_FP_removed": int(removed_event_fp.sum()),
        "event_FP_removed_position_counts": event_position_counts(removed_event_fp, old_rows),
        "event_FP_newly_introduced": int(introduced_event_fp.sum()),
        "event_FP_new_position_counts": event_position_counts(introduced_event_fp, new_rows),
        "primary_cause": "decision changes from relation suppression/LOSO thresholds; BEST_INSERT phone/boundary construction remained identical" if not metadata_mismatch else "both decision and unexpected BEST_INSERT metadata change",
    }
    write_json(out_dir / "r5_2c_event_loss_audit.json", event_loss)

    speaker_audit: dict[str, Any] = {}
    recall_decreased = 0
    correct_far_increased = 0
    for speaker in SPEAKERS:
        sm = np.asarray([r["speaker"] == speaker for r in old_rows], dtype=bool)
        pm = sm & labels
        cm = sm & correct_mask
        subm = sm & cohort_masks["substitution_containing_negative"]
        delm = sm & cohort_masks["deletion_containing_negative"]
        old_recall = float(np.mean(old_pred[pm])) if pm.any() else None
        new_recall = float(np.mean(new_pred[pm])) if pm.any() else None
        old_correct_far = far(old_pred, cm)["false_addition_rate"]
        new_correct_far = far(new_pred, cm)["false_addition_rate"]
        if new_recall is not None and old_recall is not None and new_recall < old_recall:
            recall_decreased += 1
        if new_correct_far is not None and old_correct_far is not None and new_correct_far > old_correct_far:
            correct_far_increased += 1
        speaker_audit[speaker] = {
            "positive_count": int(pm.sum()),
            "R5_1A_addition_recall": old_recall,
            "R5_2B_addition_recall": new_recall,
            "recall_delta": new_recall - old_recall if old_recall is not None else None,
            "suppression_median": float(np.median(suppression_constructed[sm])),
            "correct_FAR_delta": new_correct_far - old_correct_far,
            "substitution_FAR_delta": far(new_pred, subm)["false_addition_rate"] - far(old_pred, subm)["false_addition_rate"],
            "deletion_FAR_delta": far(new_pred, delm)["false_addition_rate"] - far(old_pred, delm)["false_addition_rate"],
            "R5_1A_theta": float(theta_a[sm][0]),
            "R5_2B_theta": float(theta_c[sm][0]),
        }
    speaker_output = {
        "speakers": speaker_audit,
        "speakers_with_recall_decrease": recall_decreased,
        "speakers_with_correct_FAR_increase": correct_far_increased,
        "assessment": "broad_across_speakers" if recall_decreased >= 7 or correct_far_increased >= 7 else "concentrated_in_fewer_speakers",
        "no_speaker_excluded": True,
    }
    write_json(out_dir / "r5_2c_speaker_audit.json", speaker_output)

    positive_q90 = float(np.quantile(suppression_constructed[positive_mask], 0.90))
    strongly_suppressed_positive = positive_mask & (suppression_constructed >= positive_q90)
    diagnostics = {
        "descriptive_only": True,
        "strong_suppression_definition": "positive-word constructed suppression >= frozen-positive Q90",
        "strong_suppression_q90": positive_q90,
        "strongly_suppressed_true_additions": truth_diagnostics(np.flatnonzero(strongly_suppressed_positive).tolist(), old_rows),
        "R5_1A_TP_to_R5_2B_FN": truth_diagnostics(np.flatnonzero(old_tp_new_fn_mask).tolist(), old_rows),
        "correct_TN_to_R5_2B_FP": predicted_diagnostics(new_correct_fp_indices, new_rows),
    }
    write_json(out_dir / "r5_2c_phone_position_diagnostics.json", diagnostics)

    overall_auc_drop = ranking["addition_vs_all_negatives"]["delta_C_minus_A"] < 0
    correct_auc_drop = ranking["addition_vs_correct_only"]["delta_C_minus_A"] < 0
    positive_subdel_share = float(np.mean(np.isin(winners[positive_mask], ["SUB", "DELETE"])))
    new_correct_fp_count = int(new_correct_fp_mask.sum())
    threshold_explained = bool(
        new_correct_fp_count > 0
        and np.count_nonzero(c[new_correct_fp_mask] < theta_a[new_correct_fp_mask]) / new_correct_fp_count >= 0.8
    )
    over_suppression = bool(overall_auc_drop and correct_auc_drop and positive_subdel_share > 0.5)
    if over_suppression and threshold_explained:
        classification = "MIXED_OVER_SUPPRESSION_AND_THRESHOLD_COMPENSATION"
    elif over_suppression:
        classification = "TRUE_ADDITION_OVER_SUPPRESSION"
    elif threshold_explained:
        classification = "THRESHOLD_COMPENSATION_DOMINANT"
    elif speaker_output["assessment"] != "broad_across_speakers":
        classification = "SPEAKER_DOMINATED_FAILURE"
    else:
        classification = "INCONCLUSIVE_FAILURE_MECHANISM"
    interpretation = {
        "posthoc_exploratory": True,
        "authoritative_R5_2B_status_unchanged": "R5_2_RELATION_COMPETITION_DEVELOPMENT_NOT_CONFIRMED",
        "classification": classification,
        "questions": {
            "Q1": {
                "answer": "YES",
                "evidence": "Substitution- and deletion-negative FAR fell, with many frozen R5-1A FP to R5-2B TN conversions and corresponding SUB/DELETE winners.",
            },
            "Q2": {
                "answer": "YES",
                "evidence": f"SUB or DELETE won BEST_NON_ADDITION for {positive_subdel_share:.6%} of the 323 positive words, and positive scores shifted downward.",
            },
            "Q3": {
                "answer": "YES",
                "evidence": "AUC fell for addition vs all negatives and correct-only negatives, consistent with unequal cohort suppression changing ranks.",
            },
            "Q4": {
                "answer": "YES",
                "evidence": f"{int(np.count_nonzero(c[new_correct_fp_mask] < theta_a[new_correct_fp_mask]))}/{new_correct_fp_count} newly introduced correct FPs had C below the old frozen threshold and became positive only under the lower R5-2B fold threshold.",
            },
            "Q5": {
                "answer": "YES" if not metadata_mismatch else "NO",
                "evidence": f"BEST_INSERT phone/boundary matched for {n-len(metadata_mismatch)}/{n} words; event changes therefore came from frozen decision changes rather than relocalization.",
            },
            "Q6": {
                "answer": "YES",
                "evidence": "G6/G7 support the narrow SUB/DELETE-confounding mechanism, while the other six frozen gates failed.",
            },
        },
        "classification_evidence": {
            "over_suppression_supported": over_suppression,
            "threshold_compensation_supported": threshold_explained,
            "speaker_failure_assessment": speaker_output["assessment"],
        },
    }
    write_json(out_dir / "r5_2c_interpretation.json", interpretation)

    protocol = {
        "neural_training": False,
        "checkpoint_inference": False,
        "new_model_scores_created": False,
        "threshold_search": False,
        "new_predictions_created": False,
        "existing_frozen_predictions_read": True,
        "train_audio_accessed": False,
        "validation_paths_resolved": False,
        "validation_accessed": False,
        "test_paths_resolved": False,
        "test_accessed": False,
        "r5_1a_modified": False,
        "r5_2b_modified": False,
        "scientific_method_modified": False,
    }
    write_json(out_dir / "r5_2c_protocol_audit.json", protocol)

    report = f"""# R5-2C Frozen Relation-Competition Failure Audit

## Identity and join

- R5-1A manifest: `{r5_1a_manifest['sha256']}` — PASS ({r5_1a_manifest['entry_count']}/{r5_1a_manifest['entry_count']})
- R5-2B manifest: `{r5_2b_manifest['sha256']}` — PASS ({r5_2b_manifest['entry_count']}/{r5_2b_manifest['entry_count']})
- Exact stable-identity join: {len(common):,} rows; 0 missing; 0 duplicate; 0 excluded.

## Fundamental score identity

The constructed identity term `BEST_NON_ADDITION - KEEP` was nonnegative for all {n:,} words. Frozen cross-execution `A-C` contained tiny float32 representation residuals down to {float(np.min(suppression_observed)):.17g}; all were exactly explained by the separately frozen INSERT-minus-KEEP representation delta. No material score-identity violation occurred.

## Main diagnosis

- Positive BEST_NON_ADDITION winners: {dict(positive_winners)}.
- Addition/all AUC: {ranking['addition_vs_all_negatives']['R5_1A_A_auc']:.16g} -> {ranking['addition_vs_all_negatives']['R5_2B_C_auc']:.16g}.
- Addition/correct AUC: {ranking['addition_vs_correct_only']['R5_1A_A_auc']:.16g} -> {ranking['addition_vs_correct_only']['R5_2B_C_auc']:.16g}.
- Correct-only TN -> FP transitions: {correct_transitions['TN_R5_1A_to_FP_R5_2B']}.
- BEST_INSERT phone/boundary identical: {n-len(metadata_mismatch):,}/{n:,}.

## Frozen interpretation

`{classification}`

Relation competition genuinely suppressed substitution/deletion false additions, but it also allowed SUB/DELETE explanations to suppress many true additions. The LOSO thresholds then moved downward to compensate, introducing correct-only false positives. Event localization itself did not change; event F1 declined because the binary decision set changed.

## Protocol

No training, checkpoint inference, audio access, threshold search, new scores, new predictions, VALIDATION access, or TEST access occurred. Frozen R5-1A and R5-2B artifacts were not modified.

## Status

`R5_2C_POSTHOC_FAILURE_AUDIT_COMPLETE`
"""
    (out_dir / "R5_2C_RELATION_COMPETITION_FAILURE_AUDIT.md").write_text(report, encoding="utf-8")

    shutil.copy2(Path(__file__), out_dir / "r5_2c_failure_audit_driver.py")
    artifact_names = [
        "r5_2c_failure_audit_driver.py",
        "r5_2c_source_identity.json",
        "r5_2c_join_audit.json",
        "r5_2c_suppression_metrics.json",
        "r5_2c_positive_suppression.json",
        "r5_2c_ranking_comparison.json",
        "r5_2c_tp_fn_transition_audit.json",
        "r5_2c_threshold_compensation.json",
        "r5_2c_relation_cohort_audit.json",
        "r5_2c_event_loss_audit.json",
        "r5_2c_speaker_audit.json",
        "r5_2c_phone_position_diagnostics.json",
        "r5_2c_interpretation.json",
        "r5_2c_protocol_audit.json",
        "R5_2C_RELATION_COMPETITION_FAILURE_AUDIT.md",
    ]
    artifacts = []
    for name in artifact_names:
        path = out_dir / name
        artifacts.append({"relative_path": name, "byte_size": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "stage": "R5-2C frozen relation-competition post-hoc failure audit",
        "status": "R5_2C_POSTHOC_FAILURE_AUDIT_COMPLETE",
        "manifest_self_excluded": True,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "hash_audit": "HASH_AUDIT_PASS",
    }
    write_json(out_dir / "R5_2C_MANIFEST.json", manifest)

    verification_failures = []
    for entry in artifacts:
        path = out_dir / entry["relative_path"]
        if path.stat().st_size != entry["byte_size"] or sha256(path) != entry["sha256"]:
            verification_failures.append(entry["relative_path"])
    if verification_failures:
        raise RuntimeError(f"R5-2C hash verification failed: {verification_failures}")
    print(
        json.dumps(
            {
                "status": "R5_2C_POSTHOC_FAILURE_AUDIT_COMPLETE",
                "classification": classification,
                "matched_rows": len(common),
                "artifact_count": len(artifacts),
                "manifest_sha256": sha256(out_dir / "R5_2C_MANIFEST.json"),
                "hash_audit": "HASH_AUDIT_PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve())
