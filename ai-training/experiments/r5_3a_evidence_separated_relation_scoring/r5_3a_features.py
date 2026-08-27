"""Frozen R5-3A evidence-separated feature construction."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FEATURE_ORDER = ("A", "S", "D")
FORBIDDEN_FEATURES = frozenset(
    {
        "C",
        "max(S,D)",
        "speaker_id",
        "phone_id",
        "word_id",
        "utterance_id",
        "boundary",
        "duration",
        "error_relation_truth",
        "manual_annotation_count",
        "validation_information",
        "test_information",
    }
)


def _finite_float64(value: Any, name: str) -> np.float64:
    result = np.float64(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def construct_feature_row(
    r5_1a_row: Mapping[str, Any], r5_2b_row: Mapping[str, Any]
) -> np.ndarray:
    """Return exactly [A,S,D] from one exact stable-identity join."""
    if r5_1a_row["source_identity"] != r5_2b_row["source_identity"]:
        raise ValueError("source_identity mismatch")
    a = _finite_float64(r5_1a_row["addition_score_A_value"], "A")
    keep = _finite_float64(r5_2b_row["keep_target_score_value"], "KEEP")
    best_sub = _finite_float64(r5_2b_row["best_sub_score_value"], "BEST_SUB")
    best_delete = _finite_float64(r5_2b_row["best_delete_score_value"], "BEST_DELETE")
    s = np.float64(best_sub - keep)
    d = np.float64(best_delete - keep)
    result = np.asarray([a, s, d], dtype=np.float64)
    validate_design_matrix(result.reshape(1, -1))
    return result


def construct_feature_matrix(
    r5_1a_rows: Sequence[Mapping[str, Any]],
    r5_2b_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    """Exact join without fuzzy matching, exclusions, or duplicate identities."""
    old_index: dict[str, Mapping[str, Any]] = {}
    new_index: dict[str, Mapping[str, Any]] = {}
    for row in r5_1a_rows:
        key = str(row["source_identity"])
        if key in old_index:
            raise ValueError(f"duplicate R5-1A identity: {key}")
        old_index[key] = row
    for row in r5_2b_rows:
        key = str(row["source_identity"])
        if key in new_index:
            raise ValueError(f"duplicate R5-2B identity: {key}")
        new_index[key] = row
    if old_index.keys() != new_index.keys():
        raise ValueError("exact row identity sets differ")
    identities = [str(row["source_identity"]) for row in r5_2b_rows]
    matrix = np.vstack([construct_feature_row(old_index[key], new_index[key]) for key in identities])
    validate_design_matrix(matrix)
    return matrix, identities


def validate_feature_names(names: Iterable[str]) -> None:
    names_tuple = tuple(names)
    if names_tuple != FEATURE_ORDER:
        raise ValueError(f"feature order must be exactly {FEATURE_ORDER}")
    if FORBIDDEN_FEATURES.intersection(names_tuple):
        raise ValueError("forbidden feature present")


def validate_design_matrix(matrix: np.ndarray) -> None:
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("classifier design matrix must have exactly three columns")
    if array.dtype != np.float64:
        raise ValueError("classifier design matrix must be float64")
    if not np.all(np.isfinite(array)):
        raise ValueError("classifier design matrix must contain only finite values")
    validate_feature_names(FEATURE_ORDER)
