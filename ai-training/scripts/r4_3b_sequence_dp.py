"""Frozen deterministic R4-3B word-level sequence DP.

This module implements the R4-3B0 numerical contract. It does not load data,
audio, checkpoints, or validation artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from typing import Sequence

import numpy as np


PHONE_VOCAB = (
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "B", "CH", "D",
    "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH",
    "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S",
    "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
)
PHONE_TO_ID = {phone: index for index, phone in enumerate(PHONE_VOCAB)}
LOG_PRIORS = {
    "MATCH": -0.14116417177608467,
    "SUBSTITUTION": -2.261305001061487,
    "DELETE_EXPECTED": -3.595794950992514,
}
OPERATION_RANK = {"MATCH": 0, "SUBSTITUTION": 1, "DELETE_EXPECTED": 2}
TIE_EPSILON = 1e-8


@dataclass(frozen=True)
class Step:
    operation: str
    expected_index: int
    expected_phone_index: int
    probe_start: int
    probe_end: int
    observed_phone_index: int | None
    acoustic_score: float
    prior_score: float

    @property
    def span_length(self) -> int:
        return self.probe_end - self.probe_start


@dataclass(frozen=True)
class Path:
    score: float
    steps: tuple[Step, ...]

    @property
    def identity(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            (
                OPERATION_RANK[step.operation],
                step.probe_start,
                step.probe_end,
                step.observed_phone_index if step.observed_phone_index is not None else -1,
            )
            for step in self.steps
        )


def _last_tie_key(path: Path) -> tuple[object, ...]:
    step = path.steps[-1]
    span_key = step.span_length if step.operation in {"MATCH", "SUBSTITUTION"} else 0
    q_key = step.observed_phone_index if step.operation == "SUBSTITUTION" else -1
    return (
        OPERATION_RANK[step.operation],
        span_key,
        q_key,
        step.probe_start,
        path.identity,
    )


def _compare_paths(left: Path, right: Path) -> int:
    if left.score > right.score + TIE_EPSILON:
        return -1
    if right.score > left.score + TIE_EPSILON:
        return 1
    left_key, right_key = _last_tie_key(left), _last_tie_key(right)
    if left_key < right_key:
        return -1
    if left_key > right_key:
        return 1
    if left.identity < right.identity:
        return -1
    if left.identity > right.identity:
        return 1
    return 0


def _retain_top_two(paths: list[Path], candidate: Path) -> None:
    by_identity = {path.identity: path for path in paths}
    prior = by_identity.get(candidate.identity)
    if prior is None or _compare_paths(candidate, prior) < 0:
        by_identity[candidate.identity] = candidate
    ordered = sorted(by_identity.values(), key=cmp_to_key(_compare_paths))
    paths[:] = ordered[:2]


def align(expected_phones: Sequence[str], log_probabilities: np.ndarray) -> dict[str, object]:
    """Align one expected word to a frozen acoustic log-probability timeline."""
    expected = tuple(PHONE_TO_ID[phone] for phone in expected_phones)
    evidence = np.asarray(log_probabilities, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[1] != len(PHONE_VOCAB):
        raise ValueError(f"Expected [T,40] acoustic evidence, got {evidence.shape}")
    if not np.isfinite(evidence).all():
        raise ValueError("Acoustic evidence contains non-finite values")
    n, probe_count = len(expected), evidence.shape[0]
    states: list[list[list[Path]]] = [[[] for _ in range(probe_count + 1)] for _ in range(n + 1)]
    states[0][0] = [Path(0.0, ())]
    prefix = np.vstack([np.zeros((1, evidence.shape[1]), dtype=np.float64), np.cumsum(evidence, axis=0)])

    for i in range(n):
        expected_phone = expected[i]
        for start in range(probe_count + 1):
            for path in tuple(states[i][start]):
                delete_step = Step(
                    "DELETE_EXPECTED", i, expected_phone, start, start, None,
                    0.0, LOG_PRIORS["DELETE_EXPECTED"],
                )
                _retain_top_two(
                    states[i + 1][start],
                    Path(path.score + LOG_PRIORS["DELETE_EXPECTED"], path.steps + (delete_step,)),
                )
                for end in range(start + 1, probe_count + 1):
                    mean = (prefix[end] - prefix[start]) / float(end - start)
                    match_acoustic = float(mean[expected_phone])
                    match_step = Step(
                        "MATCH", i, expected_phone, start, end, expected_phone,
                        match_acoustic, LOG_PRIORS["MATCH"],
                    )
                    _retain_top_two(
                        states[i + 1][end],
                        Path(path.score + LOG_PRIORS["MATCH"] + match_acoustic, path.steps + (match_step,)),
                    )
                    alternative = mean.copy()
                    alternative[expected_phone] = -np.inf
                    q_star = int(np.argmax(alternative))
                    sub_acoustic = float(mean[q_star])
                    sub_step = Step(
                        "SUBSTITUTION", i, expected_phone, start, end, q_star,
                        sub_acoustic, LOG_PRIORS["SUBSTITUTION"],
                    )
                    _retain_top_two(
                        states[i + 1][end],
                        Path(path.score + LOG_PRIORS["SUBSTITUTION"] + sub_acoustic, path.steps + (sub_step,)),
                    )

    complete = states[n][probe_count]
    if not complete:
        return {"status": "NO_VALID_PATH", "best": None, "second_best": None, "path_score_gap": None,
                "numerically_ambiguous": False}
    best = complete[0]
    second = complete[1] if len(complete) > 1 else None
    gap = best.score - second.score if second is not None else None
    return {
        "status": "OK",
        "best": best,
        "second_best": second,
        "path_score_gap": gap,
        "numerically_ambiguous": second is not None and gap <= TIE_EPSILON,
    }


def probe_centers(word_start: float, word_end: float, stride_seconds: float = 0.04) -> list[float]:
    """Exact probe grid used by R4-3A."""
    if word_end <= word_start:
        return [(word_start + word_end) / 2.0]
    centers = [word_start]
    while centers[-1] + stride_seconds < word_end - 1e-9:
        centers.append(centers[-1] + stride_seconds)
    if word_end - centers[-1] > 1e-9:
        centers.append(word_end)
    return centers
