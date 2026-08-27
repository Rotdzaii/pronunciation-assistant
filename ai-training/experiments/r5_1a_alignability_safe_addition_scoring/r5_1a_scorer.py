from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch


BLANK_INDEX = 40
PHONE_COUNT = 40


class R5_1AScorerError(RuntimeError):
    """Base class for frozen R5-1A scorer failures."""


class KeepAlignabilityError(R5_1AScorerError):
    status = "R5_1A_EXECUTION_BLOCKED_KEEP_ALIGNABILITY"

    def __init__(self, target: Sequence[int], input_length: int) -> None:
        super().__init__(
            f"{self.status}: KEEP minimum={minimum_ctc_steps(target)} input_length={input_length}"
        )


class NonFiniteAlignableLossError(R5_1AScorerError):
    pass


class NonFiniteMetricSupportError(R5_1AScorerError):
    status = "R5_1A_EXECUTION_BLOCKED_NONFINITE_METRIC_SUPPORT"


class NoFiniteThresholdScoresError(R5_1AScorerError):
    pass


@dataclass(frozen=True)
class HypothesisScore:
    target: tuple[int, ...]
    input_length: int
    minimum_steps: int
    alignable: bool
    raw_score: float
    target_score: float
    ctc_called: bool


@dataclass(frozen=True)
class InsertCandidateScore:
    boundary: int
    phone_index: int
    hypothesis: HypothesisScore


@dataclass(frozen=True)
class BestInsertResult:
    best_insert_exists: bool
    boundary: int | None
    phone_index: int | None
    best_target_score: float
    total_candidates: int
    alignable_candidates: int
    impossible_candidates: int


@dataclass(frozen=True)
class WordScore:
    keep: HypothesisScore
    best_insert: BestInsertResult
    addition_score: float


def adjacent_repeat_count(target: Sequence[int]) -> int:
    values = tuple(int(value) for value in target)
    return sum(current == previous for previous, current in zip(values, values[1:]))


def minimum_ctc_steps(target: Sequence[int]) -> int:
    values = tuple(int(value) for value in target)
    return len(values) + adjacent_repeat_count(values)


def is_alignable(target: Sequence[int], input_length: int) -> bool:
    return int(input_length) >= minimum_ctc_steps(target)


def _validate_target(target: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in target)
    if not values:
        raise ValueError("R5-1A KEEP/INSERT targets are nonempty")
    if any(value < 0 or value >= PHONE_COUNT for value in values):
        raise ValueError("Target phone outside frozen indices 0..39")
    return values


def score_hypothesis(logits: torch.Tensor, target: Sequence[int], input_length: int | None = None) -> HypothesisScore:
    values = _validate_target(target)
    if logits.ndim != 2 or logits.shape[1] != PHONE_COUNT + 1:
        raise ValueError("logits must have shape [T,41]")
    available_steps = int(logits.shape[0])
    steps = available_steps if input_length is None else int(input_length)
    if steps < 0 or steps > available_steps:
        raise ValueError("input_length must be within logits time dimension")
    minimum = minimum_ctc_steps(values)
    if steps < minimum:
        return HypothesisScore(
            target=values,
            input_length=steps,
            minimum_steps=minimum,
            alignable=False,
            raw_score=float("-inf"),
            target_score=float("-inf"),
            ctc_called=False,
        )
    log_probs = torch.log_softmax(logits[:steps], dim=-1).unsqueeze(1)
    loss = torch.nn.CTCLoss(blank=BLANK_INDEX, reduction="none", zero_infinity=True)(
        log_probs,
        torch.tensor(values, dtype=torch.long, device=logits.device),
        torch.tensor([steps], dtype=torch.long, device=logits.device),
        torch.tensor([len(values)], dtype=torch.long, device=logits.device),
    )
    scalar_loss = float(loss.detach().cpu().item())
    if not math.isfinite(scalar_loss):
        raise NonFiniteAlignableLossError(
            f"Alignable target returned non-finite CTCLoss: minimum={minimum} input_length={steps}"
        )
    raw_score = -scalar_loss
    target_score = raw_score / max(len(values), 1)
    if not math.isfinite(raw_score) or not math.isfinite(target_score):
        raise NonFiniteAlignableLossError("Alignable target produced non-finite derived score")
    return HypothesisScore(
        target=values,
        input_length=steps,
        minimum_steps=minimum,
        alignable=True,
        raw_score=raw_score,
        target_score=target_score,
        ctc_called=True,
    )


def score_keep(logits: torch.Tensor, expected: Sequence[int], input_length: int | None = None) -> HypothesisScore:
    steps = int(logits.shape[0]) if input_length is None else int(input_length)
    result = score_hypothesis(logits, expected, steps)
    if not result.alignable:
        raise KeepAlignabilityError(expected, steps)
    return result


def enumerate_insert_targets(expected: Sequence[int]) -> Iterator[tuple[int, int, tuple[int, ...]]]:
    values = _validate_target(expected)
    for boundary in range(len(values) + 1):
        for phone_index in range(PHONE_COUNT):
            yield boundary, phone_index, values[:boundary] + (phone_index,) + values[boundary:]


def select_best_insert(candidates: Iterable[InsertCandidateScore]) -> BestInsertResult:
    materialized = list(candidates)
    alignable = [candidate for candidate in materialized if candidate.hypothesis.alignable]
    for candidate in alignable:
        if not math.isfinite(candidate.hypothesis.target_score):
            raise NonFiniteAlignableLossError("Alignable INSERT candidate has non-finite TARGET score")
    if not alignable:
        return BestInsertResult(
            best_insert_exists=False,
            boundary=None,
            phone_index=None,
            best_target_score=float("-inf"),
            total_candidates=len(materialized),
            alignable_candidates=0,
            impossible_candidates=len(materialized),
        )
    winner = min(
        alignable,
        key=lambda candidate: (
            -candidate.hypothesis.target_score,
            candidate.boundary,
            candidate.phone_index,
        ),
    )
    return BestInsertResult(
        best_insert_exists=True,
        boundary=winner.boundary,
        phone_index=winner.phone_index,
        best_target_score=winner.hypothesis.target_score,
        total_candidates=len(materialized),
        alignable_candidates=len(alignable),
        impossible_candidates=len(materialized) - len(alignable),
    )


def score_word(logits: torch.Tensor, expected: Sequence[int], input_length: int | None = None) -> WordScore:
    steps = int(logits.shape[0]) if input_length is None else int(input_length)
    keep = score_keep(logits, expected, steps)
    insert_scores = [
        InsertCandidateScore(
            boundary=boundary,
            phone_index=phone_index,
            hypothesis=score_hypothesis(logits, target, steps),
        )
        for boundary, phone_index, target in enumerate_insert_targets(expected)
    ]
    best = select_best_insert(insert_scores)
    addition_score = (
        best.best_target_score - keep.target_score if best.best_insert_exists else float("-inf")
    )
    return WordScore(keep=keep, best_insert=best, addition_score=addition_score)


def extended_real_roc_auc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.size != s.size or y.size == 0:
        raise ValueError("labels and scores must be equal-length nonempty vectors")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be binary")
    if np.isnan(s).any() or np.isposinf(s).any():
        raise NonFiniteMetricSupportError("Only finite scores and negative infinity are supported")
    positive_count = int(y.sum())
    negative_count = int(y.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both classes")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and s[order[end]] == s[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be a vector")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("threshold scores may contain only finite values or negative infinity")
    finite = np.unique(values[np.isfinite(values)])
    if finite.size == 0:
        raise NoFiniteThresholdScoresError("No finite calibration scores")
    return np.concatenate((
        np.asarray([np.nextafter(finite[0], -np.inf)], dtype=np.float64),
        finite,
        np.asarray([np.nextafter(finite[-1], np.inf)], dtype=np.float64),
    ))


def predict_addition(scores: Sequence[float], threshold: float) -> np.ndarray:
    theta = float(threshold)
    if not math.isfinite(theta):
        raise ValueError("R5-1A operational threshold must be finite")
    values = np.asarray(scores, dtype=np.float64)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("scores may contain only finite values or negative infinity")
    return values >= theta


def serialize_extended_score(value: float) -> dict[str, float | bool | None]:
    scalar = float(value)
    if math.isnan(scalar) or math.isinf(scalar) and scalar > 0:
        raise ValueError("Only finite values and negative infinity are serializable")
    if scalar == float("-inf"):
        return {"score_value": None, "score_is_neg_inf": True}
    return {"score_value": scalar, "score_is_neg_inf": False}


def deserialize_extended_score(payload: dict[str, object]) -> float:
    is_negative_infinity = payload.get("score_is_neg_inf")
    value = payload.get("score_value")
    if is_negative_infinity is True:
        if value is not None:
            raise ValueError("Negative-infinity representation requires score_value=null")
        return float("-inf")
    if is_negative_infinity is False and isinstance(value, (int, float)):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError("Finite representation must contain a finite numeric value")
        return scalar
    raise ValueError("Malformed extended-real score representation")
