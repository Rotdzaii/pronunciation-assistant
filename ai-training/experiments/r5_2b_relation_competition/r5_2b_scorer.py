"""Frozen R5-2B relation-competitive exact CTC scorer.

This research-only module implements the contract identified by SHA-256
55572805C1878D41D4B6C41E1C7A31B2C3725A81050942494F0E1292092FEF38.
It is additive: it does not modify R4 or R5-1A source evidence.

Alignability, nonempty scoring, extended-real serialization, and threshold
helpers preserve the semantics of the frozen R5-1A scorer whose SHA-256 is
4DE49C9070C973EE44EFBD09DFC063C436779E723D12EC7A7A2BC4A06AF35F90.
The explicit empty-target all-blank path is new and frozen by R5-2B.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


PHONE_COUNT = 40
BLANK_INDEX = 40


class R5_2BScorerError(RuntimeError):
    """Base class for frozen R5-2B scorer failures."""


class KeepAlignabilityError(R5_2BScorerError):
    """Raised when KEEP is impossible for a population word."""


class NonFiniteAlignableScoreError(R5_2BScorerError):
    """Raised when an alignable target unexpectedly has a non-finite score."""


@dataclass(frozen=True)
class HypothesisScore:
    target: tuple[int, ...]
    input_length: int
    minimum_steps: int
    alignable: bool
    raw_score: float
    target_score: float
    ctc_called: bool
    empty_formula_used: bool


@dataclass(frozen=True)
class TargetCandidate:
    family: str
    target: tuple[int, ...]
    boundary: int | None = None
    position: int | None = None
    phone_index: int | None = None


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: TargetCandidate
    hypothesis: HypothesisScore


@dataclass(frozen=True)
class BestCandidateResult:
    exists: bool
    family: str
    candidate: TargetCandidate | None
    score: float
    total_candidates: int
    alignable_candidates: int
    impossible_candidates: int


@dataclass(frozen=True)
class BestNonAdditionResult:
    family: str
    score: float
    candidate: TargetCandidate


@dataclass(frozen=True)
class WordScore:
    keep: ScoredCandidate
    best_insert: BestCandidateResult
    best_sub: BestCandidateResult
    best_delete: BestCandidateResult
    best_nonaddition: BestNonAdditionResult
    relation_competition_score: float


@dataclass(frozen=True)
class AdditionEvent:
    word_id: str
    phone_index: int
    boundary: int


def _validate_phone_values(values: Sequence[int], *, allow_empty: bool) -> tuple[int, ...]:
    target = tuple(int(value) for value in values)
    if not allow_empty and not target:
        raise ValueError("Expected sequence must be nonempty")
    if any(value < 0 or value >= PHONE_COUNT for value in target):
        raise ValueError("Canonical phone index must be in 0..39")
    return target


def validate_expected(expected: Sequence[int]) -> tuple[int, ...]:
    return _validate_phone_values(expected, allow_empty=False)


def adjacent_repeat_count(target: Sequence[int]) -> int:
    values = _validate_phone_values(target, allow_empty=True)
    return sum(left == right for left, right in zip(values, values[1:]))


def minimum_ctc_steps(target: Sequence[int]) -> int:
    values = _validate_phone_values(target, allow_empty=True)
    return len(values) + adjacent_repeat_count(values)


def is_alignable(target: Sequence[int], input_length: int) -> bool:
    return int(input_length) >= minimum_ctc_steps(target)


def _validate_logits(logits: torch.Tensor, input_length: int | None) -> int:
    if logits.ndim != 2 or logits.shape[1] != PHONE_COUNT + 1:
        raise ValueError("logits must have shape [T,41]")
    available = int(logits.shape[0])
    steps = available if input_length is None else int(input_length)
    if steps < 0 or steps > available:
        raise ValueError("input_length must be within logits time dimension")
    return steps


def score_target(
    logits: torch.Tensor, target: Sequence[int], input_length: int | None = None
) -> HypothesisScore:
    """Score one target using frozen R5-2B empty/nonempty semantics."""

    values = _validate_phone_values(target, allow_empty=True)
    steps = _validate_logits(logits, input_length)
    minimum = minimum_ctc_steps(values)

    if not values:
        with torch.no_grad():
            log_probs = torch.log_softmax(logits[:steps], dim=-1)
            raw_tensor = log_probs[:, BLANK_INDEX].sum()
        raw_score = float(raw_tensor.detach().cpu().item())
        if not math.isfinite(raw_score):
            raise NonFiniteAlignableScoreError("Empty target produced non-finite all-blank score")
        return HypothesisScore(
            target=values,
            input_length=steps,
            minimum_steps=0,
            alignable=True,
            raw_score=raw_score,
            target_score=raw_score,
            ctc_called=False,
            empty_formula_used=True,
        )

    if steps < minimum:
        return HypothesisScore(
            target=values,
            input_length=steps,
            minimum_steps=minimum,
            alignable=False,
            raw_score=float("-inf"),
            target_score=float("-inf"),
            ctc_called=False,
            empty_formula_used=False,
        )

    with torch.no_grad():
        log_probs = torch.log_softmax(logits[:steps], dim=-1).unsqueeze(1)
        loss = torch.nn.CTCLoss(blank=BLANK_INDEX, reduction="none", zero_infinity=True)(
            log_probs,
            torch.tensor(values, dtype=torch.long, device=logits.device),
            torch.tensor([steps], dtype=torch.long, device=logits.device),
            torch.tensor([len(values)], dtype=torch.long, device=logits.device),
        )
    scalar_loss = float(loss.detach().cpu().item())
    if not math.isfinite(scalar_loss):
        raise NonFiniteAlignableScoreError("Alignable nonempty target returned non-finite CTCLoss")
    raw_score = -scalar_loss
    target_score = raw_score / max(len(values), 1)
    if not math.isfinite(raw_score) or not math.isfinite(target_score):
        raise NonFiniteAlignableScoreError("Alignable nonempty target produced non-finite score")
    return HypothesisScore(
        target=values,
        input_length=steps,
        minimum_steps=minimum,
        alignable=True,
        raw_score=raw_score,
        target_score=target_score,
        ctc_called=True,
        empty_formula_used=False,
    )


def construct_keep(expected: Sequence[int]) -> TargetCandidate:
    values = validate_expected(expected)
    return TargetCandidate(family="KEEP", target=values)


def enumerate_insert_candidates(expected: Sequence[int]) -> list[TargetCandidate]:
    values = validate_expected(expected)
    return [
        TargetCandidate(
            family="INSERT",
            target=values[:boundary] + (phone,) + values[boundary:],
            boundary=boundary,
            phone_index=phone,
        )
        for boundary in range(len(values) + 1)
        for phone in range(PHONE_COUNT)
    ]


def enumerate_sub_candidates(expected: Sequence[int]) -> list[TargetCandidate]:
    values = validate_expected(expected)
    output: list[TargetCandidate] = []
    for position, expected_phone in enumerate(values):
        for phone in range(PHONE_COUNT):
            if phone == expected_phone:
                continue
            target = list(values)
            target[position] = phone
            output.append(TargetCandidate(
                family="SUB", target=tuple(target), position=position, phone_index=phone
            ))
    return output


def enumerate_delete_candidates(expected: Sequence[int]) -> list[TargetCandidate]:
    values = validate_expected(expected)
    return [
        TargetCandidate(
            family="DELETE",
            target=values[:position] + values[position + 1 :],
            position=position,
        )
        for position in range(len(values))
    ]


def candidate_counts(expected_length: int) -> dict[str, int]:
    length = int(expected_length)
    if length < 1:
        raise ValueError("Expected length N must be at least one")
    counts = {
        "KEEP": 1,
        "INSERT": PHONE_COUNT * (length + 1),
        "SUB": (PHONE_COUNT - 1) * length,
        "DELETE": length,
    }
    counts["TOTAL"] = sum(counts.values())
    return counts


def _best_result(
    family: str,
    candidates: Iterable[ScoredCandidate],
    tie_key,
) -> BestCandidateResult:
    materialized = list(candidates)
    if any(item.candidate.family != family for item in materialized):
        raise ValueError(f"Expected only {family} candidates")
    alignable = [item for item in materialized if item.hypothesis.alignable]
    for item in alignable:
        if not math.isfinite(item.hypothesis.target_score):
            raise NonFiniteAlignableScoreError(f"Alignable {family} candidate is non-finite")
    if not alignable:
        return BestCandidateResult(
            exists=False,
            family=family,
            candidate=None,
            score=float("-inf"),
            total_candidates=len(materialized),
            alignable_candidates=0,
            impossible_candidates=len(materialized),
        )
    winner = min(
        alignable,
        key=lambda item: (-item.hypothesis.target_score, *tie_key(item.candidate)),
    )
    return BestCandidateResult(
        exists=True,
        family=family,
        candidate=winner.candidate,
        score=winner.hypothesis.target_score,
        total_candidates=len(materialized),
        alignable_candidates=len(alignable),
        impossible_candidates=len(materialized) - len(alignable),
    )


def select_best_insert(candidates: Iterable[ScoredCandidate]) -> BestCandidateResult:
    return _best_result(
        "INSERT", candidates, lambda c: (int(c.boundary), int(c.phone_index))
    )


def select_best_sub(candidates: Iterable[ScoredCandidate]) -> BestCandidateResult:
    return _best_result(
        "SUB", candidates, lambda c: (int(c.position), int(c.phone_index))
    )


def select_best_delete(candidates: Iterable[ScoredCandidate]) -> BestCandidateResult:
    return _best_result("DELETE", candidates, lambda c: (int(c.position),))


def select_best_nonaddition(
    keep: ScoredCandidate,
    best_sub: BestCandidateResult,
    best_delete: BestCandidateResult,
) -> BestNonAdditionResult:
    if keep.candidate.family != "KEEP":
        raise ValueError("KEEP candidate required")
    if not keep.hypothesis.alignable or not math.isfinite(keep.hypothesis.target_score):
        raise KeepAlignabilityError("KEEP is impossible or non-finite")
    choices: list[tuple[int, str, float, TargetCandidate]] = [
        (0, "KEEP", keep.hypothesis.target_score, keep.candidate)
    ]
    if best_sub.exists:
        choices.append((1, "SUB", best_sub.score, best_sub.candidate))
    if best_delete.exists:
        choices.append((2, "DELETE", best_delete.score, best_delete.candidate))
    maximum = max(item[2] for item in choices)
    winner = min((item for item in choices if item[2] == maximum), key=lambda item: item[0])
    return BestNonAdditionResult(family=winner[1], score=maximum, candidate=winner[3])


def compute_relation_competition_score(
    best_insert: BestCandidateResult, best_nonaddition: BestNonAdditionResult
) -> float:
    if not best_insert.exists:
        return float("-inf")
    value = best_insert.score - best_nonaddition.score
    if not math.isfinite(value):
        raise NonFiniteAlignableScoreError("Finite relation competitors produced non-finite C")
    return float(value)


def _score_candidates(
    logits: torch.Tensor, candidates: Iterable[TargetCandidate], input_length: int
) -> list[ScoredCandidate]:
    return [
        ScoredCandidate(candidate=item, hypothesis=score_target(logits, item.target, input_length))
        for item in candidates
    ]


def score_word(
    logits: torch.Tensor, expected: Sequence[int], input_length: int | None = None
) -> WordScore:
    values = validate_expected(expected)
    steps = _validate_logits(logits, input_length)
    keep_candidate = construct_keep(values)
    keep = ScoredCandidate(keep_candidate, score_target(logits, keep_candidate.target, steps))
    if not keep.hypothesis.alignable:
        raise KeepAlignabilityError("KEEP target is CTC-impossible")
    best_insert = select_best_insert(
        _score_candidates(logits, enumerate_insert_candidates(values), steps)
    )
    best_sub = select_best_sub(_score_candidates(logits, enumerate_sub_candidates(values), steps))
    best_delete = select_best_delete(
        _score_candidates(logits, enumerate_delete_candidates(values), steps)
    )
    best_nonaddition = select_best_nonaddition(keep, best_sub, best_delete)
    score = compute_relation_competition_score(best_insert, best_nonaddition)
    return WordScore(keep, best_insert, best_sub, best_delete, best_nonaddition, score)


def decision_output(word_id: str, result: WordScore, threshold: float) -> dict[str, object]:
    theta = float(threshold)
    if not math.isfinite(theta):
        raise ValueError("Operational threshold must be finite")
    positive = result.relation_competition_score >= theta and result.best_insert.exists
    event = None
    if positive:
        candidate = result.best_insert.candidate
        event = {
            "word_id": str(word_id),
            "phone_index": int(candidate.phone_index),
            "boundary": int(candidate.boundary),
        }
    return {
        "prediction": "ADDITION" if positive else "NON_ADDITION",
        "addition_event": event,
        "diagnostic_best_nonaddition_family": result.best_nonaddition.family,
        "diagnostic_is_confirmed_multiclass_relation": False,
    }


def serialize_extended_score(value: float) -> dict[str, float | bool | None]:
    scalar = float(value)
    if math.isnan(scalar) or scalar == float("inf"):
        raise ValueError("Only finite values and negative infinity are serializable")
    if scalar == float("-inf"):
        return {"score_value": None, "score_is_neg_inf": True}
    return {"score_value": scalar, "score_is_neg_inf": False}


def deserialize_extended_score(payload: Mapping[str, object]) -> float:
    flag = payload.get("score_is_neg_inf")
    value = payload.get("score_value")
    if flag is True:
        if value is not None:
            raise ValueError("Negative-infinity encoding requires null score_value")
        return float("-inf")
    if flag is False and isinstance(value, (int, float)):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError("Finite encoding requires a finite score")
        return scalar
    raise ValueError("Malformed extended-real score payload")


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("scores may contain only finite values or negative infinity")
    finite = np.unique(values[np.isfinite(values)])
    if finite.size == 0:
        raise ValueError("No finite threshold scores")
    return np.concatenate((
        np.asarray([np.nextafter(finite[0], -np.inf)], dtype=np.float64),
        finite,
        np.asarray([np.nextafter(finite[-1], np.inf)], dtype=np.float64),
    ))


def predict_addition(scores: Sequence[float], threshold: float) -> np.ndarray:
    theta = float(threshold)
    if not math.isfinite(theta):
        raise ValueError("threshold must be finite")
    values = np.asarray(scores, dtype=np.float64)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("scores may contain only finite values or negative infinity")
    return values >= theta


def threshold_choice_key(record: Mapping[str, float]) -> tuple[float, float, float, float]:
    return (
        float(record["binary_macro_f1"]),
        float(record["addition_f1"]),
        -float(record["correct_only_far"]),
        float(record["threshold"]),
    )


def select_threshold_record(records: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    if not records:
        raise ValueError("At least one threshold record is required")
    return max(records, key=threshold_choice_key)


def cohort_false_addition_rates(
    predictions: Sequence[bool], cohort_masks: Mapping[str, Sequence[bool]]
) -> dict[str, dict[str, float | int]]:
    predicted = np.asarray(predictions, dtype=np.bool_)
    output: dict[str, dict[str, float | int]] = {}
    for name, mask_values in cohort_masks.items():
        mask = np.asarray(mask_values, dtype=np.bool_)
        if mask.shape != predicted.shape:
            raise ValueError("Cohort mask shape mismatch")
        support = int(mask.sum())
        false_additions = int(np.logical_and(predicted, mask).sum())
        output[name] = {
            "support": support,
            "false_additions": false_additions,
            "rate": false_additions / support if support else 0.0,
        }
    return output


def event_metrics(
    true_events: Sequence[AdditionEvent], predicted_events: Sequence[AdditionEvent]
) -> dict[str, float | int]:
    predicted_words = [event.word_id for event in predicted_events]
    if len(predicted_words) != len(set(predicted_words)):
        raise ValueError("At most one predicted BEST_INSERT event is allowed per word")
    truth_counter = Counter(
        (event.word_id, int(event.phone_index), int(event.boundary)) for event in true_events
    )
    prediction_counter = Counter(
        (event.word_id, int(event.phone_index), int(event.boundary))
        for event in predicted_events
    )
    true_positive = sum(
        min(count, prediction_counter.get(key, 0)) for key, count in truth_counter.items()
    )
    false_positive = len(predicted_events) - true_positive
    false_negative = len(true_events) - true_positive
    precision = true_positive / len(predicted_events) if predicted_events else 0.0
    recall = true_positive / len(true_events) if true_events else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
