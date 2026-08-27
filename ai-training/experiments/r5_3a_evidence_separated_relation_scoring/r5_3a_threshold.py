"""Frozen R5 threshold candidates, decision, metrics, and tie semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdEvaluation:
    threshold: float
    binary_macro_f1: float
    addition_f1: float
    correct_only_far: float


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("threshold scores must be a nonempty finite float64 vector")
    unique = np.unique(values)
    lower = np.nextafter(unique[0], -np.inf)
    upper = np.nextafter(unique[-1], np.inf)
    return np.unique(np.concatenate(([lower], unique, [upper]))).astype(np.float64)


def apply_threshold(scores: np.ndarray, theta: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    threshold = np.float64(theta)
    if not np.all(np.isfinite(values)) or not np.isfinite(threshold):
        raise ValueError("decision inputs must be finite")
    return values >= threshold


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


def evaluate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    correct_only_negative: np.ndarray,
    theta: float,
) -> ThresholdEvaluation:
    y = np.asarray(labels, dtype=np.int64)
    correct = np.asarray(correct_only_negative, dtype=bool)
    pred = apply_threshold(scores, theta)
    if y.shape != pred.shape or correct.shape != pred.shape:
        raise ValueError("threshold metadata shapes differ")
    positive = y == 1
    negative = ~positive
    tp = int(np.count_nonzero(pred & positive))
    fp = int(np.count_nonzero(pred & negative))
    fn = int(np.count_nonzero(~pred & positive))
    tn = int(np.count_nonzero(~pred & negative))
    addition_f1 = _f1(tp, fp, fn)
    nonaddition_f1 = _f1(tn, fn, fp)
    support = int(correct.sum())
    correct_far = 0.0 if support == 0 else int(np.count_nonzero(pred & correct)) / support
    return ThresholdEvaluation(
        threshold=float(theta),
        binary_macro_f1=(addition_f1 + nonaddition_f1) / 2.0,
        addition_f1=addition_f1,
        correct_only_far=correct_far,
    )


def choose_best_evaluation(evaluations: list[ThresholdEvaluation]) -> ThresholdEvaluation:
    if not evaluations:
        raise ValueError("at least one threshold evaluation is required")
    return max(
        evaluations,
        key=lambda item: (
            item.binary_macro_f1,
            item.addition_f1,
            -item.correct_only_far,
            item.threshold,
        ),
    )


def select_threshold(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_correct_only_negative: np.ndarray,
) -> tuple[float, list[ThresholdEvaluation]]:
    evaluations = [
        evaluate_threshold(
            calibration_scores,
            calibration_labels,
            calibration_correct_only_negative,
            float(theta),
        )
        for theta in threshold_candidates(calibration_scores)
    ]
    best = choose_best_evaluation(evaluations)
    return best.threshold, evaluations
