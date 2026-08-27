"""Frozen R6-0 boundary-time and local-window helpers.

This module is deliberately independent of acoustic models and data readers.
"""

from __future__ import annotations

from decimal import Decimal
from math import isfinite
from typing import Sequence


OUTPUT_HOP_SECONDS = Decimal("0.020")
OUTPUT_FIRST_CENTER_SECONDS = Decimal("0.005")
WINDOW_RADIUS = 2


def _decimal_seconds(value: float) -> Decimal:
    if not isfinite(float(value)):
        raise ValueError("time must be finite")
    return Decimal(str(float(value)))


def gold_boundary_time(
    expected_intervals: Sequence[tuple[float, float]], boundary_index: int
) -> float:
    """Map an expected-sequence boundary to its frozen GOLD oracle time."""
    if not expected_intervals:
        raise ValueError("expected sequence must be nonempty")
    if not 0 <= boundary_index <= len(expected_intervals):
        raise ValueError("boundary index outside expected sequence")
    normalized: list[tuple[Decimal, Decimal]] = []
    for start, end in expected_intervals:
        start_d = _decimal_seconds(start)
        end_d = _decimal_seconds(end)
        if end_d < start_d:
            raise ValueError("phone interval end precedes start")
        normalized.append((start_d, end_d))
    if boundary_index == 0:
        return float(normalized[0][0])
    if boundary_index == len(normalized):
        return float(normalized[-1][1])
    left_end = normalized[boundary_index - 1][1]
    right_start = normalized[boundary_index][0]
    return float((left_end + right_start) / Decimal(2))


def relative_boundary_time(gold_time_seconds: float, mfa_word_start_seconds: float) -> float:
    """Subtract the utterance-level MFA word start exactly once."""
    return float(_decimal_seconds(gold_time_seconds) - _decimal_seconds(mfa_word_start_seconds))


def nominal_output_center(output_index: int) -> float:
    if output_index < 0:
        raise ValueError("output index must be nonnegative")
    return float(OUTPUT_FIRST_CENTER_SECONDS + OUTPUT_HOP_SECONDS * output_index)


def nearest_output_index(
    gold_time_seconds: float,
    mfa_word_start_seconds: float,
    output_steps: int,
    mfa_word_end_seconds: float,
) -> int:
    """Choose the nearest nominal center, breaking exact ties toward lower k."""
    if output_steps <= 0:
        raise ValueError("output_steps must be positive")
    gold = _decimal_seconds(gold_time_seconds)
    start = _decimal_seconds(mfa_word_start_seconds)
    end = _decimal_seconds(mfa_word_end_seconds)
    if end <= start:
        raise ValueError("MFA word end must follow start")
    if gold < start or gold > end:
        raise ValueError("GOLD boundary lies outside historical MFA word crop")
    relative = gold - start
    best_index = 0
    best_distance = abs(OUTPUT_FIRST_CENTER_SECONDS - relative)
    for index in range(1, output_steps):
        center = OUTPUT_FIRST_CENTER_SECONDS + OUTPUT_HOP_SECONDS * index
        distance = abs(center - relative)
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def five_step_window(center_index: int, output_steps: int) -> list[int]:
    """Return k-2..k+2 intersected with available output indices."""
    if output_steps <= 0:
        raise ValueError("output_steps must be positive")
    if not 0 <= center_index < output_steps:
        raise ValueError("center index outside output tensor")
    lower = max(0, center_index - WINDOW_RADIUS)
    upper = min(output_steps - 1, center_index + WINDOW_RADIUS)
    return list(range(lower, upper + 1))

