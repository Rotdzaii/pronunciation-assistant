"""Frozen R6-0 local posterior feature definitions."""

from __future__ import annotations

from typing import Sequence

import torch


PHONE_CLASS_COUNT = 40
BLANK_INDEX = 40
CLASS_COUNT = 41
PRIMARY_FEATURE = "MEAN_UNEXPECTED_PHONE_MASS"
DESCRIPTIVE_FEATURES = (
    "PEAK_UNEXPECTED_PHONE_POSTERIOR",
    "MEAN_NONBLANK_MASS",
)


def posterior_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != CLASS_COUNT:
        raise ValueError("logits must have shape [T,41]")
    if logits.dtype != torch.float32:
        raise ValueError("frozen synthetic/future logits dtype is float32")
    if logits.shape[0] <= 0 or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be nonempty and finite")
    return torch.exp(torch.log_softmax(logits, dim=-1))


def adjacent_expected_phone_set(
    expected_phone_ids: Sequence[int], boundary_index: int
) -> frozenset[int]:
    if not expected_phone_ids:
        raise ValueError("expected phone sequence must be nonempty")
    phones = tuple(int(value) for value in expected_phone_ids)
    if any(value < 0 or value >= PHONE_CLASS_COUNT for value in phones):
        raise ValueError("expected phone id outside canonical vocabulary")
    if not 0 <= boundary_index <= len(phones):
        raise ValueError("boundary index outside expected sequence")
    adjacent: set[int] = set()
    if boundary_index > 0:
        adjacent.add(phones[boundary_index - 1])
    if boundary_index < len(phones):
        adjacent.add(phones[boundary_index])
    return frozenset(adjacent)


def _validated_window(window_indices: Sequence[int], time_steps: int) -> tuple[int, ...]:
    indices = tuple(int(value) for value in window_indices)
    if not indices:
        raise ValueError("window must contain at least one frame")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("window indices must be unique and ascending")
    if indices[0] < 0 or indices[-1] >= time_steps:
        raise ValueError("window index outside logits")
    return indices


def compute_local_evidence(
    logits: torch.Tensor,
    window_indices: Sequence[int],
    expected_phone_ids: Sequence[int],
    boundary_index: int,
) -> dict[str, float | int | list[int]]:
    """Calculate exactly one primary feature and two descriptive controls."""
    posterior = posterior_from_logits(logits)
    indices = _validated_window(window_indices, posterior.shape[0])
    adjacent = adjacent_expected_phone_set(expected_phone_ids, boundary_index)
    allowed_phone_ids = [index for index in range(PHONE_CLASS_COUNT) if index not in adjacent]
    selected = posterior[list(indices)]
    selected64 = selected.to(torch.float64)

    unexpected_by_frame = selected64[:, allowed_phone_ids].sum(dim=1)
    mean_unexpected = float(unexpected_by_frame.mean().item())

    unexpected_matrix = selected64[:, allowed_phone_ids]
    peak_flat_index = int(torch.argmax(unexpected_matrix.reshape(-1)).item())
    phone_width = len(allowed_phone_ids)
    peak_window_offset = peak_flat_index // phone_width
    peak_phone_offset = peak_flat_index % phone_width
    peak_value = float(unexpected_matrix.reshape(-1)[peak_flat_index].item())

    mean_nonblank = float((1.0 - selected64[:, BLANK_INDEX]).mean().item())
    nonblank_from_sum = float(selected64[:, :PHONE_CLASS_COUNT].sum(dim=1).mean().item())

    return {
        "window_indices": list(indices),
        "adjacent_expected_phone_ids": sorted(adjacent),
        "mean_unexpected_phone_mass": mean_unexpected,
        "peak_unexpected_phone_posterior": peak_value,
        "peak_unexpected_phone_index": allowed_phone_ids[peak_phone_offset],
        "peak_unexpected_frame_index": indices[peak_window_offset],
        "mean_nonblank_mass": mean_nonblank,
        "mean_nonblank_mass_from_phone_sum": nonblank_from_sum,
    }

