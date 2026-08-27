"""Frozen R6-0 boundary-label and synthetic coverage semantics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


def boundary_event_counts(expected_length: int, events: Iterable[Mapping[str, Any]]) -> list[int]:
    if expected_length <= 0:
        raise ValueError("expected sequence length must be positive")
    counts: Counter[int] = Counter()
    for event in events:
        boundary = int(event["boundary"])
        if not 0 <= boundary <= expected_length:
            raise ValueError("Addition event boundary outside expected sequence")
        counts[boundary] += 1
    return [counts[index] for index in range(expected_length + 1)]


def build_boundary_labels(
    source_identity: str,
    expected_length: int,
    events: Iterable[Mapping[str, Any]],
    relation_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    counts = boundary_event_counts(expected_length, events)
    metadata = dict(relation_metadata or {})
    return [
        {
            "boundary_identity": f"{source_identity}|{boundary}",
            "source_identity": source_identity,
            "boundary_index": boundary,
            "addition_event_count": count,
            "addition_boundary_label": count > 0,
            "relation_metadata": metadata,
        }
        for boundary, count in enumerate(counts)
    ]


def synthetic_event_window_coverage(
    events: Iterable[Mapping[str, Any]], valid_boundary_indices: set[int]
) -> dict[str, float | int]:
    event_list = list(events)
    covered = sum(int(event["boundary"]) in valid_boundary_indices for event in event_list)
    total = len(event_list)
    return {
        "events": total,
        "covered_events": covered,
        "coverage": float(covered / total) if total else 0.0,
    }

