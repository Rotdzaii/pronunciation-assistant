from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AlignmentQualityConfig:
    boundary_tolerance_seconds: float = 0.03
    minimum_coverage_ratio: float = 0.40
    warning_coverage_ratio: float = 0.65
    minimum_word_coverage_ratio: float = 0.60
    very_short_phone_seconds: float = 0.015
    very_long_phone_seconds: float = 1.50
    max_short_phone_ratio: float = 0.25

    @classmethod
    def from_env(cls) -> "AlignmentQualityConfig":
        return cls(
            boundary_tolerance_seconds=_env_float("MFA_BOUNDARY_TOLERANCE_SECONDS", 0.03),
            minimum_coverage_ratio=_env_float("MFA_MIN_COVERAGE_RATIO", 0.40),
            warning_coverage_ratio=_env_float("MFA_WARNING_COVERAGE_RATIO", 0.65),
            minimum_word_coverage_ratio=_env_float("MFA_MIN_WORD_COVERAGE_RATIO", 0.60),
            very_short_phone_seconds=_env_float("MFA_VERY_SHORT_PHONE_SECONDS", 0.015),
            very_long_phone_seconds=_env_float("MFA_VERY_LONG_PHONE_SECONDS", 1.50),
            max_short_phone_ratio=_env_float("MFA_MAX_VERY_SHORT_PHONE_RATIO", 0.25),
        )


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in {None, ""}:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _union_duration(segments: list[dict[str, Any]]) -> float:
    ranges = sorted(
        (max(0.0, _number(item.get("start"))), max(0.0, _number(item.get("end"))))
        for item in segments
        if _number(item.get("end")) > _number(item.get("start"))
    )
    total = 0.0
    current_start: float | None = None
    current_end: float | None = None
    for start, end in ranges:
        if current_start is None:
            current_start, current_end = start, end
        elif start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + ((current_end - current_start) if current_start is not None and current_end is not None else 0.0)


def validate_alignment_quality(
    *,
    words: list[dict[str, Any]],
    phones: list[dict[str, Any]],
    audio_duration: float,
    expected_word_count: int = 0,
    oov_count: int = 0,
    empty_interval_count: int = 0,
    config: AlignmentQualityConfig | None = None,
) -> dict[str, Any]:
    """Assess timing integrity. This is not a pronunciation score."""

    config = config or AlignmentQualityConfig.from_env()
    duration = max(float(audio_duration), 0.0)
    ordered_phones = sorted(phones, key=lambda segment: (_number(segment.get("start")), _number(segment.get("end"))))
    overlap_count = 0
    out_of_bounds_count = 0
    zero_duration_count = 0
    very_short_phone_count = 0
    very_long_phone_count = 0
    previous_end: float | None = None
    for segment in ordered_phones:
        start = _number(segment.get("start"))
        end = _number(segment.get("end"))
        phone_duration = end - start
        if start < -config.boundary_tolerance_seconds or end > duration + config.boundary_tolerance_seconds:
            out_of_bounds_count += 1
        if phone_duration <= 0:
            zero_duration_count += 1
        elif phone_duration < config.very_short_phone_seconds:
            very_short_phone_count += 1
        elif phone_duration > config.very_long_phone_seconds:
            very_long_phone_count += 1
        if previous_end is not None and start < previous_end - config.boundary_tolerance_seconds:
            overlap_count += 1
        previous_end = max(previous_end or end, end)

    aligned_duration = _union_duration(ordered_phones)
    coverage = aligned_duration / duration if duration > 0 else 0.0
    word_coverage = len(words) / expected_word_count if expected_word_count else 1.0
    short_ratio = very_short_phone_count / len(ordered_phones) if ordered_phones else 1.0
    issues: list[str] = []
    status = "ok"

    def fail(message: str) -> None:
        nonlocal status
        status = "failed"
        issues.append(message)

    def warn(message: str) -> None:
        nonlocal status
        if status == "ok":
            status = "warning"
        issues.append(message)

    if duration <= 0:
        fail("audio_duration_invalid")
    if not ordered_phones:
        fail("no_phone_segments")
    if not words:
        fail("no_word_segments")
    if zero_duration_count:
        fail("zero_duration_phone")
    if out_of_bounds_count:
        fail("boundary_out_of_bounds")
    if overlap_count:
        fail("overlapping_phone_segments")
    if coverage < config.minimum_coverage_ratio:
        fail("speech_coverage_too_low")
    elif coverage < config.warning_coverage_ratio:
        warn("speech_coverage_low")
    if expected_word_count and word_coverage < config.minimum_word_coverage_ratio:
        fail("word_coverage_too_low")
    if very_long_phone_count:
        warn("very_long_phone_segments")
    if short_ratio > config.max_short_phone_ratio:
        warn("too_many_very_short_phone_segments")
    if oov_count:
        warn("transcript_contains_oov")

    score = 1.0
    score -= min(0.45, max(0.0, config.warning_coverage_ratio - coverage))
    score -= min(0.30, very_short_phone_count * 0.03 + very_long_phone_count * 0.05)
    score -= min(0.40, overlap_count * 0.20 + out_of_bounds_count * 0.20 + zero_duration_count * 0.25)
    score -= min(0.20, max(0.0, 1.0 - word_coverage) * 0.20)
    if status == "failed":
        score = min(score, 0.39)

    return {
        "status": status,
        "quality_score": round(max(0.0, min(1.0, score)), 3),
        "issues": issues,
        "metrics": {
            "audio_duration": round(duration, 3),
            "aligned_duration": round(aligned_duration, 3),
            "speech_coverage_ratio": round(coverage, 3),
            "number_of_words": len(words),
            "number_of_phones": len(ordered_phones),
            "expected_word_count": expected_word_count,
            "word_coverage_ratio": round(word_coverage, 3),
            "oov_count": oov_count,
            "empty_interval_count": empty_interval_count,
            "overlap_count": overlap_count,
            "out_of_bounds_count": out_of_bounds_count,
            "zero_duration_phone_count": zero_duration_count,
            "very_short_phone_count": very_short_phone_count,
            "very_long_phone_count": very_long_phone_count,
        },
        "thresholds": asdict(config),
    }
