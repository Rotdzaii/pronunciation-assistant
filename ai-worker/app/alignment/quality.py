from __future__ import annotations

import os
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AlignmentQualityConfig:
    boundary_tolerance_seconds: float = 0.03
    # Kept for backward compatibility. This is raw-audio coverage and is now a warning, not a hard failure.
    minimum_coverage_ratio: float = 0.40
    warning_coverage_ratio: float = 0.65
    # Legacy configuration name. The gate measures phone-union / aligned span,
    # not coverage of speech detected by VAD.
    minimum_speech_coverage_ratio: float = 0.70
    minimum_aligned_duration_seconds: float = 0.12
    maximum_boundary_silence_seconds: float = 0.50
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
            minimum_speech_coverage_ratio=_env_float("MFA_MIN_SPEECH_COVERAGE_RATIO", 0.70),
            minimum_aligned_duration_seconds=_env_float("MFA_MIN_ALIGNED_DURATION_SECONDS", 0.12),
            maximum_boundary_silence_seconds=_env_float("MFA_MAX_BOUNDARY_SILENCE_SECONDS", 0.50),
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


def _aligned_span(segments: list[dict[str, Any]]) -> tuple[float | None, float | None, float]:
    valid = [
        (_number(item.get("start")), _number(item.get("end")))
        for item in segments
        if _number(item.get("end")) > _number(item.get("start"))
    ]
    if not valid:
        return None, None, 0.0
    start = min(item[0] for item in valid)
    end = max(item[1] for item in valid)
    return start, end, max(0.0, end - start)


def _transcript_match_ratio(words: list[dict[str, Any]], expected_words: list[str] | None) -> tuple[int, float]:
    if not expected_words:
        return len(words), 1.0
    expected = Counter(str(word).casefold() for word in expected_words if str(word).strip())
    actual = Counter(str(word.get("word") or "").casefold() for word in words if str(word.get("word") or "").strip())
    matched = sum(min(actual[word], count) for word, count in expected.items())
    return matched, matched / sum(expected.values()) if expected else 1.0


def validate_alignment_quality(
    *,
    words: list[dict[str, Any]],
    phones: list[dict[str, Any]],
    audio_duration: float,
    expected_word_count: int = 0,
    expected_words: list[str] | None = None,
    oov_count: int = 0,
    empty_interval_count: int = 0,
    config: AlignmentQualityConfig | None = None,
) -> dict[str, Any]:
    """Assess alignment timing integrity. This is not pronunciation correctness."""

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

    aligned_phone_duration = _union_duration(ordered_phones)
    aligned_word_duration = _union_duration(words)
    span_start, span_end, aligned_span = _aligned_span(words or ordered_phones)
    raw_audio_coverage = aligned_phone_duration / duration if duration > 0 else 0.0
    phone_span_fill = aligned_phone_duration / aligned_span if aligned_span > 0 else 0.0
    leading_silence = max(0.0, span_start or 0.0)
    trailing_silence = max(0.0, duration - (span_end or duration))
    word_coverage = len(words) / expected_word_count if expected_word_count else 1.0
    matched_word_count, transcript_match_ratio = _transcript_match_ratio(words, expected_words)
    short_ratio = very_short_phone_count / len(ordered_phones) if ordered_phones else 1.0
    failures: list[str] = []
    warnings: list[str] = []

    def fail(message: str) -> None:
        failures.append(message)

    def warn(message: str) -> None:
        warnings.append(message)

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
    if aligned_phone_duration < config.minimum_aligned_duration_seconds:
        fail("aligned_duration_too_short")
    if phone_span_fill < config.minimum_speech_coverage_ratio:
        fail("speech_relative_coverage_too_low")
    if expected_word_count and word_coverage < config.minimum_word_coverage_ratio:
        fail("word_coverage_too_low")
    if expected_words and transcript_match_ratio < config.minimum_word_coverage_ratio:
        fail("transcript_word_coverage_too_low")
    if expected_words and matched_word_count == 0:
        fail("transcript_word_mismatch")

    if raw_audio_coverage < config.minimum_coverage_ratio:
        warn("raw_audio_coverage_below_minimum")
    elif raw_audio_coverage < config.warning_coverage_ratio:
        warn("raw_audio_coverage_low")
    if leading_silence > config.maximum_boundary_silence_seconds:
        warn("leading_silence_high")
    if trailing_silence > config.maximum_boundary_silence_seconds:
        warn("trailing_silence_high")
    if very_long_phone_count:
        warn("very_long_phone_segments")
    if short_ratio > config.max_short_phone_ratio:
        warn("too_many_very_short_phone_segments")
    if oov_count:
        warn("transcript_contains_oov")

    status = "failed" if failures else ("warning" if warnings else "ok")
    decision = "invalid" if failures else ("valid_with_warning" if warnings else "valid")
    score = 1.0
    score -= min(0.25, max(0.0, config.warning_coverage_ratio - raw_audio_coverage))
    score -= min(0.30, very_short_phone_count * 0.03 + very_long_phone_count * 0.05)
    score -= min(0.40, overlap_count * 0.20 + out_of_bounds_count * 0.20 + zero_duration_count * 0.25)
    score -= min(0.20, max(0.0, 1.0 - transcript_match_ratio) * 0.20)
    if failures:
        score = min(score, 0.39)

    metrics = {
        "audio_duration": round(duration, 3),
        "aligned_duration": round(aligned_phone_duration, 3),
        "raw_audio_coverage_ratio": round(raw_audio_coverage, 3),
        # Legacy alias retained for existing consumers. It is raw audio coverage, not a speech-quality score.
        "speech_coverage_ratio": round(raw_audio_coverage, 3),
        "phone_span_fill_ratio": round(phone_span_fill, 3),
        # Deprecated alias for existing local consumers. The primary metric is
        # phone_span_fill_ratio because the denominator is the aligned span.
        "speech_relative_coverage_ratio": round(phone_span_fill, 3),
        "aligned_word_duration_seconds": round(aligned_word_duration, 3),
        "aligned_phone_duration_seconds": round(aligned_phone_duration, 3),
        "aligned_span_seconds": round(aligned_span, 3),
        "leading_silence_seconds": round(leading_silence, 3),
        "trailing_silence_seconds": round(trailing_silence, 3),
        "number_of_words": len(words),
        "number_of_phones": len(ordered_phones),
        "expected_word_count": expected_word_count,
        "word_coverage_ratio": round(word_coverage, 3),
        "transcript_word_match_ratio": round(transcript_match_ratio, 3),
        "matched_word_count": matched_word_count,
        "oov_count": oov_count,
        "empty_interval_count": empty_interval_count,
        "overlap_count": overlap_count,
        "out_of_bounds_count": out_of_bounds_count,
        "zero_duration_phone_count": zero_duration_count,
        "very_short_phone_count": very_short_phone_count,
        "very_long_phone_count": very_long_phone_count,
    }
    return {
        "status": status,
        "decision": decision,
        "alignment_quality_status": decision,
        "alignment_quality_warnings": warnings,
        "alignment_quality_failures": failures,
        "quality_score": round(max(0.0, min(1.0, score)), 3),
        "issues": [*failures, *warnings],
        "warnings": warnings,
        "failures": failures,
        "metrics": metrics,
        "thresholds": asdict(config),
    }
