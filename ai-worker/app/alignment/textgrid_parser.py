from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.contracts.alignment_contract import (
    AlignmentError,
    MFA_ALIGNMENT_NOTE,
    build_alignment_result,
    build_alignment_segment,
)


WORD_TIER_NAMES = {"word", "words"}
PHONE_TIER_NAMES = {"phone", "phones"}


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('""', '"')
    return value


def _extract_value(line: str) -> str | None:
    if "=" not in line:
        return None
    return line.split("=", 1)[1].strip()


def _tier_kind(name: str | None) -> str | None:
    normalized = (name or "").strip().lower()
    if normalized in WORD_TIER_NAMES:
        return "word"
    if normalized in PHONE_TIER_NAMES:
        return "phone"
    return None


def _parse_float(raw_value: str | None, field_name: str, path: Path) -> float:
    try:
        return float(str(raw_value).strip())
    except (TypeError, ValueError) as exc:
        raise AlignmentError(f"Invalid TextGrid {field_name} in {path}") from exc


def parse_textgrid(textgrid_path: str | Path) -> dict[str, Any]:
    """Parse common long-form Praat/MFA TextGrid interval tiers.

    The parser intentionally avoids external dependencies. It supports common
    MFA TextGrid files with IntervalTier names such as words/Words and
    phones/Phones.
    """

    path = Path(textgrid_path)
    if not path.exists():
        raise AlignmentError(f"TextGrid file not found: {path}")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="utf-16").splitlines()
    except OSError as exc:
        raise AlignmentError(f"Unable to read TextGrid file: {path}") from exc

    tiers: dict[str, list[dict[str, Any]]] = {"word": [], "phone": []}
    current_tier_name: str | None = None
    current_tier_kind: str | None = None
    pending_interval: dict[str, Any] | None = None

    for raw_line in lines:
        line = raw_line.strip()
        value = _extract_value(line)
        if line.startswith("name") and value is not None:
            current_tier_name = _unquote(value)
            current_tier_kind = _tier_kind(current_tier_name)
            pending_interval = None
            continue

        if current_tier_kind is None:
            continue

        if re.match(r"intervals\s*\[\d+\]:", line):
            pending_interval = {}
            continue

        if pending_interval is None or value is None:
            continue

        if line.startswith("xmin"):
            pending_interval["start"] = _parse_float(value, "xmin", path)
        elif line.startswith("xmax"):
            pending_interval["end"] = _parse_float(value, "xmax", path)
        elif line.startswith("text"):
            label = _unquote(value).strip()
            if label and {"start", "end"} <= pending_interval.keys():
                index = len(tiers[current_tier_kind])
                segment = build_alignment_segment(
                    index=index,
                    segment_type=current_tier_kind,
                    phone=label if current_tier_kind == "phone" else None,
                    word=label if current_tier_kind == "word" else None,
                    start=pending_interval["start"],
                    end=pending_interval["end"],
                )
                tiers[current_tier_kind].append(segment)
            pending_interval = None

    words = tiers["word"]
    phones = tiers["phone"]
    if not words and not phones:
        raise AlignmentError(
            f"No word or phone intervals found in TextGrid {path}. Expected tiers named words/word or phones/phone."
        )

    segments = phones or words
    return build_alignment_result(
        segments=segments,
        status="success",
        method="mfa",
        note=MFA_ALIGNMENT_NOTE,
        words=words,
        phones=phones,
        metadata={
            "is_forced_alignment": True,
            "is_fallback": False,
            "mfa_used": True,
            "textgrid_parse_success": True,
            "fallback_alignment": False,
            "word_segments_count": len(words),
            "phone_segments_count": len(phones),
        },
    )
