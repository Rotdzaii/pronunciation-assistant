from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from app.contracts.alignment_contract import AlignmentError, MFA_ALIGNMENT_NOTE, build_alignment_result, build_alignment_segment


WORD_TIER_NAMES = {"word", "words", "orthography", "transcript"}
PHONE_TIER_NAMES = {"phone", "phones", "phoneme", "phonemes", "segment", "segments"}


def _unquote(value: str) -> str:
    value = value.strip()
    return value[1:-1].replace('""', '"') if len(value) >= 2 and value[0] == '"' and value[-1] == '"' else value


def _extract_value(line: str) -> str | None:
    return line.split("=", 1)[1].strip() if "=" in line else None


def _tier_kind(name: str | None) -> str | None:
    normalized = unicodedata.normalize("NFKC", name or "").strip().casefold()
    if normalized in WORD_TIER_NAMES or "word" in normalized:
        return "word"
    if normalized in PHONE_TIER_NAMES or "phone" in normalized or "phon" in normalized:
        return "phone"
    return None


def _parse_float(value: str | None, field: str, path: Path) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise AlignmentError(f"Invalid TextGrid {field}.", code="textgrid_invalid") from exc


def normalize_phone(phone: str) -> str:
    return unicodedata.normalize("NFKC", phone).strip()


def _word_for_phone(phone: dict[str, Any], words: list[dict[str, Any]]) -> str | None:
    start = float(phone["start"])
    end = float(phone["end"])
    for word in words:
        if start >= float(word["start"]) - 0.001 and end <= float(word["end"]) + 0.001:
            return word.get("word")
    return None


def parse_textgrid(textgrid_path: str | Path) -> dict[str, Any]:
    """Parse long-form MFA/Praat TextGrid interval tiers into normalized timings."""

    path = Path(textgrid_path)
    if not path.is_file():
        raise AlignmentError("TextGrid file was not found.", code="textgrid_missing")
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        try:
            lines = path.read_text(encoding="utf-16").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise AlignmentError("TextGrid could not be decoded.", code="textgrid_invalid") from exc
    except OSError as exc:
        raise AlignmentError("TextGrid could not be read.", code="textgrid_invalid") from exc

    raw_tiers: dict[str, list[tuple[float, float, str]]] = {"word": [], "phone": []}
    current_kind: str | None = None
    pending: dict[str, Any] | None = None
    empty_intervals = 0
    for raw_line in lines:
        line = raw_line.strip()
        value = _extract_value(line)
        if line.startswith("name") and value is not None:
            current_kind = _tier_kind(_unquote(value))
            pending = None
            continue
        if current_kind is None:
            continue
        if re.match(r"intervals\s*\[\d+\]:", line):
            pending = {}
            continue
        if pending is None or value is None:
            continue
        if line.startswith("xmin"):
            pending["start"] = _parse_float(value, "xmin", path)
        elif line.startswith("xmax"):
            pending["end"] = _parse_float(value, "xmax", path)
        elif line.startswith("text"):
            label = _unquote(value).strip()
            if not label:
                empty_intervals += 1
            elif {"start", "end"} <= pending.keys():
                raw_tiers[current_kind].append((pending["start"], pending["end"], label))
            pending = None

    if not raw_tiers["word"] and not raw_tiers["phone"]:
        raise AlignmentError("TextGrid contains no recognized word or phone intervals.", code="textgrid_invalid")

    words = [
        build_alignment_segment(index=index, segment_type="word", word=label, start=start, end=end)
        for index, (start, end, label) in enumerate(sorted(raw_tiers["word"]))
    ]
    phones = [
        build_alignment_segment(
            index=index,
            segment_type="phone",
            phone=normalize_phone(label),
            start=start,
            end=end,
        )
        for index, (start, end, label) in enumerate(sorted(raw_tiers["phone"]))
    ]
    for phone in phones:
        phone["word"] = _word_for_phone(phone, words)

    return build_alignment_result(
        segments=phones or words,
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
            "empty_interval_count": empty_intervals,
        },
    )
