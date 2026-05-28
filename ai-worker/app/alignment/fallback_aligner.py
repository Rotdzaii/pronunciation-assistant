from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import Any

from app.contracts.alignment_contract import (
    FALLBACK_ALIGNMENT_NOTE,
    build_alignment_result,
    build_alignment_segment,
)


def _audio_duration_seconds(audio_path: str | Path) -> float:
    path = Path(audio_path)
    try:
        with wave.open(str(path), "rb") as audio_file:
            frame_count = audio_file.getnframes()
            frame_rate = audio_file.getframerate()
            if frame_rate > 0:
                return max(frame_count / float(frame_rate), 0.001)
    except (wave.Error, FileNotFoundError, OSError):
        pass

    try:
        import librosa

        return max(float(librosa.get_duration(path=str(path))), 0.001)
    except Exception:
        return 1.0


def _prompt_words(prompt_text: str | None) -> list[str]:
    return re.findall(r"[A-Za-z']+", prompt_text or "")


def _split_evenly(total_duration: float, count: int) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    step = total_duration / count
    return [(index * step, (index + 1) * step) for index in range(count)]


def align_prompt_fallback(
    audio_path: str | Path,
    prompt_text: str | None = None,
    canonical_phones: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Create an approximate alignment by evenly splitting audio duration.

    This is only scaffolding for segment-level inference. It does not inspect
    speech content and must not be treated as forced alignment.
    """

    duration = _audio_duration_seconds(audio_path)
    words = _prompt_words(prompt_text)
    phones = [str(phone).strip() for phone in (canonical_phones or []) if str(phone).strip()]

    if phones:
        boundaries = _split_evenly(duration, len(phones))
        word_count = max(len(words), 1)
        segments = []
        for index, ((start, end), phone) in enumerate(zip(boundaries, phones)):
            word = words[min(int(index * word_count / len(phones)), word_count - 1)] if words else None
            segments.append(
                build_alignment_segment(
                    index=index,
                    segment_type="phone",
                    phone=phone,
                    word=word,
                    start=start,
                    end=end,
                )
            )
        granularity = "phone"
    else:
        fallback_words = words or [Path(audio_path).stem]
        boundaries = _split_evenly(duration, len(fallback_words))
        segments = [
            build_alignment_segment(
                index=index,
                segment_type="word",
                phone=None,
                word=word,
                start=start,
                end=end,
            )
            for index, ((start, end), word) in enumerate(zip(boundaries, fallback_words))
        ]
        granularity = "word"

    return build_alignment_result(
        segments=segments,
        method="fallback_even_split",
        note=FALLBACK_ALIGNMENT_NOTE,
        metadata={
            "audio_duration_seconds": round(duration, 3),
            "granularity": granularity,
            "prompt_text": prompt_text,
            "fallback_alignment": True,
        },
    )
